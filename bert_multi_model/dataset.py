from __future__ import absolute_import, division, print_function, unicode_literals

import codecs
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Tuple, Callable, List
import pickle
import json
from tqdm import tqdm
from collections import OrderedDict
import re

from pathlib import Path
from vocab import Vocabulary
from pad_sequence import keras_pad_fn, my_pad


class NamedEntityRecognitionDataset(Dataset):
    def __init__(self, train_data_dir: str, vocab, maxlen=30, model_dir=Path('data_in')) -> None:
        """
        :param train_data_in:
        :param transform_fn:
        """
        self.model_dir = model_dir

        list_of_total_source_str, list_of_total_target_str = self.load_data(train_data_dir=train_data_dir)
        self.create_ner_dict(list_of_total_target_str)
        self._corpus = list_of_total_source_str
        self._label = list_of_total_target_str
        
        self.vocab = vocab
        self.maxlen = maxlen
        self.pad = my_pad
        with open(self.model_dir / "ner_to_index.json", 'rb') as f:
            self.ner_to_index = json.load(f)
            
    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # preprocessing
        # str -> id -> cls, sep -> pad
        
        token_ids_with_cls_sep, tokens = self.transform_source_fn(self._corpus[idx].lower())
        #print(token_ids_with_cls_sep, tokens)
        list_of_ner_ids, list_of_ner_label = self.transform_target_fn(self._label[idx], tokens)
        #print(list_of_ner_ids, list_of_ner_label) #[ 0 13 14 14  4  7] ['B-PER', 'I-PER', 'I-PER', 'O', 'B-NOH', 'I-NOH']一开始是0，之后不到30的用1填充
        
        assert len(token_ids_with_cls_sep) == len(list_of_ner_ids)
        #print(list_of_ner_ids)
        x_input = torch.tensor(token_ids_with_cls_sep).long()
        #print(x_input.size()) #30
        token_type_ids = torch.tensor(len(x_input) * [0])
        label = torch.tensor(list_of_ner_ids).long()
        #print("x_input.size(): ", x_input.size())
        #print("token_type_ids: ", token_type_ids)
        #print("label.size(): ", label.size())

        return x_input, token_type_ids, label
    
    def transform_source_fn(self, text):
        # text = "첫 회를 시작으로 13일까지 4일간 총 4회에 걸쳐 매 회 2편씩 총 8편이 공개될 예정이다."
        # label_text = "첫 회를 시작으로 <13일:DAT>까지 <4일간:DUR> 총 <4회:NOH>에 걸쳐 매 회 <2편:NOH>씩 총 <8편:NOH>이 공개될 예정이다."
        # text = "트래버 모리슨 학장은 로스쿨 학생과 교직원이 바라라 전 검사의 사법정의에 대한 깊이 있는 지식과 경험으로부터 많은 것을 배울 수 있을 것이라고 말했다."
        # label_text = "<트래버 모리슨:PER> 학장은 로스쿨 학생과 교직원이 <바라라:PER> 전 검사의 사법정의에 대한 깊이 있는 지식과 경험으로부터 많은 것을 배울 수 있을 것이라고 말했다."
        tokens = list(text)
        
        #print(tokens)
        tokens_list = [self.vocab.token_to_idx.get(n, 100) for n in tokens]
        tokens_list.insert(0, self.vocab.token_to_idx['[CLS]'])
        tokens_list.append(self.vocab.token_to_idx['[SEP]'])
        token_ids_with_cls_sep = self.pad(tokens_list, pad_id=self.vocab.PAD_ID, maxlen=self.maxlen)
        #print(self.vocab.PAD_ID, token_ids_with_cls_sep) #PAD_ID=0, 
        '''
        for n in tokens:
            try:
                self.vocab.token_to_idx[n]
            except KeyError:
                print(n)
        '''

        return token_ids_with_cls_sep, tokens


    def transform_target_fn(self, label_text, tokens):

        regex_ner = re.compile('<(.+?):[A-Z]{3}>') # NER Tag가 2자리 문자면 {3} -> {2}로 변경 (e.g. LOC -> LC) 인경우
        regex_filter_res = regex_ner.finditer(label_text)

        list_of_ner_tag = []
        list_of_ner_text = []
        list_of_tuple_ner_start_end = []

        count_of_match = 0
        for match_item in regex_filter_res:
            #print(match_item) #<_sre.SRE_Match object; span=(17, 24), match='<정:PER>'>
            ner_tag = match_item[0][-4:-1]  # <4일간:DUR> -> DUR
            ner_text = match_item[1]  # <4일간:DUR> -> 4일간
            #print(match_item[0], match_item[1]) #<10년:NOH> 10년
            start_index = match_item.start() - 6 * count_of_match  # delete previous '<, :, 3 words tag name, >'
            end_index = match_item.end() - 6 - 6 * count_of_match

            list_of_ner_tag.append(ner_tag)
            list_of_ner_text.append(ner_text)
            list_of_tuple_ner_start_end.append((start_index, end_index))
            count_of_match += 1

        #print(label_text, tokens, list_of_tuple_ner_start_end)
        list_of_ner_label = []
        entity_index = 0
        is_entity_still_B = True
        
        for index, tup in enumerate(tokens):
            token = tup

            if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
                index += 1  # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

            if entity_index < len(list_of_tuple_ner_start_end):
                start, end = list_of_tuple_ner_start_end[entity_index]

                if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                    is_entity_still_B = True
                    entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end) else entity_index
                    start, end = list_of_tuple_ner_start_end[entity_index]

                if start <= index and index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                    entity_tag = list_of_ner_tag[entity_index]
                    if is_entity_still_B is True:
                        entity_tag = 'B-' + entity_tag
                        list_of_ner_label.append(entity_tag)
                        is_entity_still_B = False
                    else:
                        entity_tag = 'I-' + entity_tag
                        list_of_ner_label.append(entity_tag)
                else:
                    is_entity_still_B = True
                    entity_tag = 'O'
                    list_of_ner_label.append(entity_tag)

            else:
                entity_tag = 'O'
                list_of_ner_label.append(entity_tag)


        # ner_str -> ner_ids -> cls + ner_ids + sep -> cls + ner_ids + sep + pad + pad .. + pad
        list_of_ner_ids = [self.ner_to_index['[CLS]']] + [self.ner_to_index[ner_tag] for ner_tag in list_of_ner_label] + [self.ner_to_index['[SEP]']]
        #print(list_of_ner_ids)
        list_of_ner_ids = self.pad(list_of_ner_ids, pad_id=self.vocab.PAD_ID, maxlen=self.maxlen)
        #print(list_of_ner_ids, '\n')
        return list_of_ner_ids, list_of_ner_label
    
    def create_ner_dict(self, list_of_total_target_str):
        """ if you want to build new json file, you should delete old version. """

        if not os.path.exists(self.model_dir / "ner_to_index.json"):
            regex_ner = re.compile('<(.+?):[A-Z]{3}>')
            list_of_ner_tag = []
            for label_text in list_of_total_target_str:
                regex_filter_res = regex_ner.finditer(label_text)
                for match_item in regex_filter_res:
                    ner_tag = match_item[0][-4:-1]
                    if ner_tag not in list_of_ner_tag:
                        list_of_ner_tag.append(ner_tag)

            ner_to_index = {"[CLS]":0, "[SEP]":1, "[PAD]":2, "[MASK]":3, "O": 4}
            for ner_tag in list_of_ner_tag:
                ner_to_index['B-'+ner_tag] = len(ner_to_index)
                ner_to_index['I-'+ner_tag] = len(ner_to_index)

            # save ner dict in data_in directory
            with open(self.model_dir / 'ner_to_index.json', 'w', encoding='utf-8') as io:
                json.dump(ner_to_index, io, ensure_ascii=False, indent=4)
            self.ner_to_index = ner_to_index
        else:
            self.set_ner_dict()

    def set_ner_dict(self):
        with open(self.model_dir / "ner_to_index.json", 'rb') as f:
            self.ner_to_index = json.load(f)

    def load_data(self, train_data_dir):
        list_of_file_name = [file_name for file_name in os.listdir(train_data_dir) if '.txt' in file_name]
        list_of_full_file_path = ["{}/{}".format(train_data_dir, file_name) for file_name in list_of_file_name]
        print("num of files: ", len(list_of_full_file_path))

        list_of_total_source_str, list_of_total_target_str = [], []
        for i, full_file_path in enumerate(list_of_full_file_path):
            list_of_source_no, list_of_source_str, list_of_target_str = self.load_data_from_txt(file_full_name=full_file_path)
            #print(list_of_source_no, list_of_source_str, list_of_target_str)
            #list_of_source_no ['1\n', '2\n', '3\n', '4\n', '5\n', '6\n', '7\n', '8\n', '9\n', '10\n']
            #list_of_source_str ['每句话', '每句话', '每句话']
            #list_of_target_str ['有tag的每句话', '有tag的每句话', '有tag的每句话']
            list_of_total_source_str.extend(list_of_source_str)
            list_of_total_target_str.extend(list_of_target_str)
        assert len(list_of_total_source_str) == len(list_of_total_target_str)

        return list_of_total_source_str, list_of_total_target_str

    def load_data_from_txt(self, file_full_name):
        with codecs.open(file_full_name, "r", "utf-8") as io:
            lines = io.readlines()

            # parsing에 문제가 있어서 아래 3개 변수 도입!
            prev_line = ""
            save_flag = False
            count = 0
            sharp_lines = []

            for line in lines:
                if prev_line == "\n" or prev_line == "":
                    save_flag = True
                if line[:3] == "## " and save_flag is True:
                    count += 1
                    sharp_lines.append(line[3:])
                if count == 3:
                    count = 0
                    save_flag = False

                prev_line = line
            
            #print(sharp_lines)
            list_of_source_no, list_of_source_str, list_of_target_str = sharp_lines[0::3], sharp_lines[1::3], sharp_lines[2::3]
        return list_of_source_no, list_of_source_str, list_of_target_str


if __name__ == '__main__':


    text = "첫 회를 시작으로 13일까지 4일간 총 4회에 걸쳐 매 회 2편씩 총 8편이 공개될 예정이다."
    label_text = "첫 회를 시작으로 <13일:DAT>까지 <4일간:DUR> 총 <4회:NOH>에 걸쳐 매 회 <2편:NOH>씩 총 <8편:NOH>이 공개될 예정이다."
    ner_formatter = NamedEntityRecognitionFormatter()
    token_ids_with_cls_sep, tokens, prefix_sum_of_token_start_index = ner_formatter.transform_source_fn(text)
    ner_formatter.transform_target_fn(label_text, tokens, prefix_sum_of_token_start_index)

