from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pickle
import argparse
import torch
import gluonnlp as nlp

from model import Config, BertMulti_CRF
from pad_sequence import keras_pad_fn
from vocab import Vocabulary
from pathlib import Path

def main(parser):

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_config = Config(json_path=model_dir / 'config.json')

    '''
    file = open('vocab.txt', 'r') 
    count = 0
    token_to_idx = {}
    for line in file:
        line = line.rstrip()
        token_to_idx[line] = count
        count += 1
    vocab = Vocabulary(token_to_idx=token_to_idx)
    
    # save vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    '''
    
    # load vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
        
    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # Model
    model = BertMulti_CRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    model_state_dict = torch.load('{}/BertMulti_CRF-lr5e-05-bs256/model.state'.format(model_dir))
    model.load_state_dict(model_state_dict)
    
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    decoder_from_res = DecoderFromNamedEntitySequence(index_to_ner=index_to_ner)

    while(True):
        input_text = input("문장을 입력하세요: ")
        input_text = input_text.rstrip('\n')
        words = input_text.split()
        input_text = input_text.replace(' ', '')
        tokens = list(input_text)
        list_of_input_ids = [vocab.token_to_idx.get(n, 100) for n in tokens]
        list_of_input_ids.insert(0, vocab.token_to_idx['[CLS]'])
        list_of_input_ids.append(vocab.token_to_idx['[SEP]'])
        
        space_list = []
        for word in words:
            if len(word) == 1:
                space_list.append(1)
                continue
            else:
                space_list.append(1)
                for a in range(len(word)-1):
                    space_list.append(0)
                
        #print(space_list)
        #print(list_of_input_ids)
        x_input = torch.tensor(list_of_input_ids).long()
        x_input = x_input.view(1, -1)
        #print(x_input.size())
        if torch.cuda.is_available():
            x_input = x_input.cuda() 
            
        ## for multi-bert crf
        list_of_pred_ids = model(x_input)
        #print(list_of_pred_ids)
        list_of_ner_word, decoding_ner_sentence = decoder_from_res(input_token_list=tokens, \
                            list_of_pred_ids=list_of_pred_ids[0][1:-1], space_list=space_list)
        
        print("list_of_ner_word:", list_of_ner_word)
        print("decoding_ner_sentence:", decoding_ner_sentence[1:])


class DecoderFromNamedEntitySequence():
    def __init__(self, index_to_ner):
        self.index_to_ner = index_to_ner

    def __call__(self, input_token_list, list_of_pred_ids, space_list):
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids]
        #print(pred_ner_tag)
        #print("len: {}, input_token:{}".format(len(input_token), input_token))
        #print("len: {}, pred_ner_tag:{}".format(len(pred_ner_tag), pred_ner_tag))

        tokens = input_token_list
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag, sentence_with_tag = "", "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word, "tag": prev_entity_tag})

                    sentence_with_tag += ':' + entity_tag + ">"
                                   
                entity_word = tokens[i]
                prev_entity_tag = entity_tag
                
                if space_list[i] == 1:
                    sentence_with_tag += ' <' + tokens[i]
                else:
                    sentence_with_tag += '<' + tokens[i]
                    
            elif "I-"+entity_tag in pred_ner_tag_str:
                
                entity_word += tokens[i]
                
                if space_list[i] == 1:
                    sentence_with_tag += ' ' + tokens[i]
                else:
                    sentence_with_tag += tokens[i]
                    
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word, "tag":entity_tag})
                    sentence_with_tag += ':' + entity_tag + ">"
                    
                if space_list[i] == 1:
                    sentence_with_tag += ' ' + tokens[i]
                else:
                    sentence_with_tag += tokens[i]
                    
                entity_word, entity_tag, prev_entity_tag = "", "", "" 
       
        return list_of_ner_word, sentence_with_tag


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./', help="Directory containing config.json of model")

    main(parser)