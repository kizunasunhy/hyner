from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import numpy as np
from konlpy.tag import Twitter
from collections import Counter
from threading import Thread
from pad_sequence import keras_pad_fn

class Vocabulary(object):
    """Vocab Class"""

    def __init__(self, token_to_idx=None):

        self.token_to_idx = {}
        self.idx_to_token = {}
        self.idx = 0

        self.PAD = self.padding_token = "[PAD]"
        self.START_TOKEN = "<S>"
        self.END_TOKEN = "<T>"
        self.UNK = "[UNK]"
        self.CLS = "[CLS]"
        self.MASK = "[MASK]"
        self.SEP = "[SEP]"
        self.SEG_A = "[SEG_A]"
        self.SEG_B = "[SEG_B]"
        self.NUM = "<num>"

        self.cls_token = self.CLS
        self.sep_token = self.SEP

        self.special_tokens = [self.PAD,
                               self.START_TOKEN,
                               self.END_TOKEN,
                               self.UNK,
                               self.CLS,
                               self.MASK,
                               self.SEP,
                               self.SEG_A,
                               self.SEG_B,
                               self.NUM]
        self.init_vocab()

        if token_to_idx is not None:
            self.token_to_idx = token_to_idx
            self.idx_to_token = {v: k for k, v in token_to_idx.items()}
            self.idx = len(token_to_idx) - 1

            # if pad token in token_to_idx dict, get pad_id
            if self.PAD in self.token_to_idx:
                self.PAD_ID = self.transform_token2idx(self.PAD)
            else:
                self.PAD_ID = 0

    def init_vocab(self):
        for special_token in self.special_tokens:
            self.add_token(special_token)
        self.PAD_ID = self.transform_token2idx(self.PAD)

    def __len__(self):
        return len(self.token_to_idx)

    def to_indices(self, tokens):
        return [self.transform_token2idx(X_token) for X_token in tokens]

    def add_token(self, token):
        if not token in self.token_to_idx:
            self.token_to_idx[token] = self.idx
            self.idx_to_token[self.idx] = token
            self.idx += 1

    def transform_token2idx(self, token, show_oov=False):
        try:
            return self.token_to_idx[token]
        except:
            if show_oov is True:
                print("key error: " + str(token))
            token = self.UNK
            return self.token_to_idx[token]

    def transform_idx2token(self, idx):
        try:
            return self.idx_to_token[idx]
        except:
            print("key error: " + str(idx))
            idx = self.token_to_idx[self.UNK]
            return self.idx_to_token[idx]

    def build_vocab(self, list_of_str, threshold=1, vocab_save_path="./data_in/token_vocab.json",
                    split_fn=Twitter().morphs):
        """Build a token vocab"""

        def do_concurrent_tagging(start, end, text_list, counter):
            for i, text in enumerate(text_list[start:end]):
                text = text.strip()
                text = text.lower()

                try:
                    tokens_ko = split_fn(text)
                    # tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
                    counter.update(tokens_ko)

                    if i % 1000 == 0:
                        print("[%d/%d (total: %d)] Tokenized input text." % (
                            start + i, start + len(text_list[start:end]), len(text_list)))

                except Exception as e:  # OOM, Parsing Error
                    print(e)
                    continue

        counter = Counter()

        num_thread = 4
        thread_list = []
        num_list_of_str = len(list_of_str)
        for i in range(num_thread):
            thread_list.append(Thread(target=do_concurrent_tagging, args=(
                int(i * num_list_of_str / num_thread), int((i + 1) * num_list_of_str / num_thread), list_of_str,
                counter)))

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        # vocab_report
        print(counter.most_common(10))  # print most common tokens
        tokens = [token for token, cnt in counter.items() if cnt >= threshold]

        for i, token in enumerate(tokens):
            self.add_token(str(token))

        print("len(self.token_to_idx): ", len(self.token_to_idx))

        import json
        with open(vocab_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_idx, f, ensure_ascii=False, indent=4)

        return self.token_to_idx


def main():
    print("안녕하세요")


if __name__ == '__main__':
    main()