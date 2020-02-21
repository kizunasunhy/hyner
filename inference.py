from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pickle
import argparse
import torch
import gluonnlp as nlp

from model import Config, KobertCRF, KobertBiLSTMCRF
from vocab_tokenizer import Tokenizer
from pad_sequence import keras_pad_fn
from vocab_tokenizer import Vocabulary, Tokenizer

from gluonnlp.data import SentencepieceTokenizer
from pathlib import Path

def main(parser):

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    tok_path = "./tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    vocab_file = './kobert_model/kobertvocab_f38b8a4d6d.json'
    vocab_of_gluonnlp = nlp.vocab.BERTVocab.from_json(open(vocab_file, 'rt').read())
    token_to_idx = vocab_of_gluonnlp.token_to_idx
    vocab = Vocabulary(token_to_idx=token_to_idx)  
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
    
    # load ner_to_index.json
    with open(model_dir / "ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # Model
    model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    model_state_dict = torch.load('{}/KobertCRF-lr5e-05-bs200/model.state'.format(model_dir))
    model.load_state_dict(model_state_dict)
    
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=tokenizer, index_to_ner=index_to_ner)

    while(True):
        input_text = input("입력하세요: ")
        list_of_input_ids = tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
        #print(list_of_input_ids)
        x_input = torch.tensor(list_of_input_ids).long()
        
        if torch.cuda.is_available():
            x_input = x_input.cuda() 
            
        ## for bert crf
        list_of_pred_ids = model(x_input)
        #print(list_of_pred_ids)
        list_of_ner_word, decoding_ner_sentence = decoder_from_res(input_text, list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
        print("list_of_ner_word:", list_of_ner_word)
        print("decoding_ner_sentence:", decoding_ner_sentence[6:-5])


class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, text, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        #print("len: {}, input_token:{}".format(len(input_token), input_token))
        #print("len: {}, pred_ner_tag:{}".format(len(pred_ner_tag), pred_ner_tag))

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        #tokens = input_token[1:-1]
        tokens = input_token
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag, sentence_with_tag = "", "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag})
                    print('fook')
                    sentence_with_tag += ':' + prev_entity_tag + ">"
                    
                    
                entity_word = input_token[i]
                prev_entity_tag = entity_tag
                
                if '▁' in tokens[i]:
                    sentence_with_tag += tokens[i].replace('▁', ' <')
                else:
                    sentence_with_tag += '<' + tokens[i] 
                    
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
                
                if '▁' in tokens[i]:
                    sentence_with_tag += tokens[i].replace('▁', ' ')
                else:
                    sentence_with_tag += tokens[i]
                    
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag})
                    sentence_with_tag += ':' + entity_tag + ">"
                    
                if '▁' in tokens[i]:
                    sentence_with_tag += tokens[i].replace('▁', ' ')
                else:
                    sentence_with_tag += tokens[i]
                    
                entity_word, entity_tag, prev_entity_tag = "", "", "" 
       
        return list_of_ner_word, sentence_with_tag


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./kobert_model', help="Directory containing config.json of model")

    main(parser)