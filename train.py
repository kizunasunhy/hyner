from gluonnlp.data import SentencepieceTokenizer
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_transformers import AdamW, WarmupLinearSchedule
from tqdm import tqdm, trange
from apex import amp
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler  import LambdaLR

from vocab_tokenizer import Vocabulary, Tokenizer
from pad_sequence import keras_pad_fn
from model import Config, KobertCRF, KobertBiLSTMCRF, KobertOnly, BiLSTM, BiLSTM_CRF
from dataset import NamedEntityRecognitionFormatter, NamedEntityRecognitionDataset
from utils import set_seed, CheckpointManager, SummaryManager

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
import gluonnlp as nlp
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main(parser):
    '''
    #to create bert_config
    kobert_model_dir = '/home/kizunasunhy/my_bert_ner/kobert_model/bert_model.json'
    with open(kobert_model_dir, 'w', encoding='utf-8') as f:
        json.dump(bert_config, f, ensure_ascii=False, indent=4)
    '''    
        
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)   
    model_config = Config(json_path=model_dir / 'config.json')
    model_config.learning_rate = args.lr
    
    tok_path = './tokenizer_78b3253a26.model'
    
    ptr_tokenizer = SentencepieceTokenizer(tok_path)
    '''
    file = open('{}/vocab.txt'.format(model_dir), 'r') 
    count = 0
    dic = {}
    for line in file:
        line = line.rstrip()
        dic[line] = count
        count += 1
    token_to_idx = dic
    '''
    vocab_file = args.model_dir + '/kobertvocab_f38b8a4d6d.json'
    vocab_of_gluonnlp = nlp.vocab.BERTVocab.from_json(open(vocab_file, 'rt').read())
    token_to_idx = vocab_of_gluonnlp.token_to_idx
    model_config.vocab_size = len(token_to_idx)
    vocab = Vocabulary(token_to_idx=token_to_idx)        
    print("len(token_to_idx): ", len(token_to_idx))
    
    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
    ner_formatter = NamedEntityRecognitionFormatter(vocab=vocab, tokenizer=tokenizer, maxlen=model_config.maxlen, model_dir=model_dir)
    
    # Train & Val Datasets
    cwd = Path.cwd()
    data_in = cwd / "data"
    train_data_dir = "{}/NER-master/말뭉치 - 형태소_개체명".format(data_in)
    tr_ds = NamedEntityRecognitionDataset(train_data_dir=train_data_dir, model_dir=model_dir)
    tr_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn, transform_target_fn=ner_formatter.transform_target_fn)
    tr_dl = DataLoader(tr_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=2, drop_last=False)

    val_data_dir = "{}/NER-master/validation_set".format(data_in)
    val_ds = NamedEntityRecognitionDataset(train_data_dir=val_data_dir, model_dir=model_dir)
    val_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn, transform_target_fn=ner_formatter.transform_target_fn)
    val_dl = DataLoader(val_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=2, drop_last=False)
 
    # Model
    #model = KobertCRF(config=model_config, num_classes=len(tr_ds.ner_to_index))
    #model = KobertCRFViz(config=model_config, num_classes=len(tr_ds.ner_to_index))
    #model = KobertBiLSTMCRF(config=model_config, num_classes=len(tr_ds.ner_to_index))
    #model = KobertOnly(config=model_config, num_classes=len(tr_ds.ner_to_index))
    #model = BiLSTM(config=model_config, num_classes=len(tr_ds.ner_to_index))
    model = BiLSTM_CRF(config=model_config, num_classes=len(tr_ds.ner_to_index))
    model.train()
    
    # optim
    train_examples_len = len(tr_ds)
    val_examples_len = len(val_ds)
    print("num of train: {}, num of val: {}".format(train_examples_len, val_examples_len))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # num_train_optimization_steps = int(train_examples_len / model_config.batch_size / model_config.gradient_accumulation_steps) * model_config.epochs
    t_total = len(tr_dl) // model_config.gradient_accumulation_steps * model_config.epochs
    #optimizer = AdamW(optimizer_grouped_parameters, lr=model_config.learning_rate, eps=model_config.adam_epsilon)
    optimizer = torch.optim.Adam(model.parameters(), model_config.learning_rate)
    if args.lr_schedule:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=model_config.warmup_steps, t_total=t_total)
        #lmbda = lambda epoch: 0.5
        #scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    
    #Create model output directory
    output_dir = os.path.join(model_dir, '{}-lr{}-bs{}'.format(model.name, model_config.learning_rate, model_config.batch_size))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(output_dir)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    '''
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    '''
    model.to(device)
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if args.continue_train:
        revert_to_best(model, optimizer, output_dir)
        logging.info("==== continue training: %s ====", '{}-lr{}-bs{}' \
                    .format(model.name, model_config.learning_rate, model_config.batch_size))
        
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(tr_ds))
    logger.info("  Num Epochs = %d", model_config.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", model_config.batch_size)
    logger.info("  Gradient Accumulation steps = %d", model_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    log_file = open('{}/log.tsv'.format(output_dir), 'at')
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('epoch', 'train loss', 'eval_loss', 'eval global accuracy', \
                                              'micro_f1_score', 'macro_f1_score', 'learning_rate'), file=log_file)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_epoch = 0
    best_steps = 0
    patience = args.patience
    f_scores = []
    model.zero_grad()
    set_seed()
    criterion = nn.CrossEntropyLoss()
    
    train_begin = datetime.now()
    '''
    train_iterator = trange(int(model_config.epochs), desc="Epoch")  
    for _epoch, _ in enumerate(train_iterator):
    '''
    for _epoch in range(model_config.epochs):
        #epoch_iterator = tqdm(tr_dl, desc="Iteration")
        epoch_iterator = tr_dl
        epoch = _epoch
        
        for step, batch in enumerate(epoch_iterator):
            
            model.train()
            #print(batch)
            
            x_input, token_type_ids, y_real = map(lambda elm: elm.to(device), batch)
            #print(x_input.size(), token_type_ids.size(), y_real.size()) #都是batch_size*max_len
            #print(y_real)
            if model.name == "KobertOnly":
                y_out = model(x_input, token_type_ids, y_real)
                y_out.requires_grad_()
                y_out.contiguous()
                y_real.contiguous()
                y_real_ = y_real.view(-1)
                y_out_ = y_out.view(-1, len(tr_ds.ner_to_index))
                loss = criterion(y_out_, y_real_)
                _, sequence_of_tags = F.softmax(y_out, dim=2).max(2)
            elif model.name == "BiLSTM":
                y_out = model(x_input, token_type_ids, y_real)
                y_out.requires_grad_()
                y_out.contiguous()
                y_real.contiguous()
                
                y_out1 = F.log_softmax(y_out, dim=2)
                y_out1 = y_out1.view(-1, len(tr_ds.ner_to_index))
                
                y_real_ = y_real.view(-1)
                mask = (y_real_ != 1).float()
                #print(len(mask))
                original_len = int(torch.sum(mask))
                #print(x_input[0], y_real[0], original_len, '\n')
                y_out1 = y_out1[range(y_out1.shape[0]), y_real_] * mask
                loss = -torch.sum(y_out1) / original_len
                
                _, sequence_of_tags = F.softmax(y_out, dim=2).max(2)
            else:
                log_likelihood, sequence_of_tags = model(x_input, token_type_ids, y_real)
                loss = -1 * log_likelihood
            
            '''
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            '''
            
            if model_config.gradient_accumulation_steps > 1:
                loss = loss / model_config.gradient_accumulation_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:      
                loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)
            tr_loss += loss.item()

            if (step + 1) % model_config.gradient_accumulation_steps == 0:
                optimizer.step()
                if args.lr_schedule:
                    scheduler.step()  # Update learning rate schedule
                    #print(scheduler.state_dict())
                model.zero_grad()
                global_step += 1

                with torch.no_grad():
                    sequence_of_tags = torch.tensor(sequence_of_tags).to(device)
                    #print(sequence_of_tags.size(), y_real.size())
                    mb_acc = (sequence_of_tags == y_real).float()[y_real != vocab.PAD_ID].mean()

                tr_acc = mb_acc.item()
                tr_loss_avg = tr_loss / global_step
                tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}

                if (step + 1) % 20 == 0:
                    logging.info('epoch : {}, global_step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}' \
                                 .format(epoch + 1, global_step, tr_summary['loss'], tr_summary['acc']))
                
                # evaluation and save model
                if model_config.logging_steps > 0 and global_step % model_config.logging_steps == 0:

                    eval_summary = evaluate(model, val_dl)
                    
                    f_scores.append(eval_summary['macro_f1_score'])
                    
                    # Save model checkpoint
                    summary = {'train': tr_summary, 'eval': eval_summary}
                    summary_manager.update(summary)
                    summary_manager.save('summary.json')

                    # Save
                    is_best = eval_summary["macro_f1_score"] >= best_dev_acc  # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)  
                    is_best_str = 'BEST' if is_best else '< {:.4f}'.format(max(f_scores))
                    logging.info('[Los trn]  [Los dev]  [global acc]  [micro f1]  [macro f1]     [global step]    [LR]')
                    logging.info('{:8.2f}  {:9.2f}  {:9.2f}  {:11.4f}  {:9.4f} {:4}  {:9}  {:14.8f}' \
                                 .format((tr_loss - logging_loss) / model_config.logging_steps, eval_summary['eval_loss'], \
                                         eval_summary['eval_global_acc'], eval_summary['micro_f1_score'], \
                                         eval_summary['macro_f1_score'], is_best_str, global_step, model_config.learning_rate))
                    print('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(epoch, tr_loss, \
                                                              eval_summary['eval_loss'], eval_summary['eval_global_acc'], \
                                                              eval_summary['micro_f1_score'], eval_summary['macro_f1_score'], \
                                                              model_config.learning_rate), file=log_file)
                    log_file.flush()
                    
                    logging_loss = tr_loss
                    
                    if is_best:
                        best_dev_acc = eval_summary["macro_f1_score"]
                        best_dev_loss = eval_summary["eval_loss"]
                        best_steps = global_step
                        best_epoch = epoch
                        #checkpoint_manager.save_checkpoint(state, 'best-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1, global_step, best_dev_acc))
                        #logging.info("Saving model checkpoint as best-epoch-{}-step-{}-acc-{:.3f}.bin".format(epoch + 1, global_step, best_dev_acc))
                        logging.info("Saving model at epoch{}, step{} in {}".format(epoch, global_step, output_dir))
                        torch.save(model.state_dict(), '{}/model.state'.format(output_dir))
                        torch.save(optimizer.state_dict(), '{}/optim.state'.format(output_dir))
                        patience = args.patience
                        
                    else:
                        revert_to_best(model, optimizer, output_dir)
                        patience -= 1
                        logging.info("==== revert to epoch[%d], step%d. F1 score: %.4f, patience: %d ====", \
                                     best_epoch, best_steps, max(f_scores), patience)
                        
                        if patience == 0:
                            break
                
        else:
            
            continue
            
        break
        
    #print("global_step = {}, average loss = {}".format(global_step, tr_loss / global_step))
    #print(ptr_tokenizer('안녕하세요 중국에서 온 손홍양입니다'))
    
    train_end = datetime.now()
    train_elapsed = elapsed(train_end - train_begin)
    logging.info('==== training time elapsed: %s, epoch: %s ====', train_elapsed, epoch)
        
    return global_step, tr_loss / global_step, best_steps

def elapsed(td_obj: timedelta) -> str:
        """
        string formatting for timedelta object
        Args:
            td_obj:  timedelta object
        Returns:
            string
        """
        seconds = td_obj.seconds
        if td_obj.days > 0:
            seconds += td_obj.days * 24 * 3600
        hours = seconds // 3600
        seconds -= hours * 3600
        minutes = seconds // 60
        seconds -= minutes * 60
        return '{}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    
def revert_to_best(model, optimizer, model_path):
    
    model_state_dict = torch.load('{}/model.state'.format(model_path))
    model.load_state_dict(model_state_dict)
    optim_state_dict = torch.load('{}/optim.state'.format(model_path))
    optimizer.load_state_dict(optim_state_dict)
    
def evaluate(model, val_dl, prefix="NER"):
    """ evaluate accuracy and return result """
    results = {}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0

    list_of_y_real = []
    list_of_pred_tags = []
    count_correct = 0
    count_correct1 = 0
    total_count = 0
    useless_tag = [1, 2, 3]
    dict_dir = './kobert_model/ner_to_index.json'
    criterion = nn.CrossEntropyLoss()
    
    with open(dict_dir, 'rb') as f:
            ner_to_index = json.load(f)  
    index_to_ner = {v: k for k, v in ner_to_index.items()}
    n_tags = len(ner_to_index)
    conf_matrix = np.zeros((len(ner_to_index), len(ner_to_index)), dtype=np.int32)
    
    for batch in tqdm(val_dl, desc="Evaluating"):
        model.train()
        x_input, token_type_ids, y_real = map(lambda elm: elm.to(device), batch)
        with torch.no_grad():
            inputs = {'input_ids': x_input,
                      'token_type_ids': token_type_ids,
                      'tags': y_real}
            if model.name == "KobertOnly":
                y_out = model(x_input, token_type_ids, y_real)
                y_out.contiguous()
                y_real.contiguous()
                y_real_ = y_real.view(-1)
                y_out_ = y_out.view(-1, n_tags)
                loss = criterion(y_out_, y_real_)
                _, sequence_of_tags = F.softmax(y_out, dim=2).max(2)
                eval_loss += loss.item()
            elif model.name == "BiLSTM":
                y_out = model(x_input, token_type_ids, y_real)
                y_out.contiguous()
                y_real.contiguous()
                
                y_out_ = F.log_softmax(y_out, dim=2)
                y_out_ = y_out_.view(-1, n_tags)
                
                y_real_ = y_real.view(-1)
                mask = (y_real_ > 1).float()
                original_len = int(torch.sum(mask))
                y_out_ = y_out_.view(-1, n_tags)
                y_out_ = y_out_[range(y_out_.shape[0]), y_real_] * mask
                loss = -torch.sum(y_out_) / original_len
                eval_loss += loss.item()
                _, sequence_of_tags = F.softmax(y_out, dim=2).max(2)
            else:
                log_likelihood, sequence_of_tags = model(**inputs)
                eval_loss += -1 * log_likelihood.float().item()
        nb_eval_steps += 1
        

        y_real = y_real.to('cpu')
        sequence_of_tags = torch.tensor(sequence_of_tags).to('cpu')
        count_correct += (sequence_of_tags == y_real).float()[y_real != 1].sum()  # 0,1,2,3 -> [CLS], [SEP], [PAD], [MASK] index
        total_count += len(y_real[y_real != 1])
        
        y_real = y_real.view(1, -1)
        y_real = torch.squeeze(y_real)
        sequence_of_tags = sequence_of_tags.view(1, -1)
        sequence_of_tags = torch.squeeze(sequence_of_tags)
        y_real_np = np.array(y_real)
        #print(y_real.size())
        y_pred_np = np.array(sequence_of_tags)
        index = [i for i, j in enumerate(y_real.tolist()) if j not in useless_tag]
        index_np = np.array(index)
        list_of_y_real.extend(y_real_np[index_np])
        #print(y_real)
        #print(y_real_np[index_np], '\n')
        list_of_pred_tags.extend(y_pred_np[index_np])
        '''
        for seq_elm in y_real.tolist():
            list_of_y_real += seq_elm
        for seq_elm in sequence_of_tags.tolist():
            list_of_pred_tags += seq_elm
        '''
        for i, (y_pred, y_real) in enumerate(zip(y_real_np[index_np], y_pred_np[index_np])):
            conf_matrix[y_real, y_pred] += 1
                
    #print(len(list_of_y_real)) #22241

    #Confusion matrix with precision and recall accuracy for each tag
    print(("{: >1}{: >7}{: >7}%s{: >9}" % ("{: >6}" * (n_tags-4))).format("ID", "NAME", "Total", 
            *([index_to_ner[i] for i in range(4, n_tags)] + ["Rec Per"])))
    for i in range(4, n_tags):
        print(("{: >1}{: >7}{: >7}%s{: >9}" % ("{: >6}" * (n_tags-4))).format(
            str(i), index_to_ner[i], str(conf_matrix[i].sum()),
            *([conf_matrix[i][j] for j in range(4, n_tags)] +
                ["%.3f" % (conf_matrix[i][i] * 100. / max(1, conf_matrix[i].sum()))])
        ))
    
    column = np.zeros((1, n_tags), dtype=np.float32)
    for i in range(4, n_tags):
        if sum(conf_matrix[r][i] for r in range(n_tags))==0:
            column[0][i] = 0
        else:
            column[0][i] = round(100 * conf_matrix[i][i]/(sum(conf_matrix[r][i] for r in range(n_tags))), 3)
                  
    print(("{: >7}{: >10}%s" % ("{: >6}" * (n_tags-4))).format(
        "Pre Per", "", *([('%.2f' % column[0][i]) for i in range(4, n_tags)])))
               
    # Global accuracy
    global_acc = 100. * conf_matrix.trace() / max(1, conf_matrix.sum())
    #logger.info("Global accuracy is %i/%i=%.4f%%" % (conf_matrix.trace(), conf_matrix.sum(), global_acc))
        
    assert len(list_of_y_real) == len(list_of_pred_tags)
    micro_f1 = f1_score(list_of_y_real, list_of_pred_tags, average="micro")
    macro_f1 = f1_score(list_of_y_real, list_of_pred_tags, average="macro")
    #logger.info("micro fq score: {:.4}, macro f1 score: {:.4}".format(micro_f1, macro_f1))
    
    eval_loss = eval_loss / nb_eval_steps
    acc = (count_correct / total_count).item()  # tensor -> float 
    #print(acc1)
    result = {"eval_global_acc": global_acc, "eval_loss": eval_loss, "micro_f1_score": micro_f1, "macro_f1_score": macro_f1}
    results.update(result)

    return results
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', default=False, action='store_true', help='use fp16 training')
    parser.add_argument('--data_dir', default='./data', help="Directory containing config.json of data")
    #parser.add_argument('--model_dir', default='/home/kizunasunhy/my_bert_ner/bert_multi_model', help="Directory containing config.json of model")
    parser.add_argument('--model_dir', default='./kobert_model', help="Directory containing config.json of model")
    parser.add_argument('--patience', type=int, default=10, help="Patience if macro f1 score is not increasing")
    parser.add_argument('--continue_train', default=False, action='store_true', help="Continue training.")
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_schedule', default=False, action='store_true', help='Using learning rate scheduler')
    main(parser)