# hyner
English | [Korean](/README_KOR.md)

hyner is a Korean named entity recognizer (NER) based on KoBERT.
## Usage
### Requirement
```
PyTorch 0.4 or higher
scikit-learn
tqdm
pandas
MXNet == 1.5.0 or higher
gluonnlp == 0.8.1
sentencepiece
pytorch_transformers
pytorch-crf
konlpy #Need Java 
```
We highly recommned the conda virtual environment. And for PyTorch 0.4, we've tested that only torch0.4 + cuda9.2 can work. Otherwise you will get a "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS" error.
```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```
### Inference Demo
First, please download our pretrained model: [Model file](https://drive.google.com/drive/folders/1aiq8m1kh5esD3tdmGjJlBddG5-Sgrb9k?usp=sharing)  
Put it under `/kobert_model/KobertCRF-lr5e-05-bs200` directory. And it's very easy to see the result from a simple demo.
```
$ python inference.py
```
For example, if you input "도연이는 2018년에 골드만삭스에 입사했다.", you can get:
```
list_of_ner_word: [{'word': ' 도연이', 'tag': 'PER'}, {'word': ' 2018년에', 'tag': 'DAT'}, {'word': ' 골드만삭스', 'tag': 'ORG'}]
decoding_ner_sentence: <도연이:PER>는 <2018년에:DAT> <골드만삭스:ORG>에 입사했다.
```
## Dataset
### NER tag
PER: Person  
LOC: Location  
ORG: Organization  
POH: Others  
DAT: Date  
TIM: Time  
DUR: Duration  
MNY: Money  
PNT: Proportion  
NOH: Other measure words  

Please refer to this link:
[Dataset](https://github.com/kmounlp/NER)  
Put the "말뭉치 - 형태소_개체명" folder under `data/NER-master` directory.
## Training
### Preperation
Please refer to this link: [KoBERT Model file](https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params)  
Download the model file and put it under `/kobert_model` directory.

### Start training
```
$ python train.py --fp16 --lr_schedule
```
We highly recommend using NVIDIA's Automatic Mixed Precision (AMP) for acceleration.
Install the [APEX](https://github.com/NVIDIA/apex) first and then turn on the "-fp16" option.
## Performance
### Criteria
There are several stantard to evaluate the performance of a multi-class classification model like NER.
First the simplest criteria is global accuracy. If we've got the confusion matrix, 

`global accuracy = confusion_matrix.trace()/confusion_matrix.sum()`

But it doesn't reflect the accuracy of every class's accuracy. And in multi class classification's situation,
micro f1 score and macro f1 score are more frequently used for evaluation.
micro f1 score doesn't distinguish classes, instead it calculates the overall TP (True Positive), FP (False Positive), FN (False Negative):
```
precision = TP/ (TP + FP)
recall = TP/( TP + FN)
micro f1 score = 2 * precision * recall/(precision + recall)
```
While macro f1 score uses the same formula to calculate every class's f1 scores F11, F12, F13,... 
and then average them. In the situation of n classes, macro f1 score is like this:
```
macro f1 score = (F11 + F12 + F13,...)/n
```
In this project, we consider macro f1 score the most, and micro f1 score and global accuray at the same time.

The results in 25 epochs (with early stop, patience = 10) are as follows.
| Model | macro f1 score |
| ------------ | ------------- |
| BiLSTM-lr0.005-bs200 | 0.8096 |
| BiLSTM_CRF-lr0.005-bs200 | 0.8289 |
| KobertOnly-lr5e-05-bs200 | 0.8909 |
| KobertCRF-lr5e-05-bs200 | 0.8940  |

## In the future
### Users Dictionary
Still in the process of developement.  
Users dictionary has this format:  

후룬베얼	LOC  
알리바바	ORG  
컨버스	ORG  
유튜브	ORG  
추자현	PER  
언더아머	ORG  

For example, if you input "미국 해군의 플레처급 구축함 DD-509 '컨버스'에 대한 내용은 플레처급 구축함 문서를 참조하십시오."
then will get the result:
```
list_of_ner_word: [{'word': '미국해군의', 'tag': 'ORG'}, {'word': 'DD-509', 'tag': 'POH'}, {'word': '컨버스', 'tag': 'ORG'}]
decoding_ner_sentence: <미국 해군의:ORG> 플레처급 구축함 <DD-509:POH> '<컨버스:ORG>'에 대한 내용은 플레처급 구축함 문서를 참조하십시오.
```
### Other model's evaluation
We are doing the evaluation of BERT multilingual cased model. 
```
cd bert_multi_model
```
And then start training like before
```
$ python train.py --fp16 --lr_schedule
```
In this case, we used the BertTokenizer and the BertModel of 
"bert-base-multilingual-cased" from pytorch_transformers package.

| Model | macro f1 score |
| ------------ | ------------- |
| BertMulti_CRF-lr5e-05-bs256 | 0.8776 |

So we discovered that the pretrained KoBERT model outperformed BERT-multi model.
### RESTful API
We've finished the development the RESTful API, using the Docker.
