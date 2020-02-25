# hyner
[English](/README.md) | Korean

hyner 는 KoBERT 기반으로 개발한 한국어 개체명 인식기입니다.

## 실행방법
### 필요한 페키지
```
PyTorch 0.4 or higher
scikit-learn
tqdm
MXNet
```
We highly recommned the conda virtual environment. And for PyTorch 0.4, we've tested that only torch0.4 + cuda9.2 can work. Otherwise you will get a "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS" error.
```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```
### 데모
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
## 데이터셋
Please refer to this link:
[Dataset](https://github.com/kmounlp/NER)

Put the "말뭉치 - 형태소_개체명" folder under `data/NER-master` directory.
## 학습
### 준비
이 링크를 참고하세요: [KoBERT Model file](https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params)

Download the model file and put it under `/kobert_model` directory.

### 학습 시작
```
$ python train.py --fp16 --lr_schedule
```
We highly recommend using NVIDIA's Automatic Mixed Precision (AMP) for acceleration.
Install the [APEX](https://github.com/NVIDIA/apex) first and then turn on the "-fp16" option.
## 성능
### 평가기준
There are several stantard to evaluate the performance of a multi-class classification model like NER.
First the simplest criteria is global accuracy. If we've got the confusion matrix, 

`global accuracy = confusion_matrix.trace()/confusion_matrix.sum()`

But it doesn't reflect the accuracy of every class's accuracy. Meanwhile there are micro f1 score and macro f1 score. In this project, we consider macro f1 score the most, and micro f1 score and global accuray at the same time.
### 결과
| Model | macro f1 score |
| ------------ | ------------- |
| BiLSTM-lr0.005-bs200 | 0.8096 |
| BiLSTM_CRF-lr0.005-bs200 | 0.8289 |
| KobertOnly-lr5e-05-bs200 | 0.8909 |
| KobertCRF-lr5e-05-bs200 | 0.8940  |

## 알으로 할 것
### Users Dictionary
Still in developement. we will complete this function in the future.
### 다른 모델
We are doing the evaluation of BERT multilingual cased model. 
