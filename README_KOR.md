# hyner
[English](/README.md) | Korean

hyner는 KoBERT 기반으로 개발한 한국어 개체명 인식기입니다.

## 실행방법
### 필요한 페키지
```
PyTorch 0.4 or higher
scikit-learn
tqdm
MXNet
```
Conda 가상환경을 권장합니다. And for PyTorch 0.4, we've tested that only torch0.4 + cuda9.2 can work. Otherwise you will get a "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS" error.
```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```
### 데모
pretrained 모델을 다운 받으세요: [Model file](https://drive.google.com/drive/folders/1aiq8m1kh5esD3tdmGjJlBddG5-Sgrb9k?usp=sharing)

`/kobert_model/KobertCRF-lr5e-05-bs200`에 놓으시고 데모 스크립트 실행하세요.
```
$ python inference.py
```
예를 들어서 "도연이는 2018년에 골드만삭스에 입사했다." 입력하면:
```
list_of_ner_word: [{'word': ' 도연이', 'tag': 'PER'}, {'word': ' 2018년에', 'tag': 'DAT'}, {'word': ' 골드만삭스', 'tag': 'ORG'}]
decoding_ner_sentence: <도연이:PER>는 <2018년에:DAT> <골드만삭스:ORG>에 입사했다.
```
## 데이터셋
이 링크를 참고하세요:
[Dataset](https://github.com/kmounlp/NER)

"말뭉치 - 형태소_개체명"폴더를 `data/NER-master`에 놓으세요.
## 학습
### 준비
이 링크를 참고하세요: [KoBERT Model file](https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params)

다운로드 하고 `/kobert_model`경로에 놓으세요.

### 학습 시작
```
$ python train.py --fp16 --lr_schedule
```
NVIDIA의 Automatic Mixed Precision (AMP) GPU가속화를 권장합니다.
[APEX](https://github.com/NVIDIA/apex) 설치하고 "-fp16" 옥션을 켜주세요.
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
아직 개발 중입니다..
### 다른 모델
BERT-multilingual-cased 모델도 개발하고 있습니다..
```
cd bert_multi_model
```
전처럼 학습시킵니다:
```
$ python train.py --fp16 --lr_schedule
```
BERT-multi-cased 모델의 경우는 토큰말고 character 레벨을 바탕으로 개발한겁니다.
