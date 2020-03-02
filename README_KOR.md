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

하지만 각 class의 정확도를 반영하지 못합니다.

또한, multi class classification문제에서
micro f1 score와 macro f1 score가 가장 많이 쓰는 기준이다.

micro f1 score는 각 class를 구분하지 않고
전체의 TP (True Positive), FP (False Positive), FN (False Negative) 를 계산하고,
```
precision = TP/ (TP + FP)
recall = TP/( TP + FN)
micro f1 score = 2 * precision * recall/(precision + recall)
```
macro f1 score는 위에 계산식으로 우선 각 class의 f1 score를 계산하고
F1, F1, F1,... 그리고 average를 한다. 가령 class가 n개 있는 경우:
```
macro f1 score = (F1 + F1 + F1,...)/n
```
본 프로젝트에서는 macro f1 score를 이용해서 평가합니다.
### 결과
| Model | macro f1 score |
| ------------ | ------------- |
| BiLSTM-lr0.005-bs200 | 0.8096 |
| BiLSTM_CRF-lr0.005-bs200 | 0.8289 |
| KobertOnly-lr5e-05-bs200 | 0.8909 |
| KobertCRF-lr5e-05-bs200 | 0.8940  |

## 앞으로 할 것
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
| Model | macro f1 score |
| ------------ | ------------- |
| BiLSTM-lr0.005-bs200 | 0.8385 |
