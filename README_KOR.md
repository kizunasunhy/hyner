# hyner
[English](/README.md) | Korean

hyner는 KoBERT 기반으로 개발한 한국어 개체명 인식기입니다.

## 실행방법
### 필요한 페키지
```
PyTorch 0.4 or higher
scikit-learn
tqdm
MXNet == 1.5.0 or higher
gluonnlp == 0.8.1
tensorflow == 1.14.0
sentencepiece
pytorch-crf
```
Conda 가상환경을 권장합니다. PyTorch 0.4 같은 경우에는, we've tested that only torch0.4 + cuda9.2 can work. Otherwise you will get a "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS" error.
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
### 개체명 인식 태그
총 8개의 태그  
PER: 사람이름  
LOC: 지명  
ORG: 기관명  
POH: 기타  
DAT: 날짜  
TIM: 시간  
DUR: 기간  
MNY: 통화  
PNT: 비율  
NOH: 기타 수량표현    
이 링크를 참고하세요:
[Dataset](https://github.com/kmounlp/NER)  
"말뭉치 - 형태소_개체명"폴더를 `data/NER-master`에 놓으세요.
## 학습
### 준비
이 링크를 참고하세요: [KoBERT Model file](https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params)  
다운로드 하고 `/kobert_model`경로에 놓으세요.

### 학습 시작
```
$ python train.py --lr_schedule
```
NVIDIA의 Automatic Mixed Precision (AMP) GPU가속화를 권장합니다.
[APEX](https://github.com/NVIDIA/apex) 설치하고 "-fp16" 옥션을 켜주세요.
## 성능
### 평가기준
개체명 인식은 benchmark 여러가지 있습니다. 우선 tag 종류는 보통 5가지만 있는데(PER LOC ORG MISC O)
데터셋 따라서 더 구체적으로 구분할 수도 있습니다. 예를 들어, 본 프로젝트에서 쓰는tag가 11가지 있습니다.  
또한, 정확도를 평가하는 기준도 여러가지가 있습니다. 우선 가장 간단한 기준은 global accuray입니다.  
가령 confusion matrix 있으면,  
```
global accuracy = confusion_matrix.trace()/confusion_matrix.sum()
```
하지만 각 class의 정확도를 반영하지 못합니다.  
그리고, multi class classification문제에서
micro f1 score와 macro f1 score가 가장 많이 쓰는 기준이다.  
micro f1 score는 각 class를 구분하지 않고
전체의 TP (True Positive), FP (False Positive), FN (False Negative) 를 계산하고,
```
precision = TP/ (TP + FP)
recall = TP/( TP + FN)
micro f1 score = 2 * precision * recall/(precision + recall)
```
macro f1 score는 위에 계산식으로 우선 각 class의 f1 score를 계산하고
F11, F12, F13,... 그리고 average를 한다. 가령 class가 n개 있는 경우:
```
macro f1 score = (F11 + F12 + F13,...)/n
```
본 프로젝트에서는 macro f1 score를 이용해서 평가합니다.
### 결과
25 epoch (early stop 있는 경우, patience = 10) 의 결과가 아래와 같습니다.
| Model | macro f1 score |
| ------------ | ------------- |
| BiLSTM-lr0.005-bs200 | 0.8096 |
| BiLSTM_CRF-lr0.005-bs200 | 0.8289 |
| KobertOnly-lr5e-05-bs200 | 0.8909 |
| KobertCRF-lr5e-05-bs200 | 0.8940  |

## 앞으로 할 것
### Users Dictionary
Users dictionary가 이런 형식을 가지고 있습니다.

후룬베얼 LOC  
알리바바 ORG  
컨버스 ORG  
유튜브 ORG  
추자현 PER  
언더아머 ORG  

예르르 들어, "미국 해군의 플레처급 구축함 DD-509 '컨버스'에 대한 내용은 플레처급 구축함 문서를 참조하십시오." 입력하면,
```
list_of_ner_word: [{'word': '미국해군의', 'tag': 'ORG'}, {'word': 'DD-509', 'tag': 'POH'}, {'word': '컨버스', 'tag': 'ORG'}]
decoding_ner_sentence: <미국 해군의:ORG> 플레처급 구축함 <DD-509:POH> '<컨버스:ORG>'에 대한 내용은 플레처급 구축함 문서를 참조하십시오.
```
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

KoBERT 모델과 비교해보면 역시 character 기반이 좀 부족하다고 볼 수 있습니다.
### RESTful API
개발할 예정입니다..
