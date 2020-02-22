# hyner
English | [Korean](/README_KOR.md)

hyner is a Korean named entity recognizer (NER) based on KoBERT.
## Usage
### Requirement
```
PyTorch 0.4 or higher
scikit-learn
tqdm
MXNet
```
We highly recommned the conda virtual environment. And for PyTorch 0.4, we've tested that only torch0.4 + cuda9.2 can work. Otherwise you will get a "RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS" error.
```
conda install pytorch=0.4.1 cuda92 -c pytorch
```
### Inference Demo
First, please download our pretrained model.
https://drive.google.com/open?id=1uJkbM3vT0kURxzIV7x8VHDkn9USkv2Ou
And put the `model.state` and `optim.state` in `/logdir/corpus.cut2.sdo0.0.emb100.lr0.001.lrd0.6.bs250`

We provided a trained model and it's very easy to see the result from a simple demo.
```
python inference.py
```
For example, if you input "도연이는 2018년에 골드만삭스에 입사했다.", you can get:
```
list_of_ner_word: [{'word': ' 도연이', 'tag': 'PER'}, {'word': ' 2018년에', 'tag': 'DAT'}, {'word': ' 골드만삭스', 'tag': 'ORG'}]
decoding_ner_sentence: <도연이:PER>는 <2018년에:DAT> <골드만삭스:ORG>에 입사했다.
```
## Dataset
Please refer to this link:
https://github.com/kmounlp/NER
## Training
### Preperation

### Start training
```
python train.py --fp16 --lr_schedule
```
We highly recommend using NVIDIA's Automatic Mixed Precision (AMP) for acceleration.
https://github.com/NVIDIA/apex
Install the apex first and then turn on the "-fp16" option.
## Users Dictionary
Still in developement. we will complete this function in the future.
