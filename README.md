[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zHsKfIy0)
# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- AMD Ryzen Threadripper 3960X 24-Core Processor
- NVIDIA GeForce RTX 3090
- CUDA Version 12.2

### Requirements
pandas==2.1.4
numpy==1.23.5
wandb==0.16.1
tqdm==4.66.1
pytorch_lightning==2.1.2
transformers[torch]==4.35.2
rouge==1.0.1
jupyter==1.0.0
jupyterlab==4.0.9

## 1. Competiton Info

### Overview

Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다. 

### Timeline

- Aug 30, 2024 - Start Date
- Sep 12, 2024 - Final submission deadline

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- 이번 대회에 사용된 데이터는 모두 대화문 데이터이며 train data, validation data, test data는 각각 12,457개, 499개, 499개 입니다. 각각의 대화문은 최소 2명에서 최대 7명의 화자가 등장하며, 최소 2턴에서 최대 60턴까지 대화가 이어집니다.
- 대화문에서 발화자는 #Person"N"#으로 구분되어있습니다. 대화문에 존재하는 개인정보(예: 전화번호, 주소 등)는 다음과 같이 마스킹되어 있습니다. 예) 전화번호 -> #PhoneNumber#
- 대회 데이터셋: DialogSum Dataset: CC BY-NC-SA 4.0 license 단, 해당 데이터을 한국어로 번역하여 활용 원본: https://github.com/cylnlp/dialogsum

### Evaluation Metric

- Dialogue Summarization task에서는 여러 인물들이 나눈 대화 내용을 요약하는 문제입니다. 예측된 요약 문장을 3개의 정답 요약 문장과 비교하여 metric의 평균 점수를 산출합니다. 본 대회에서는 ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1, 총 3가지 종류의 metric으로부터 산출된 평균 점수를 더하여 최종 점수를 계산합니다.

- 해당 평가지표를 활용한 이유는 다음과 같습니다. DialogSum 데이터셋은 Multi-Reference Dataset으로 multi-reference에 대한 average를 보는 것이 중요합니다. 따라서 데이터셋의 특성에 맞추어 최종 점수 산출도 평균을 활용했습니다. 따라서, 3개의 정답 요약 문장의 metric 평균 점수를 활용하기에 metric 점수가 100점이 만점이 아니며, 3개의 정답 요약 문장 중 하나를 랜덤하게 선택하여 산출된 점수가 약 70점 정도임을 말씀드립니다.

- ROUGE는 텍스트 요약, 기계 번역과 같은 태스크를 평가하기 위해 사용되는 대표적인 metric입니다. 모델이 생성한 요약본 혹은 번역본을 사람이 만든 참조 요약본과 비교하여 점수를 계산합니다.

    ROUGE-Recall: 참조 요약본을 구성하는 단어들 중 모델 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수입니다.
    
    ROUGE-Precision: 모델 요약본을 구성하는 단어들 중 참조 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수입니다.

- ROUGE-N과 ROUGE-L은 비교하는 단어의 단위 개수를 어떻게 정할지에 따라 구분됩니다.

    ROUGE-N은 unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표입니다.
    
        ROUGE-1는 모델 요약본과 참조 요약본 간에 겹치는 unigram의 수를 비교합니다.
        
        ROUGE-2는 모델 요약본과 참조 요약본 간에 겹치는 bigram의 수를 비교합니다.
    
    ROUGE-L: LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정합니다. n-gram에서 n을 고정하지 않고, 단어의 등장 순서가 동일한 빈도수를 모두 세기 때문에 보다 유연한 성능 비교가 가능합니다.

- ROUGE-F1은 ROUGE-Recall과 ROUGE-Precisioin의 조화 평균입니다.

- 한국어 데이터 특성 상 정확한 ROUGE score 산출하기 위하여 문장 토큰화를 진행한 후 평가합니다. 한국어 형태소 분석기를 통해 의미를 갖는 최소한의 단위인 형태소 단위로 문장을 쪼갠 뒤 모델이 생성한 문장과 정답 문장을 비교하여 ROUGE score를 산출합니다.

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

- ref: https://dimensionstp.github.io/competition/upstage_NLP_competition/#modeling-method
- ref:
- 가설1: 대회의 DialogSum Dataset은 Solar로 영한 번역한 버젼이어서 summary 역시 어색한 말투임. 이 상태로 모델 진행해도 성능향상이 힘들다고 판단. Solar로 한영 back-translation을 진행함. 한,영 같이 모델 학습 후 영어로 추론하여, Solar로 영어를 한글로 번역한 것을 제출하는 전략을 취하고자 함.
    - train, dev, test -> solar로 영어번역 진행: 100 row per 1 batch로 xlsx파일 생성, 9분 소요, 총 시간 이틀 소요, 과금 $10 이내.
- 가설2: 빅데이터를 학습한 최신 instruct 모델일수록 성능이 좋을 것이다.
- 가설3: 이미 광범위한 어휘를 학습한 모델이므로 일반적인 채팅내용의 단어를 추가하여 학습하는 것은 불필요하며 기술적으로도 구현하기 복잡하다.- 
    - Model: Meta-Llama-3.1-8B-Instruct
    - encoder_max_len: 1000 max4071, mean620
    - decoder_max_len: 200 max960, mean143의 중간	
    - generation_max_length: 150, Summary텍스트길이는 Dialogue텍스트길이와 0.85, dialogue_turns와 0.54 상관관계 가짐. 인위적인 조정보다 모델에 맡기는게 나을 것이다.
    - per_device_train_batch_size: 32 일반적으로 32부터 셋팅한다.
    - LB: 

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
