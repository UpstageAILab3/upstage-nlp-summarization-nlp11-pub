[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zHsKfIy0)
# Dialogue Summarization | 일상 대화 요약
- Upstage AI Stages [https://stages.ai/en/competitions/312/overview/description](https://stages.ai/en/competitions/320/overview/description)
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
- OS: Ubuntu 20.04
- Python
- IDE: Jupyter Notebook, VSCode

### Requirements
- _Write Requirements_

## 1. Competiton Info

### Overview

Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다. 
* 목표 : DialogueSum Solar로 번역한 한국어본을 활용, 일상 대화에 대한 요약을 생성하는 모델 개발
* 평가 기준 : 예측된 요약 문장을 3개의 정답 요약 문장과 비교하여 metric의 평균 점수를 산출
  - 본 대회에서는 ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1, 총 3가지 종류의 metric으로부터 산출된 평균 점수를 더하여 최종 점수를 계산
3개의 정답 요약 문장 중 하나를 랜덤하게 선택하여 산출된 점수가 약 70점 정도.
    - ROUGE-1는 모델 요약본과 참조 요약본 간에 겹치는 unigram의 수
    - ROUGE-2는 모델 요약본과 참조 요약본 간에 겹치는 bigram의 수
    - ROUGE-L: LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정
    - ROUGE-F1은 ROUGE-Recall과 ROUGE-Precisioin 조화평균
- 한국어 데이터 특성 상 정확한 ROUGE score 산출을 위해 문장 토큰화를 진행한 후 평가.


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

- 대화문에서 발화자는 #Person"N"#으로 구분되어있습니다. 대화문에 존재하는 개인정보(예: 전화번호, 주소 등)는 다음과 같이 마스킹되어 있습니다. 예) 전화번호 -> #PhoneNumber#
- 대회 데이터셋: DialogSum Dataset: CC BY-NC-SA 4.0 license 단, 해당 데이터을 한국어로 번역하여 활용 원본: https://github.com/cylnlp/dialogsum
- train : 12457
- dev : 499
- test : 499 (250, hidden-test : 249)


- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

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
