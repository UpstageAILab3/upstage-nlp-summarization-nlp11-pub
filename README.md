# Dialogue Summarization | 일상 대화 요약
- Upstage AI Stages https://stages.ai/en/competitions/320/overview/description

## Team
![](https://github.com/UpstageAILab3/upstage-nlp-summarization-nlp11/blob/main/docs/images/image0.png)

## 0. Overview
### Environment
- AMD Ryzen Threadripper 3960X 24-Core Processor
- NVIDIA GeForce RTX 3090
- CUDA Version 12.2
- OS: Ubuntu 20.04
- Python
- IDE: Jupyter Notebook, VSCode


### 주요 라이브러리
- torch==1.9.0
- torchvision==0.10.0
- pandas==1.3.3
- numpy==1.21.2
- timm==0.4.12

### Requirements
- 데이터셋 처리 및 모델 학습을 위한 충분한 GPU 자원 : 모델을 효율적으로 학습시키기 위해 고성능 GPU가 필수적
- 다양한 데이터 증강 기법 적용 : 모델의 일반화 성능을 높이기 위한 데이터 증강
- 모델 학습 상태 추적 : 학습 과정 모니터링 및 실험 관리 도구인 wandb 설치 및 설정
- 필수 라이브러리 설치 : PyTorch, torchvision, pandas, numpy, timm 등을 설치하여 모델 학습 및 평가에 필요한 환경을 구성

## 1. Competiton Info

### Overview
* 목표 : 일상 대화에 대한 요약을 효과적으로 생성
* 평가 기준 : Rouge1, Rouge2, RougeL - 각 스코어를 계산한 후 문장 토큰화를 진행한 후 최종 Rouge Score 산출

#### 대화 타입(20종)
- 일상 대화
- 쇼핑
- 전화 통화
- 직업 면접
- 음식 주문
- 인터뷰
- 길 묻기
- 영화
- 사회적 만남
- 체크인
- 면접
- 날씨
- 여행
- 초대
- 주말 계획
- 정보 요청
- 휴가
- 길 안내
- 약속 잡기
- 비즈니스 대화


### Timeline

- 2024년 8월 29일 10:00 - 대회 시작일
- 2024년 9월 10일 19:00 - 최종 제출 마감일


## 2. Components

### Directory
```
│       
├── 03.Meta-Llama-3.1-8B.ipynb
├── Model_Solar.ipynb
├── ko-gemma-2-9B-summarize.ipynb
├── preprocessor.ipynb
├── t5.ipynb
│   
├── docs
│   └── 패스트캠퍼스_Upstage_AI_Lab_3기_NLP_경진대회_발표자료_11조.pdf
│   └── 패스트캠퍼스_Upstage_AI_Lab_3기_NLP_경진대회_현황공유판_11조.xlsx
│       
```

## 3. Data descrption

### Dataset overview
- 각 발화자를 구분하기 위해 #Person'N'# 사용
- 발화자의 대화가 끝나면 \n으로 구분
- 전화번호, 주소 등 개인 정보 마스킹

#### 학습 데이터
- 총 12,457개의 대화
- train.csv
  - fname : 대화 고유번호
  - dialogue : 최소 2명에서 최대 7명이 등장하는 대화
  - summary : 대화 요약문
  - topic : 대화 주제
  
#### 검증 데이터
- 총 499개의 대화
- dev.csv
  - fname : 대화 고유번호
  - dialogue : 최소 2명에서 최대 3명이 등장하는 대화
  - summary : 대화 요약문
  - topic : 대화 주제  

#### 테스트 데이터
- 총 499개의 대화
- dev.csv
  - fname : 대화 고유번호
  - dialogue : 최소 2명에서 최대 3명이 등장하는 대화


### EDA

- 대화 길이 분석 : 요약문을 생성하기 위한 입출력 길이를 결정하는 데 중요한 정보 파악

- 단어 빈도 분석 : 자주 등장하는 단어를 파악하여 데이터 증강 및 샘플링 전략을 수립

- 텍스트 분석 : 텍스트 정규화를 위해 포맷을 벗어난 텍스트 파악


### Data Processing

#### 데이터 증강
모델의 일반화 성능을 향상시키기 위해 다양한 데이터 증강 기법을 적용

##### EDA
4가지 규칙에 따라 단어 수준에서 변경하여 새로운 문장 생성
- SR : 유의어 교체
- RI : 임의 단어 삽입
- RS : 단어 위치 변경
- RD : 임의 단어 삭제

##### AEDA
- 다양한 문장부호를 원문에 임의 위치에 추가

##### Back Translation
- 제공된 데이터셋을 한국어->영어->한국어로 역번역하여 다양한 데이터 추가

##### 외부 데이터셋 활용
- 공개되어 있는 대화 말뭉치의 포맷을 대회의 데이터셋과 동일하게 변경하여 일반화 성능 높일 수 있는 데이터 확보


## 4. Modeling

### Model descrition

사용된 주요 사전 학습 모델

- digit82/kobart-summarization
- eenzeenee/t5-base-korean-summarization
- meta-llama/Meta-Llama-3.1-8B
- upstage/SOLAR-10.7B-v1.0
- rtzr/ko-gemma-2-9b-it


### Modeling Process

- 데이터 증강 : 모델의 일반화 성능을 향상시키기 위해 다양한 증강 기법을 통해 데이터셋을 확장
- 파인 튜닝 : 사전 학습된 모델을 불러오고, 태스크에 맞는 추론하도록 추가 학습
- 학습 루프 수행 : 에포크 단위로 모델을 학습시키고, 학습 상태를 wandb로 모니터링


## 5. Result
  
### Leader Board
- Midterm Rouge Score 46.6249 - 1위
- Final Rouge Score 44.6438 - 1위

### Presentation
- [패스트캠퍼스_Upstage_AI_Lab_3기_NLP_경진대회_발표자료_11조.pdf](/docs/패스트캠퍼스_Upstage_AI_Lab_3기_NLP_경진대회_발표자료_11조.pdf)
  
  
### Meeting Log
- [패스트캠퍼스_Upstage_AI_Lab_3기_NLP_경진대회_현황공유판_11조.xlsx](/docs/패스트캠퍼스_Upstage_AI_Lab_3기_NLP_경진대회_현황공유판_11조.xlsx)

