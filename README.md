<div align="right">
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/boostcampaitech7/level1-imageclassification-cv-18&count_bg=%23C6D2FF&title_bg=%23555555&icon=&icon_color=%23FFFFFF&title=hits&edge_flat=false"/></a>
</div>

# 딥하조
![대회 타이틀](https://github.com/user-attachments/assets/a3a97f02-1e01-4ea0-85cc-4d6f204df5cd)

- 2024.09.11 ~ 2024.09.26
- ImageNet Sketch 이미지 데이터 분류
- 1st Prize 🏆
- Naver Connect & Upstage 주관 대회
- [main code](./main)
- [프로젝트 리포트 (README)](./Sketch%20Data%20Multi-Classification%20Project%20Report.pdf)

## Leaderboard
![리더보드](https://github.com/user-attachments/assets/05f98560-85fb-43b7-b272-bef54f9a97e1)



## 팀원 소개

| [![](https://avatars.githubusercontent.com/chan-note)](https://github.com/chan-note) | [![](https://avatars.githubusercontent.com/Donghwan127)](https://github.com/Donghwan127) | [![](https://avatars.githubusercontent.com/batwan01)](https://github.com/batwan01) | [![](https://avatars.githubusercontent.com/taehan79-kim)](https://github.com/taehan79-kim) | [![](https://avatars.githubusercontent.com/nOctaveLay)](https://github.com/nOctaveLay)  | [![](https://avatars.githubusercontent.com/Two-Silver)](https://github.com/Two-Silver)  |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| [임찬혁](https://github.com/chan-note)                  | [서동환](https://github.com/Donghwan127)                  | 🦇[박지완](https://github.com/batwan01)          | [김태한](https://github.com/taehan79-kim)                  | 🐈[임정아](https://github.com/nOctaveLay)                  | 🐡[이은아](https://github.com/Two-Silver)                  |

## 대회 소개
![image](https://github.com/user-attachments/assets/e889ae72-c64f-48bb-95f0-ce7c73d56e4c)

Sketch 이미지 분류 경진대회는 주어진 데이터를 활용하여 모델을 제작하고 어떤 객체를 나타내는지 분류하는 대회입니다.

Computer Vision에서는 다양한 형태의 이미지 데이터가 활용되고 있습니다. 이 중, 비정형 데이터의 정확한 인식과 분류는 여전히 해결해야 할 주요 과제로 자리잡고 있습니다. 특히 사진과 같은 일반 이미지 데이터에 기반하여 발전을 이루어나아가고 있습니다.

****하지만 일상의 사진과 다르게 스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.****

이러한 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높이는데에 초점을 두었습니다. 이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.

## 사용된 데이터셋 정보

- **데이터셋 이름**: Sketch Data (ImageNet Sketch)
- **출처**: [Sketch Data 다운로드 링크](https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz)

### 데이터셋 설명

원본 **ImageNet Sketch** 데이터셋은 50,889개의 이미지 데이터로 구성되어 있으며, 1,000개의 객체에 대해 각각 대략 50개의 이미지를 포함하고 있습니다. 이 데이터셋은 일반적인 객체들의 핸드 드로잉 이미지로 구성되어 있으며, 실제 객체를 대표하는 다양한 스타일과 특징을 보여줍니다.

이번 Sketch 이미지 분류 경진대회에서 제공되는 데이터셋은 **네이버 커넥트재단 부스트캠프 AI Tech**에서 원본 데이터를 직접 검수하고 정제한 것입니다. 1,000개의 클래스 중 이미지 수량이 많은 상위 500개의 객체를 선정하였으며, 총 25,035개의 이미지 데이터가 포함되어 있습니다. 해당 이미지 데이터는 다음과 같이 구성됩니다.
- **학습 데이터**: 15,021개
- **Private & Public 평가 데이터**: 10,014개

```bash
data/
│
├── sample_submission.csv
├── test.csv
├── train.csv
│
├── test/
│   ├── 0.JPEG
│   ├── 1.JPEG
│   ├── 2.JPEG
│   ├── ...
│
├── train/
│   ├── n01443537/
│   │   ├── sketch_0.JPEG
│   │   ├── sketch_1.JPEG
│   │   ├── sketch_2.JPEG    
│   │   ├── ...
│   │
│   ├── n01484850/
│   │   ├── sketch_0.JPEG
│   │   ├── sketch_1.JPEG
│   │   ├── sketch_2.JPEG    
│   │   ├── ...
│   │   
│   ├── ... 
```

### License
이 데이터셋은 [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)에 따라 사용됩니다. 자세한 내용은 [라이선스 링크](https://creativecommons.org/licenses/by/4.0/)에서 확인할 수 있습니다.

이 라이선스 하에서, 여러분은 다음과 같은 권리를 가집니다:
- **공유** — 어떤 매체나 형식으로도 데이터를 복사, 배포할 수 있습니다.
- **변경** — 데이터를 리믹스, 변형, 수정하고 상업적 목적으로도 사용할 수 있습니다.
단, 다음의 조건을 준수해야 합니다:
- **저작자 표시** — 적절한 출처를 제공하고, 라이선스 링크를 명시하며, 변경 여부를 표시해야 합니다.
자세한 내용은 [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)에서 확인할 수 있습니다.

## Project Timeline
- 프로젝트 타임라인
![프로젝트 타임라인](https://github.com/user-attachments/assets/82d524f8-79c1-4bbb-ab78-44b9220b8d8b)

- 단계별 성능 변화 
![accu_timeline](https://github.com/user-attachments/assets/e2109364-d711-40a8-b5e0-2ffb7330e616)

- 자세한 내용은 [프로젝트 리포트 (README)](./Sketch%20Data%20Multi-Classification%20Project%20Report.pdf) 를 참고해주세요.
  
## Model select
- backbone 모델 탐색
  
![image](https://github.com/user-attachments/assets/0302c586-42ae-492f-be48-292009b86f77)

- Fine-tuning layers
  - MLP-3 : Linear(1024, 1024)-ReLU()-Linear(1024, 500)
  - Linear :(1024, 500)

- Presiction method
  - train-validation split : train set을 한 번 나누어 성능을 평가하고 테스트 합니다.
  - 5-fold CV : 5-fold cross validation 방법을 통해 모델을 다섯 번 학습하고 voting 하여 테스트 합니다.

- 모델 선정
  - **EVA02-large**
  - **EVA-giant**
 
- 하이퍼파라미터 튜닝
  
<img width="715" alt="하이퍼파라미터튜닝" src="https://github.com/user-attachments/assets/b6e1b279-b541-434e-abba-5267f93bba9b">

## 문제 정의
- 문제 1 : 수직으로 뒤집힌 Sketch 이미지 예측에 대한 취약점을 발견했습니다.
  
  -> 해결방법 1 : **Augmentations**

  
- 문제 2 : Sketch 이미지 데이터의 특성에 따른 취약점을 발견했습니다.
  - 단순한 정보를 가지고 있지만, 같은 클래스라도 다양한 변형 발생해서 모델이 쉽게 혼란스러워집니다.
  - 초기 학습 단계에서 복잡한 변형이 포함된 데이터를 학습 시, 모델이 패턴을 제대로 학습하지 못합니다.
  - 데이터의 불규칙성과 디테일 부족으로 인해, 모델이 과적합 되거나, 다양한 변형에 잘 대응하지 못합니다.
    
    -> 해결방법 2 : **Curriculum learning**


## Augmentations
<img width="578" alt="증강 성능" src="https://github.com/user-attachments/assets/15b09b70-0a0d-4df0-a064-f6b4e20a6126">

- HorizontalFlip
- VerticalFlip
- Rotate
  
**- VerticalFlip (수직 뒤집기) 증강을 추가했을 때 성능이 좋아짐을 확인했습니다.**

## Curriculum learning 
- 데이터 증강 관점에서 Curriculum learning을 수행합니다.
  
| Epoch Range | 적용 증강                              |
|-------------|--------------------------------------|
| 0 - 5       | 없음                                  |
| 5 - 10      | 수평/수직 뒤집기, 회전                    |
| 10 - 15     | 수평/수직 뒤집기, 회전, Elastic 변형, Grid 왜곡 |

<img width="676" alt="curri" src="https://github.com/user-attachments/assets/527da330-fef8-42fc-8e55-366356908769">

- (orange : train)   (blue : test)
- 증강이 복잡해질 때마다, 정확도가 떨어지고 다시 올라가는 과정에서 모델이 보다 복잡한 패턴 학습합니다.
- 증강이 바뀌는 5 epoch 마다 Learning rate를 감소시켜 갑작스러운 변화로부터 학습을 안정화합니다.
- **최종적으로 단일 모델 최고 성능을 달성했습니다. (Accuracy : 0.9370)**



## Voting & Ensemble
- Soft Voting & Hard Voting
  
![image](https://github.com/user-attachments/assets/34e6350e-a600-451f-a148-ab25359eb4bc)

- Ensemble
  - Soft-Soft
  <img width="847" alt="Snipaste_2024-10-11_21-01-08" src="https://github.com/user-attachments/assets/12143ac2-88a5-4a29-bb42-0ca47834625f">

  - Soft-Hard
  <img width="804" alt="Snipaste_2024-10-11_21-01-19" src="https://github.com/user-attachments/assets/3923ba34-a779-40ee-b178-16ce783c5fb6">

  - Hard-Hard
  <img width="553" alt="Snipaste_2024-10-11_21-01-27" src="https://github.com/user-attachments/assets/395f99f5-2339-49b9-87e2-d245b9f2f3b3">

- Ensemble 성능 측정
<img width="803" alt="Snipaste_2024-10-11_21-02-35" src="https://github.com/user-attachments/assets/10d75c04-f6c0-44c0-8228-f9133998e50e">


- **Soft-Soft, Hard-Hard 앙상블 Method 모두 비슷하게 좋은 성능을 보입니다. (Accuracy : 0.939)**

- **다음 모델들을 Hard-Hard 앙상블한 모델이 가장 좋은 성능을 보입니다. (Accuracy : 0.94)**
  -  **EVA02-large-Linear**
  -  **EVA-giant-Linear**
  -  **EVA02-large-curriculum-mlp-3**





