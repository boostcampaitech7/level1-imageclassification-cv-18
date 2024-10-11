# 코드 사용 방법
## 학습 및 테스트
### Fold를 이용한 커리큘럼 학습
```console
python3 curriculum_train_test.py
```

### Fold만 사용
```console
python3 train_test.py
```

### 옵션 사용
옵션을 사용하여 설정을 조정할 수 있습니다.\
모든 옵션 확인을 원한다면, -h 또는 --help 옵션을 사용하세요.

```console
python train_test.py --<옵션>
```
예를 들어,

```console
python train_test.py --epochs 20
```
여러 옵션을 다음과 같이 사용할 수 있습니다:

```console
python train_test.py --lr 0.001 --epochs 20
```

## 모델 수정
### 데이터
**Fold** - `base/dataloader.py`\
**Curriculum** - `curriculum/curriculum_dataloader.py`

## 모델
`train_test.py`와 `curriculum_train_test.py`는 `PyTorch` 모델, `Timm` 모델, `Custom` 모델을 지원합니다.\
`model/SimpleCNN.py`를 수정하여 커스텀 모델을 만들 수 있습니다.\
또한 `base/customize_layer.py` 또는 `curriculum/curriculum_dataloader.py`의 `ModelSelector`를 수정해야 합니다.

## 전이 학습
전이 학습을 원한다면, 간단히 `--pretrained` 옵션을 `True`로 추가하거나, `train_test.py` 또는 `curriculum_train_test.py`에서 default 값을 변경하세요.