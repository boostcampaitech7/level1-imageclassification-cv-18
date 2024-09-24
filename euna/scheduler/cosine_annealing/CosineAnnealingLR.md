## Consine Annealing
Document : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

 Learning Rate를 Cosine 곡선 패턴으로 점차적으로 감소시켜 더 부드러운 수렴을 달성함.

→ 모델이 local minima에 갇히는 것을 방지함.

에포크가 진행될수록 부드럽게 학습률을 줄여줌.

consine 그래프를 그리면서 learning rate가 진동하는 방식.

1. 학습률은 코사인 함수에 따라 초기 값에서 시작하며, T_max 에포크동안 점차적으로 최소 학습률인 eta_min까지 줄어듦.
2. 학습률이 T_max 동안 감소한 후,  그 값에 도달하면 더 이상 증가하지 않고 최소 학습률에 수렴한 상태로 유지.

```python
optimizer (Optimizer) – Wrapped optimizer.
T_max (int) – Maximum number of iterations.
eta_min (float) – Minimum learning rate. Default: 0.
last_epoch (int) – The index of last epoch. Default: -1.
verbose (bool | str) – If True, prints a message to stdout for each update. Default: False.
```

- T_max
    
    : 학습률이 최대값에서 최소값으로 감소하는 데 걸리는 에포크 수
    
- eta_min
    
    : 학습이 진행될수록 학습률이 이 값에 수렴함.