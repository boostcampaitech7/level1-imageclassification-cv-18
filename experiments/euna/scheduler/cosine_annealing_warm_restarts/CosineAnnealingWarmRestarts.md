## CosineAnnealingWarmRestarts

Document : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html

일정 주기마다 학습률을 다시 리셋하여 높은 학습률로 시작할 수 있음.

→ 학습이 진행될수록 local minima에 갇히지 않고 최적화를 계속 진행.

```python
optimizer (Optimizer) – Wrapped optimizer.

T_0 (int) – Number of iterations until the first restart.

T_mult (int, optional) – A factor by which T_i increases after a restart. Default: 1.

eta_min (float, optional) – Minimum learning rate. Default: 0.

last_epoch (int, optional) – The index of the last epoch. Default: -1.

verbose (bool | str) – If True, prints a message to stdout for each update. Default: False.
```

- T_0 (필수)
    
    : 첫 번째 Warm Restart가 발생하기까지의 에포크 수, 첫 번째 주기의 길이 결정 
    
- T_mult
    
    : 첫 번째 주기 T_0 이후의 각 주기마다 주기 길이를 배수로 늘리기 위한 계수
    
    T_mult=2 → 주기가 이전 주기의 2배가 됨.
    
- eta_min
    
    : 학습률이 코사인 함수로 감소할 때 도달할 최소 학습률