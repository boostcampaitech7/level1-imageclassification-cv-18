## CyclicLR

Document : https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html

학습률을 주기적으로 증가하고 감소시키는 방식의 학습률 스케줄러

학습률을 일정한 주기 동안 최소값과 최대값 사이에서 오르고 내리게 함.

→ 학습률이 너무 작아 너무 일찍 수렴하는 문제 해결

```python
optimizer (Optimizer) – Wrapped optimizer.

base_lr (float or list) – Initial learning rate which is the lower boundary in the cycle for each parameter group.

max_lr (float or list) – Upper learning rate boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_lr - base_lr). The lr at any cycle is the sum of base_lr and some scaling of the amplitude; therefore max_lr may not actually be reached depending on scaling function.

step_size_up (int) – Number of training iterations in the increasing half of a cycle. Default: 2000

step_size_down (int) – Number of training iterations in the decreasing half of a cycle. If step_size_down is None, it is set to step_size_up. Default: None

mode (str) – One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above. If scale_fn is not None, this argument is ignored. Default: ‘triangular’

gamma (float) – Constant in ‘exp_range’ scaling function: gamma**(cycle iterations) Default: 1.0

scale_fn (function) – Custom scaling policy defined by a single argument lambda function, where 0 <= scale_fn(x) <= 1 for all x >= 0. If specified, then ‘mode’ is ignored. Default: None

scale_mode (str) – {‘cycle’, ‘iterations’}. Defines whether scale_fn is evaluated on cycle number or cycle iterations (training iterations since start of cycle). Default: ‘cycle’

cycle_momentum (bool) – If True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’. Default: True

base_momentum (float or list) – Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled inversely to learning rate; at the peak of a cycle, momentum is ‘base_momentum’ and learning rate is ‘max_lr’. Default: 0.8

max_momentum (float or list) – Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle amplitude (max_momentum - base_momentum). The momentum at any cycle is the difference of max_momentum and some scaling of the amplitude; therefore base_momentum may not actually be reached depending on scaling function. Note that momentum is cycled inversely to learning rate; at the start of a cycle, momentum is ‘max_momentum’ and learning rate is ‘base_lr’ Default: 0.9

last_epoch (int) – The index of the last batch. This parameter is used when resuming a training job. Since step() should be invoked after each batch instead of after each epoch, this number represents the total number of batches computed, not the total number of epochs computed. When last_epoch=-1, the schedule is started from the beginning. Default: -1

verbose (bool | str) – If True, prints a message to stdout for each update. Default: False.
```

- base_lr (필수)
    
    : 학습률의 최소값
    
- max_lr (필수)
    
    : 학습률의 최대값
    
- step_size_up (필수)
    
    : 학습률이 최소값에서 최대값까지 증가하는 단계 수
    
- mode
    
    : 학습률이 증가하고 감소하는 패턴을 설정하는 방식
    
    - triangular 
    : 학습률이 base_lr에서 max_lr까지 선형적으로 증가한 후 다시 base_lr로 선형적으로 감소.
    - triangular2
        
        : 매 반복 주기마다 학습률 변동 폭이 절반씩 줄어듭니다. 학습률이 점차적으로 안정화되는 효과를 줍니다.
        
    - exp_range
        
        : 학습률이 주기마다 `gamma` 비율로 지수적으로 감소
        
- cycle_momentum
    
    : 모멘텀을 사용하는 optimizer를 사용할 때 사용,
    
    기본값이 True이기에, Adam을 사용할 때는 False