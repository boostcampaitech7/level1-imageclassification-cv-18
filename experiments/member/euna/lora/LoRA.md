# LoRA

논문 : https://arxiv.org/abs/2106.09685

깃헙 :  https://github.com/microsoft/LoRA

## LoRA (Low-Rank Adaption)

딥러닝 모델을 효율적으로 미세 조정하는 방법.

딥러닝 모델은 수백만 개의 숫자로 이루어져 있는데, 

이미 학습된 모델을 사용할 때, 모든 매개변수를 다시 학습시키는 것은 시간이 많이 걸리고 계산 비용이 많이 듦.

→ Fine-tuning

모델 전체를 수정하는 대신,

매개변수의 일부를 더 작은 차원으로 줄여서 (저차원으로 변환)

학습해야 할 양을 크게 줄이는 방식.

매개변수의 일부를 저차언 행렬로 분해하는 방법으로

헹렬의 차원을 줄임으로써 모델의 일부만 조정할 수 있게 하고, 

전체 모델의 나머지 부분은 Freeze 시킴.

1. 모델의 특정 가중치 행렬 W를 두개의 저차원 행렬 A와 B로 분해
    
    ( W → A X B)
    
2. 학습할 때는 A와 B를 업데이트하고, 나머지 큰 가중치 행렬은 유지
3. 최종적으로는 이 두 저차원 행렬을 결합해 원래 가중치 행렬을 복원하는 식으로 모델을 파인 튜닝함.

## LoraConfig

- **r**
    
    LoRA가 적용되는 가중치 행렬을 얼마나 저차원으로 줄일 것인지 결정하는 값.
    
    값이 클수록 더 많은 정보가 반영되지만, 계산량도 증가함.
    
- **lora_alpha**
    
    LoRA가 가중치 업데이트를 적용할 때 스케일링을 조절.
    
    학습하는 동안, 저차원 근사로 인해 생긴 가중치 변화에 얼마나 크게 영향을 줄 것인지 결정.
    
- **target_modules**
    
    LoRA가 적용될 모델 내의 특정 부분(모듈)을 지정함.
    
    성능에 중요한 영향을 미치는 모듈을 선택함.
    
- **lora_dropout**
    
    LoRA가 적용된 부분에서 과적합을 방지하기 위해 신경망의 일부 뉴런을 학습 중 무작위로 비활성화하는 기법.
    
- **bias**
    
    LoRA를 적용할 때 bias를 처리하는 방법.
    
    (”none”: bias 무시, “all”: 모든 bias에 LoRA 적용, “lora_only”: LoRA가 적용된 모듈에만)
    
- **modules_to_save**
    
    LoRA 적용 후 학습이 끝난 모델의 어떤 부분을 저장할지 결정.
    

```python
lora_config = LoraConfig(
        r=args.lora_r,  # LoRA의 rank
        lora_alpha=args.lora_alpha,
        target_modules = ["attn.qkv", "attn.proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["head"],
    )

    try:
        peft_model = get_peft_model(model, lora_config)
        print("LoRA가 성공적으로 적용되었습니다.")
    except ValueError as e:
        print(f"오류 발생: {e}")
        print("모델 구조를 다시 확인하고 target_modules를 조정해주세요.")

    model = get_peft_model(model, lora_config)
    print(f"학습 가능한 파라미터: {count_parameters(model)}")

```