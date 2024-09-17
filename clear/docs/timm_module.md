  

# 2022

  
[**timm/beitv2_base_patch16_224.in1k_ft_in22k**](https://huggingface.co/timm/beitv2_base_patch16_224.in1k_ft_in22k)

- 이미지 사이즈 : 224 x 224
- 2022.10.03 논문
- semantic-rich visual tokenizer를 마스크 예측의 재구성 타겟으로 사용
    - 마스킹된 이미지 모델링을 pixel level부터 semantic level 까지 promote하는 조직적인 방법을 제공
- 연속적인 의미 공간을 코드 압축으로 이산화하는 토큰을 훈련하기 위해 vector-quantized knowledge distillation 제안
- 전역 의미 표현을 향상

[**timm/convmixer_768_32.in1k**](https://huggingface.co/timm/convmixer_768_32.in1k)

- 이미지 사이즈 : 224x224
- ViT가 성능이 좋은 것은 더 좋은 트랜스포머 아키택처 때문? 패치 임베딩 때문?
- ConvMixer 제안
	- 패치에서 직접 입력으로 작동
	- 공간 및 채널 차원의 혼합을 분리
	- 네트워크 전체에서 동일한 크기와 해상도 n을 유지
	- ViT, MLP-Mixer 및 일부 변형 능가, ResNet같은 기존 비전 모델 능가

# 2021

[**timm/coat**](https://huggingface.co/timm/coat_lite_mini.in1k)

- 이미지 사이즈 : 224x224
- 2021.08.26 논문
- coat_lite, coat 해당
- Co-scale과 conv-attentional이 있는 이미지 트랜스포머 classifier
    - co-scale 매커니즘
        - 개별 스케일에서 트랜스포머의 인코더 브랜치의 무결성을 유지
        - 서로 다른 스케일에서 학습한 표현이 서로 효과적으로 통신할 수 있도록 함
        - 이 매커니즘을 실현하기 위해 일련의 직렬 / 병렬 블록 설계
    - Conv-attentional 매커니즘
        - 효율적 컨볼루션과 유사한 구현
        - 인수 분해 주의력 모듈 (factorized attention module)에서 상대적인 위치 임베딩 공식 구현
    - CoaT
        - 풍부한 멀티 스케일 및 문맥 모델링 기능 제공
- 물체 감지/ 인스턴스 분할을 잘함.
- 다운스트림 컴퓨터 비전 작업에 적용
- **rw 버전은 Ross Wightman이 직접 학습시킨 버전임.**
- 참고 : [Maxxvit와 결합한 버전](https://huggingface.co/timm/coatnet_nano_rw_224.sw_in1k) - 2022 vit이다.
    - coatnet_0_rw_224
    - coatnet_1_rw_224
    - coatnet_bn_0_rw_224
    - coatnet_nano_rw_224
    - coatnet_rmlp_1_rw_224
    - coatnet_rmlp_2_rw_224
    - coatnet_rmlp_nano_rw_224
- 참고 : coatnet대신 coatNext 쓴 버전
    - coatnext_nano_rw_224


[**timm/botnet26t_256.c1_in1k**](https://huggingface.co/timm/botnet26t_256.c1_in1k)

- 이미지 사이즈 : 256x256
- 2021.08.02 논문
- paper configuration을 준수하지 않았음, 합리적인 훈련 시간을 가지게 되었고, self-attention block의 빈도가 줄었음.
- 공간 컨볼루션을 ResNet의 마지막 병목 블록에서 글로벌 셀프 어텐션으로 대체
    - 인스턴스 분할 및 객체 감지의 기준선을 크게 개선
    - 지연 시간의 오버헤드를 최소화 -> 매개 변수를 줄임
    - EfficientNet모델보다 compute 시간이 1.64배 빠름. 정확도 증가

[**timm/convit_base.fb_in1k**](https://huggingface.co/timm/convit_base.fb_in1k)

- 이미지 사이즈 : 224x224
- 2021.06.10 논문
- CNN과 ViT를 결합한 모델
- GPSA 도입
	- 부드러운 컨볼루션 귀납적 편향을 장착
	- 위치 자체 attention의 한 형태
- GPSA 레이어 초기화
	- 컨볼루션 레이어의 로컬리티 모방
	- 각 주의 헤드에 위치 대 콘텐츠 정보 t에 대한 주의를 조절하는 게이팅 매개변수 조절 -> 로컬리티를 피할 수 있는 자유 부여
- ImageNet에서 DeiT 능가, 훨씬 향상된 샘플 효율성
  

[**timm/cait_xxs36_224.fb_dist_in1k**](https://huggingface.co/timm/cait_xxs36_224.fb_dist_in1k)

- 이미지 사이즈 : 224x224
- 2021.04.07 논문
- 이미지 트랜스포머의 최적화
- 조기에 포화 되지 않는 모델 생성

  

# 2020

  
  
  

# 2017

  

**timm/bat_resnext26ts.ch_in1k**

- 이미지 사이즈 : 256x256
- 2017.4.11 논문
- 동일한 토폴로지(topology)를 가진 변환 집합을 집계하는 빌딩 블록을 반복하여 구성
- 하이퍼 파라미터가 몇 개 안됨. 균일하고 다중 분기(multi-branch)된 아키텍처
- 깊이, 너비의 차원 이외에도 카디널리티(변환 집합의 크기)를 필수 요소로 노출
    - 카디널리티 증가 -> 분류 정확성 증가
    - 용량을 늘릴 때 카디널리티를 증가 시키는 것이 깊게 하거나 넓히는 것보다 더 효과적

  

# 2015

  

[**timm/inception_v3.tf_adv_in1k**](https://huggingface.co/timm/inception_v3.tf_adv_in1k)

- 이미지 사이즈 : 299x299
- 2015.12.11 논문
- CNN 기반
- 공격적인 정규화와 적절히 인수분해된 컨벌루션을 통해 네트워크를 확장.