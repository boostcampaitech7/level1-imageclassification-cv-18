# pytorch-image-models

- [results-imagenet-a-clean](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet-a-clean.csv)



# 2022

  
[**timm/beitv2**](https://huggingface.co/timm/beitv2_base_patch16_224.in1k_ft_in22k)

- 이미지 사이즈 : 224 x 224
- 2022.10.03 논문
- semantic-rich visual tokenizer를 마스크 예측의 재구성 타겟으로 사용
    - 마스킹된 이미지 모델링을 pixel level부터 semantic level 까지 promote하는 조직적인 방법을 제공
- 연속적인 의미 공간을 코드 압축으로 이산화하는 토큰을 훈련하기 위해 vector-quantized knowledge distillation 제안
- 전역 의미 표현을 향상

[**timm/deit3_base_patch16_224**](https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k)

- 이미지 사이즈 : 224x224
- 2022.04.14
- ViT 아키텍처 사전 정보를 개선
- ViT의 supervised Training 검토
- ResNet-50 훈련을 위해 도입된 레시피 기반, 단순
- 간단한 데이터 증강 절차 포함

[**timm/convnext**](https://huggingface.co/timm/convnext_atto.d2_in1k)

- 이미지 사이즈 : train 224x224 test 288x288
- 2022.03.02
- 바닐라 ViT : 물체 감지 및 의미론적 분할과 같은 일반적인 컴퓨터 비전 작업에 적용될 때 어려움
	- 개선 : 계층적 트랜스 포머(Swin Transformer) : 비전 backbone으로 사용
	- 그러나 이런 하이브리드 접근 방식의 효과는 트랜스포머의 본질적인 우수성에 기인
- 설계 공간을 재검토, 순수 ConvNet이 달성할 수 있는 한계 Test
-  표준 ResNet을 점진적으로 현대화, 성능 차이에 기여하는 몇 가지 주요 구성 요소 발견
- 결과 : ConvNext
- 가장 성능이 좋은 모델 : [convnextv2_huge.fcmae_ft_in22k_in1k_512](https://huggingface.co/timm/convnextv2_huge.fcmae_ft_in22k_in1k_512)

[**timm/convmixer_768_32.in1k**](https://huggingface.co/timm/convmixer_768_32.in1k)

- 이미지 사이즈 : 224x224
- 2022.01.24
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

[**timm/crossvit**](https://huggingface.co/timm/crossvit_9_240.in1k)

- 이미지 사이즈 : 240x240
- 2021.08.22 논문
- 다중 스케일 특징 표현을 학습
- branch 트랜스포머 : 서로 다른 이미지 패치를 결합하여 더 강력한 이미지 특징을 생성 
- 1. 계산 복잡성이 다른 두 개의 개별 분기로 작은 patch 및 큰 patch 토큰을 처리
- 2. 이런 토큰들을 순수하게 attention에 의해 여러 번 융합하여 서로를 보완
- 계산을 줄이기 위해 각 분기에 대한 단일 토큰을 쿼리로 사용
	- 다른 분기와 정보를 교환하는 교차 주의 기반 토큰 융합 모듈을 개발
	- 계산 및 메모리 복잡성에 대해 linear time이 걸림


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

  
[**timm/dm_nfnet_f0.dm_in1k**](https://huggingface.co/timm/dm_nfnet_f0.dm_in1k)

- 이미지 사이즈 : train 192x192, test 256x256
- 2022.02.11
- NFNet (Normalization Free Network)
- 배치 정규화는 배치 크기와 예제 간의 상호 작용에 의존하기 때문에 바람직하지 않은 속성이 많음.
- 정규화 레이어 없이 심층 ResNet 훈련 성공, But 불안 요소 존재
	- 최고의 배치 정규화 네트워크의 테스트 정확도와 일치하지 않음
	- 대규모 학습률이나 강렬한 데이터 증강에 대해 불안정함
- 이런 불안정성을 극복하기 위해 적응형 기울기 클리핑 기술 개발
- 크게 개선된 자유로운 정규화 ResNet 클래스 설계
- 소규모 모델은 8.7배 더 빠르고, EfficientNet-B7의 테스트 정확도와 일치
- Fine-tuning시 배치 정규화된 모델보다 훨씬 더 나은 성능 달성

# 2020

  
# 2019

[**timm/cs3darknet_focus_l.c2ns_in1k**](https://huggingface.co/timm/cs3darknet_focus_l.c2ns_in1k)

- 이미지 사이즈 train 256 x 256, test 288 x 288
- 2019.11.27
- 많은 계산 추론 문제 완화 (최적화)
- **cs3edgenet_x** - Darknet 모델에서 MobileNet-V1을 사용
- 파라미터만 다를 뿐, 같은 형제들
	- cs3darknet_focus_l 
	- cs3darknet_focus_m 
	- cs3darknet_l
	- cs3darknet_m
	- cs3darknet_x 
	- cs3edgenet_x 
	- cs3se_edgenet_x 
	- cs3sedarknet_l 
	- cs3sedarknet_x 
	- cspdarknet53 
	- cspresnet50 
	- cspresnext50

[**timm/dla34.in1k**](https://huggingface.co/timm/dla34.in1k)

- 이미지 사이즈 : 224x224
- 2019.01.04
- Deep Layer Aggregation
- 시각적 인식에선 풍부한 표현이 필요함
	- 낮은 수준 -> 높은 수준
	- 작은 수준 -> 큰 수준
	- 미세한 수준 -> 거친 수준
- 표현을 합성하고 집계하는 것만으론 충분치 않다.
- 표현을 결합하면 무엇과 어디에 대한 추론이 향상
- 네트워크 전체에서 레이어와 블록을 잘 집계하는 방법에 대해 관심을 가짐
- Skip connection의 한계 보안 - 심층 레이어 집계 구조
	- Skip connection은 한 단계 연산으로만 융합
	- 레이어 간의 정보를 더 잘 융합하기 위해 더 깊은 집계로 표준 아키텍처 보강
	- 심층 레이어 집계 구조 : 특징 계층 구조를 반복적이고 계층적으로 병합
		- 더 나은 정확도, 더 적은 매개 변수
		- 인식, 해상도 향상
# 2018

[**timm/darknet53.c2ns_in1k**](https://huggingface.co/timm/darknet53.c2ns_in1k)

- 이미지 사이즈 : train 256x256, test 288x288
- 2018.04.08
- YOLO v3 구현

[**timm/densenet121.ra_in1k**](https://huggingface.co/timm/densenet121.ra_in1k)

- 이미지 사이즈 : train 224x224, test 288x288
- 2018.01.28
- 입력에 가까운 레이어와 출력에 가까운 레이어간의 짧은 연결을 포함
- 각 레이어를 feed-forward 방식으로 다른 모든 레이어에 연결
- 기존 컨벌루션 네트워크는 L개의 연결 -> 네트워크는 L(L+1)의 2방향 연결
	- 각 레이어에 대해 이전 모든 레이어의 특징 map이 입력으로 사용
	- 자체 특징 map이 모든 후속 레이어의 입력으로 사용
- Vanishing gradient 문제 완화
- 특징 전파 강화, 특징 재사용 장려, 매개 변수의 수를 크게 줄임
- **densenetblur121d** - Blur Pooling 사용
  
# 2017

[**timm/dpn68.mx_in1k**](https://huggingface.co/timm/dpn68.mx_in1k)

- 이미지 사이즈 : 224x224
- 2017.08.01
- 


[**timm/bat_resnext26ts.ch_in1k**]()

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