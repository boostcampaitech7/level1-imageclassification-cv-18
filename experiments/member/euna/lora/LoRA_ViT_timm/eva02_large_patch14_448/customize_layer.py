import torch.nn as nn

def customize_layer(model, num_classes):
    '''
    사용하고자하는 model 구조 확인 후 작성 (레이어 이름)
    print(model) 로 모델 구조 확인 가능 

    ex) model.fc , model.classifer (fc, classifier ... )

    만약 model 구조 수정 안하고 기존 레이어 몇 개만 열어서 학습하고 싶으면
    레이어 정의만 주석하면 됩니다.

    model.model.fc < 이부분이 달라닙니다 예를들어 TimModel (ResNet)의 model 레이어안에 fc 레이어 (TimModel 의 경우 model 이라는 레이어안에 구현해놓은 듯)
    '''

    # head 레이어 수정
    model.model.head.original_module = nn.Sequential(
        nn.Linear(model.model.head.original_module.in_features, 1000),  # 중간 레이어로 1000차원 설정
        nn.ReLU(),
        nn.Linear(1000, num_classes)  # 최종 출력 클래스 수로 맞춤
    )

    # 파라미터 학습 가능하게 설정
    for param in model.model.head.original_module.parameters():
        param.requires_grad = True

    return model