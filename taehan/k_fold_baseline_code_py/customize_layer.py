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

    # 레이어 정의 예시
    model.model.fc = nn.Sequential(
        nn.Linear(model.model.head.in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes)
    )
    
    # 파라미터 학습 가능하게 수정
    for param in model.model.head.parameters():
        param.requires_grad = True

    return model