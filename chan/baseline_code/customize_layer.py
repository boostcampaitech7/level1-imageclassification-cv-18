import torch.nn as nn

def customize_layer(model, num_classes):
    '''
    사용하고자하는 model 구조 확인 후 작성 (레이어 이름)
    print(model) 로 모델 구조 확인 가능 

    ex) model.fc , model.classifer (fc, classifier ... )

    만약 model 구조 수정 안하고 기존 레이어 몇 개만 열어서 학습하고 싶으면
    레이어 정의만 주석하면 됩니다.
    '''

    # 레이어 정의 예시
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1000),
        nn.ReLU(),
        nn.Linear(1000, num_classes)
    )
    
    # 파라미터 학습 가능하게 수정
    for param in model.fc.parameters():
        param.requires_grad = True

    return model