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
    
    model.model.head = nn.Sequential( # MLP HEAD
    nn.Linear(model.model.head.in_features, 2048, bias=True),  # 첫 번째 FC 레이어
    nn.GELU(approximate='none'),
    nn.Dropout(p=0.3, inplace=False),
    
    nn.Linear(in_features=2048, out_features=1024, bias=True),  # 두 번째 FC 레이어
    nn.GELU(approximate='none'),
    nn.Dropout(p=0.3, inplace=False),
    
    nn.Linear(in_features=1024, out_features=512, bias=True),  # 세 번째 FC 레이어
    nn.GELU(approximate='none'),
    nn.Dropout(p=0.3, inplace=False),
    
    nn.Linear(in_features=512, out_features=500, bias=True)  # 출력 레이어
)
    
    # 파라미터 학습 가능하게 수정
    for n,p in model.model.named_parameters():
        p.requires_grad=False

# 추가한 FC 레이어의 파라미터만 학습 가능하도록 설정
    for name, param in model.model.head.named_parameters():
        param.requires_grad = True

    return model