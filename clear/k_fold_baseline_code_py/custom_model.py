import torch.nn as nn
from model import SimpleCNN, TimmModel

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self,
        model_type: str,
        num_classes: int,
        **kwargs
    ):
        # 모델 유형에 따라 적절한 모델 객체를 생성
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)

        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)

        else:
            raise ValueError("Unknown model type specified.")
            
    def get_model_name(self): return self.model_name

    def get_model(self) -> nn.Module:
        return self.model
    

def customize_transfer_layer(model, num_classes):
    
    for param in model.parameters():
        param.requires_grad = False

    # 레이어 정의 예시
    model.model.fc = nn.Sequential(
        nn.Linear(model.model.head.in_features, 1024),
        nn.GELU(),
        nn.Linear(1024, num_classes)
    )

    # 파라미터 학습 가능하게 수정
    for param in model.model.head.parameters():
        param.requires_grad = True
    return model