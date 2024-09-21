import torch
from torch import nn

import torch.nn.functional as F 
from tqdm import tqdm
import data
def inference(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader
):
    
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad(): 
        for images in tqdm(test_loader):
            images = images.to(device)

            # 모델을 통해 예측 수행
            # ensemble을 위해 스코어 벡터로 반환
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            # preds = logits.argmax(dim=1)

            # 예측 스코어 벡터 저장
            # predictions.append(logits.cpu().numpy())

            # 예측 결과 저장
            predictions.extend(logits.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가

    return predictions
