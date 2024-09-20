import torch.nn as nn
import torch
import torch.optim as optim
import os
import argparse
import pandas as pd
import logging
import time
import torch.nn.functional as F 
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from torch.utils.tensorboard import SummaryWriter

from loss import CrossEntropyLoss
from model_selector import ModelSelector
from dataloader import CustomDataset, TorchvisionTransform, AlbumentationsTransform
from customize_layer import customize_layer
from trainer import Trainer

def plot_confusion_matrix(y_true, y_pred, num_classes, top_k=10):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 상위 k개의 클래스를 선택
    top_k_indices = np.argsort(np.sum(cm, axis=1))[-top_k:]
    cm_reduced = cm[np.ix_(top_k_indices, top_k_indices)]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=top_k_indices, yticklabels=top_k_indices)
    plt.title(f'Confusion Matrix (Top {top_k} Classes)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    save_path = f'confusion_matrix_top_{top_k}.png'
    plt.savefig(save_path)
    plt.close()
    
    print(f'Confusion matrix (Top {top_k} Classes) saved to {save_path}')

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



val_transform = AlbumentationsTransform(is_train=False)

# train 데이터 전체에 대해 테스트
test_info = pd.read_csv('/data/ephemeral/home/data/train.csv')
weight_dir = './Experiments/kfold_last/weights'
test_dataset = CustomDataset(
    root_dir='/data/ephemeral/home/data/train',
    info_df=test_info,
    transform=val_transform,
    is_inference=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    drop_last=False
)

model_selector = ModelSelector(
            model_type= 'timm',
            num_classes = 500,
            model_name= 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            pretrained= True
        )

model = model_selector.get_model()

for param in model.parameters():
    param.requires_grad = False
    
    model = customize_layer(model, 500)


# 사전 학습된 모델을 로드
model.load_state_dict(torch.load(os.path.join(weight_dir, '0_bestmodel.pt')))
model.to('cuda')
model.eval()

# 테스트 데이터셋에 대한 예측 수행
predictions = inference(
    model=model,
    device='cuda',
    test_loader=test_loader
)

# 최종 예측값 결정
predictions = np.array(predictions)
final_predictions = np.argmax(predictions, axis=1)

# Confusion matrix 생성 및 저장
true_labels = test_info['target'].values
plot_confusion_matrix(true_labels, final_predictions, num_classes=500, top_k=10)