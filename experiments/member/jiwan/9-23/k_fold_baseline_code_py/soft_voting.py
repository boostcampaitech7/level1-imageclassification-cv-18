import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 진행률 표시를 위한 tqdm
from typing import Callable, Union, Tuple  # 타입 힌트를 위한 import

from train_test import inference
from dataloader import CustomDataset, AlbumentationsTransform
from model_selector import ModelSelector

def test_kfold(test_csv, test_dir, weight_dir, k_folds, test_csv_dir):
    # k-fold 모델 앙상블을 위한 코드
    test_info = pd.read_csv(test_csv)
    device = 'cuda'
    val_transform = AlbumentationsTransform(is_train=False)

    test_dataset = CustomDataset(
        root_dir=test_dir,
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

    # 모델 설정
    model_selector = ModelSelector(
        model_type='timm',
        num_classes=500,
        model_name='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
        pretrained=True
    )
    model = model_selector.get_model()

    weights = os.listdir(weight_dir)
    print(weights)

    # k-fold 앙상블
    k_fold_predictions = []
    for fold in range(k_folds):
        print(f"Fold {fold + 1} inference")
        print("-------")
        model.load_state_dict(torch.load(os.path.join(weight_dir, f'{fold}_bestmodel.pt')))
        
        # 모델로 추론 실행
        predictions = inference(
            model=model,
            device=device,
            test_loader=test_loader
        )
        k_fold_predictions.append(predictions)

    # K-fold 예측 확률을 배열로 변환
    k_fold_predictions = np.array(k_fold_predictions)  # (fold size, test_size, num_classes)
    print(f"K-fold predictions shape: {np.shape(k_fold_predictions)}")

    # 확률 평균화
    average_probs = np.mean(k_fold_predictions, axis=0)
    return average_probs, test_info  # test_info도 반환

def soft_voting(predictions_list):
    # 각각의 average_probs의 평균을 하여 하나의 average_probs를 만듦
    return np.mean(predictions_list, axis=0)  # 리스트의 평균을 반환

def save_csv(average_probs, test_info, test_csv_dir):
    # 최종 예측값 결정 (확률이 가장 높은 클래스 선택)
    final_predictions = np.argmax(average_probs, axis=1)

    # 결과를 CSV로 저장
    csv_name = "k-fold_ensemble.csv"
    result_info = test_info.copy()
    result_info['target'] = final_predictions

    result_info = result_info.reset_index().rename(columns={"index": "ID"})

    save_path = os.path.join(test_csv_dir, csv_name)
    result_info.to_csv(save_path, index=False)

# ---------------------------------------------------------
test_csv = "/data/ephemeral/home/data/test.csv"
test_dir = "/data/ephemeral/home/data/train"
test_csv_dir = 'Experiments/debug/test_csv'
weight_dir1 = "Experiments/debug/weights"
weight_dir2 = ""
k_folds = 2
model_num = 1  # ----------------------------------------------------------

# K-fold 추론 및 소프트 보팅
arr = []
average_probs, test_info = test_kfold(test_csv, test_dir, weight_dir1, k_folds, test_csv_dir)
# arr.append(average_probs)

# final_average_probs = soft_voting(arr)  # 최종 평균 확률 계산
save_csv(average_probs, test_info, test_csv_dir)  # CSV 저장
