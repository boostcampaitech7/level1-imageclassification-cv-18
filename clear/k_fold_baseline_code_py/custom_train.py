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

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from torch.utils.tensorboard import SummaryWriter

# from loss import CrossEntropyLoss
# from custom_model import ModelSelector, customize_transfer_layer
# from data import CustomDataset, TorchvisionTransform, AlbumentationsTransform

# 하나의 함수는 하나의 기능만 하도록
# 클래스가 클래스의 기능을 할 수 있도록
# data는 data_loader 단에 묶을 수 있도록

import torch.nn as nn
import torch
import torch.optim as optim
import os
import logging

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        model : nn.Module,
        model_name : str,
        pretrained : bool,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss,
        epochs: int,
        root_log: str,
        early_stopping_patience : int = 5 
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정

        # 데이터 초기화
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더

        # 모델 초기화
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.model = model

        # 하이퍼 파라미터
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수

        # 모델 저장, 로그 저장
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
        self.weights, self.logs, self.tensorboards = self.__set_logs(root_log)
        self.train_log = os.path.join(self.logs, "train_log.log")
    
    def __set_logs(self,root_log):

        # 가중치, 로그, TensorBoard 경로 설정
        weight_dir = os.path.join(root_log, 'weights')
        log_dir = os.path.join(root_log, 'logs')
        tensorboard_dir = os.path.join(root_log, 'tensorboard')

        # 디렉토리 생성 (존재하지 않으면 생성)
        os.makedirs(weight_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        return weight_dir, log_dir, tensorboard_dir


    def save_model(self, epoch, loss):

        # 모델 저장 경로 설정
        os.makedirs(self.weights, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.weights, f'{self.model_name}_{self.pretrained}_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)


        # 수정 필요
        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.weights, f'best_{self.model_name}_{self.pretrained}_epoch_{epoch}_loss_{loss:.4f}.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")


    def train_epoch(self) -> float:

        # 한 에폭 동안의 훈련을 진행
        self.model.train()

        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def validate(self) -> float:

        # 모델의 검증을 진행
        self.model.eval()

        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def train(self) -> None:

        # 전체 훈련 과정을 관리
        logging.basicConfig(
            level=logging.INFO,  # 로그 레벨을 INFO로 설정
            format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 형식
            handlers=[
                logging.FileHandler(self.train_log),  # 로그를 파일에 기록
                logging.StreamHandler()  # 로그를 콘솔에도 출력
            ]
        )

        writer = SummaryWriter(log_dir=self.tensorboards)

        logger = logging.getLogger()
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate()
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

            self.save_model(epoch, val_loss)

            writer.add_scalar('Loss/train', train_loss, epoch)  # 훈련 손실 기록
            writer.add_scalar('Loss/validation', val_loss, epoch)  # 검증 손실 기록
        writer.close()    

    #-------------------------------------------------------