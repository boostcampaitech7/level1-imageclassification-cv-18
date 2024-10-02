import torch.nn as nn
import torch
import torch.optim as optim
import os
import logging
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss,
        epochs: int,
        weight_path: str,
        log_path: str,
        tensorboard_path: str,
        model_name : str,
        pretrained : bool
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.weight_path = weight_path  # 모델 저장 경로
        self.log_path = log_path # 로그 저장 경로
        self.tensorboard_path = tensorboard_path # 로그 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
        self.highest_accu = 0.0
        self.model_name = model_name
        self.pretrained = pretrained
    def save_model(self, epoch, loss, accu, fold):
        # 모델 저장 경로 설정
        os.makedirs(self.weight_path, exist_ok=True)

        # # 현재 에폭 모델 저장
        # current_model_path = os.path.join(self.weight_path, f'{self.model_name}_{self.pretrained}_epoch_{epoch}_loss_{loss:.4f}.pt')
        # torch.save(self.model.state_dict(), current_model_path)

        # # 최상위 3개 모델 관리
        # self.best_models.append((loss, epoch, current_model_path))
        # self.best_models.sort()
        # if len(self.best_models) > 3:
        #     _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
        #     if os.path.exists(path_to_remove):
        #         os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.weight_path, f'{fold}_bestmodel.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

        if accu > self.highest_accu:
            self.highest_accu = accu
            best_model_path = os.path.join(self.weight_path, f'{fold}_bestmodel_accu.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. accu = {accu:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
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
            logits = F.softmax(outputs, dim=1)
            preds = logits.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
        self.scheduler.step()
        return total_loss / len(self.train_loader), correct / total * 100

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()

        total_loss = 0.0
        total_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                logits = F.softmax(outputs, dim=1)
                preds = logits.argmax(dim=1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()

        return total_loss / len(self.val_loader), correct / total * 100

    def train(self, fold) -> None:

        # 전체 훈련 과정을 관리
        logging.basicConfig(
            level=logging.INFO,  # 로그 레벨을 INFO로 설정
            format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 형식
            handlers=[
                logging.FileHandler(self.log_path),  # 로그를 파일에 기록
                logging.StreamHandler()  # 로그를 콘솔에도 출력
            ]
        )

        train_writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_path, f'train_fold{fold+1}'))
        validation_writer = SummaryWriter(log_dir=os.path.join(self.tensorboard_path, f'validation_fold{fold+1}'))

        logger = logging.getLogger()
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}, Validation Loss: {val_loss:.4f}, Validataion Accuracy: {val_acc:.2f}\n")

            self.save_model(epoch, val_loss, val_acc, fold)

            train_writer.add_scalar('train/Loss', train_loss, epoch)  # 훈련 손실 기록
            train_writer.add_scalar('train/Accuracy', train_acc, epoch)  # 훈련 손실 기록
            validation_writer.add_scalar('validation/Loss', val_loss, epoch)  # 검증 손실 기록
            validation_writer.add_scalar('validation/Accuracy', val_acc, epoch)  # 검증 손실 기록

            train_writer.add_scalar('train_validation/Loss', train_loss, epoch)  # 훈련 손실 기록
            train_writer.add_scalar('train_validation/Accuracy', train_acc, epoch)
            validation_writer.add_scalar('train_validation/Loss', val_loss, epoch)  # 검증 손실 기록
            validation_writer.add_scalar('train_validation/Accuracy', val_acc, epoch)  # 검증 손실 기록
        
        train_writer.close()    
        validation_writer.close()    