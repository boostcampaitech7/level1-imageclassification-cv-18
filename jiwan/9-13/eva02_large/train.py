import torch.nn as nn
import torch
import torch.optim as optim
import os
import tqdm
import argparse
import pandas as pd
import logging
import time
import torch.nn.functional as F


from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model_selector import ModelSelector
from dataloader import CustomDataset, TorchvisionTransform, AlbumentationsTransform
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Loss(nn.Module):

    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        return self.loss_fn(outputs, targets)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        tensorboard_path: str
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

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.weight_path, exist_ok=True)

        # 현재 에폭 모델 저장
        current_model_path = os.path.join(self.weight_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

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
            best_model_path = os.path.join(self.weight_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()

        total_loss = 0.0
        total = 0
        correct = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            # 만약 images[0]이 torch.Tensor라면
            # image_np = images[0].cpu().numpy()  # tensor를 numpy array로 변환
            # image_np = np.transpose(image_np, (1, 2, 0))  # (C, H, W)에서 (H, W, C)로 변환 필요

            # cv2.imwrite("debut.jpg", image_np)
            #assert False
            # print(images[0].max())
            # print(images[0].min())
            self.optimizer.zero_grad()
            outputs = self.model(images)
            #print(outputs.shape)
            #assert False

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            logits = F.softmax(outputs, dim=1)
            preds = logits.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

        return total_loss / len(self.train_loader), correct / total * 100

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()

        total_loss = 0.0
        total = 0
        correct = 0
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

    def train(self) -> None:

        # 전체 훈련 과정을 관리
        logging.basicConfig(
            level=logging.INFO,  # 로그 레벨을 INFO로 설정
            format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 형식
            handlers=[
                logging.FileHandler(self.log_path),  # 로그를 파일에 기록
                logging.StreamHandler()  # 로그를 콘솔에도 출력
            ]
        )

        writer = SummaryWriter(log_dir=self.tensorboard_path)

        logger = logging.getLogger()
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}, Validation Loss: {val_loss:.4f}, Validataion Accuracy: {val_acc:.2f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step()

            writer.add_scalar('Loss/train', train_loss, epoch)  # 훈련 손실 기록
            writer.add_scalar('Accuracy/train', train_acc, epoch)  # 훈련 손실 기록
            writer.add_scalar('Loss/validation', val_loss, epoch)  # 검증 손실 기록
            writer.add_scalar('Accuracy/validation', val_acc, epoch)  # 검증 손실 기록
        writer.close()    


def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"is_available cuda : {torch.cuda.is_available()}")
    print(f"current use : cuda({torch.cuda.current_device()})\n")
    return device

def setup_directories(save_rootpath):
    # 가중치, 로그, TensorBoard 경로 설정
    weight_dir = os.path.join(save_rootpath, 'weights')
    log_dir = os.path.join(save_rootpath, 'logs')
    tensorboard_dir = os.path.join(save_rootpath, 'tensorboard')

    # 디렉토리 생성 (존재하지 않으면 생성)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    return weight_dir, log_dir, tensorboard_dir

def train():
    # set cuda
    device = set_cuda(args.gpu) 

    #set save dir
    weight_dir, log_dir, tensorboard_dir = setup_directories(args.save_rootpath)
    logfile = os.path.join(log_dir, "trianlog.log")

    # 데이터 준비
    traindata_dir = args.train_dir
    traindata_info_file = args.train_csv

    train_info = pd.read_csv(traindata_info_file)
    num_classes = len(train_info['target'].unique()) # 클래스 수

    train_df, val_df = train_test_split(train_info, test_size=0.2, stratify=train_info['target'], random_state=42)
    train_transform = AlbumentationsTransform(is_train=True)
    val_transform = AlbumentationsTransform(is_train=False)

    train_dataset = CustomDataset(
    root_dir=traindata_dir,
    info_df=train_df,
    transform=train_transform
    )

    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # set model       
    model_selector = ModelSelector(
        model_type= args.model_type,
        num_classes = num_classes,
        model_name= args.model_name,
        pretrained=True
    )

    
    model = model_selector.get_model()
    model = model.to(device)

    import torch.nn as nn

    num_features = model.model.head.in_features

    # 새로운 MLP HEAD 레이어 추가
    model.model.head = nn.Sequential( # MLP HEAD
        nn.Linear(num_features, 1024, bias=True),  # 추가할 FC 레이어
        nn.GELU(approximate='none'),
        nn.Dropout(p=0.0, inplace=False),
        nn.Identity(),
        nn.Linear(in_features=1024, out_features=500, bias=True) # 추가할 FC 레이어
    )

    # 전이 학습
    for n,p in model.model.named_parameters():
        p.requires_grad=False

    # 추가한 FC 레이어의 파라미터만 학습 가능하도록 설정
    for name, param in model.model.head.named_parameters():
        param.requires_grad = True

    model = model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    steps_per_epoch = len(train_loader)

    scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=steps_per_epoch * args.step_size,
    gamma=args.gamma
    )

    # loss
    loss_fn = Loss() # CrossEntropyLoss
    loss_fn = loss_fn.to(device)
    # train
    trainer = Trainer(
    model=model,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=args.epochs,
    weight_path= weight_dir,
    log_path= logfile,
    tensorboard_path= tensorboard_dir
    )

    trainer.train()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='cuda:(gpu)')
    
    # default 부분 수정해서 사용!!

    # 모델 선택
    parser.add_argument('--model_type', type=str, default='timm', help='사용할 모델 이름')
    parser.add_argument('--model_name', type=str, default='timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', help='timm model을 사용할 경우 timm 모델 중 선택')

    # 데이터 경로
    parser.add_argument('--train_dir', type=str, default="/data/ephemeral/home/data/train", help='훈련 데이터셋 루트 디렉토리 경로')
    parser.add_argument('--test_dir', type=str, default="/data/ephemeral/home/data/test", help='테스트 데이터셋 루트 디렉토리 경로')
    parser.add_argument('--train_csv', type=str, default="/data/ephemeral/home/data/train.csv", help='훈련 데이터셋 csv 파일 경로')
    parser.add_argument('--test_csv', type=str, default="/data/ephemeral/home/data/test.csv", help='테스트 데이터셋 csv 파일 경로')
    parser.add_argument('--save_rootpath', type=str, default="Experiments/debug", help='가중치, log, tensorboard 그래프 저장을 위한 path 실험명으로 디렉토리 구성')
    
    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=30, help='에포크 설정')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rage')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--step_size', type=int, default=15, help='몇 번째 epoch 마다 학습률 줄일지 선택')
    parser.add_argument('--gamma', type=float, default=0.1, help='학습률에 얼마를 곱하여 줄일지 선택')

    args = parser.parse_args()

    start_time = time.time()
    train()
    end_time = time.time()

    print(f" End : {(end_time - start_time)/60} min")