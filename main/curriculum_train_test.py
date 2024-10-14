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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from torch.utils.tensorboard import SummaryWriter

from loss import CrossEntropyLoss
from model.model_selector import ModelSelector
from base.customize_layer import customize_layer
from curriculum.curriculum_dataloader import CustomDataset, TorchvisionTransform, AlbumentationsTransform
from curriculum.curriculum_trainer import Trainer

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
    save_csv_dir =  os.path.join(save_rootpath, 'test_csv')

    # 디렉토리 생성 (존재하지 않으면 생성)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(save_csv_dir, exist_ok=True)

    return weight_dir, log_dir, tensorboard_dir, save_csv_dir

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

def train_test():
    # set cuda
    device = set_cuda(args.gpu) 

    # set save dir
    weight_dir, log_dir, tensorboard_dir, test_csv_dir = setup_directories(args.save_rootpath)
    logfile = os.path.join(log_dir, "train_log.log")

    # 데이터 준비
    traindata_dir = args.train_dir
    traindata_info_file = args.train_csv

    train_info = pd.read_csv(traindata_info_file)
    y = train_info.iloc[:,2].tolist()
   
    num_classes = len(train_info['target'].unique()) 

    if args.transform == "TorchvisionTransform":
        train_transform = TorchvisionTransform(is_train=True)
        val_transform = TorchvisionTransform(is_train=False)
    elif args.transform == "AlbumentationsTransform":
        train_transform = AlbumentationsTransform(is_train=True)
        val_transform = AlbumentationsTransform(is_train=False)

    # 폴드 수 설정
    k_folds = args.num_k_fold
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 각 폴드마다 루프
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_info, y)):

        train_fold_file = f"train_fold_{fold+1}.csv"
        val_fold_file = f"val_fold_{fold+1}.csv"
        
        # Train과 validation 데이터를 나눔
        train_fold_data = train_info.iloc[train_idx]
        val_fold_data = train_info.iloc[test_idx]
        
        # CSV로 저장
        train_fold_data.to_csv(train_fold_file, index=False)
        val_fold_data.to_csv(val_fold_file, index=False)    

        print(f"Fold {fold + 1}")
        print("-------")

        val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_fold_data,
        transform=val_transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        # 모델 설정
        model_selector = ModelSelector(
            model_type= args.model_type,
            num_classes = num_classes,
            model_name= args.model_name,
            pretrained= args.pretrained
        )

        model = model_selector.get_model()

        if args.pretrained == True:
            for param in model.parameters():
                param.requires_grad = False
        
            model = customize_layer(model, num_classes)

        model = model.to(device)
        
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
        )

        # loss
        loss_fn = CrossEntropyLoss() 
        
        # Trainer 인스턴스 생성 및 학습
        trainer = Trainer(
            model=model,
            device=device,
            train_dir = args.train_dir,
            train_data=train_fold_data,  # train_loader 대신 train_data 전달
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=args.epochs,
            weight_path= weight_dir,
            log_path= logfile,
            tensorboard_path= tensorboard_dir,
            model_name=args.model_name,
            pretrained=args.pretrained,
            batch_size=args.batch_size
        )

        # 학습 시작
        trainer.train(fold)
    #-------------------------------------------------------

    # test
    test_info = pd.read_csv(args.test_csv)

    test_dataset = CustomDataset(
        root_dir=args.test_dir,
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

    weights = os.listdir(weight_dir)
    print(weights)

    # k-fold ensemble
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

        np.save(os.path.join(test_csv_dir, f"fold{fold+1}_eva_large_curriculum_mlp_gelu.npy"), predictions)
        
        csv_name_fold = f"fold{fold+1}_eva_large_curriculum_mlp_gelu.csv"
        result_info = test_info.copy()
        predictions = np.argmax(predictions, axis=1)
        result_info['target'] = predictions 
        result_info = result_info.reset_index().rename(columns={"index": "ID"})
        save_path = os.path.join(test_csv_dir, csv_name_fold)
        result_info.to_csv(save_path, index=False)

    k_fold_predictions = np.array(k_fold_predictions) # (fold size, test_size, num_classes)
    print(f"K-fold predictions shape: {np.shape(k_fold_predictions)}")

    np.save(os.path.join(test_csv_dir, f"all_fold_eva_large_curriculum_mlp_gelu.npy"), k_fold_predictions)

    # 확률 평균화
    average_probs = np.mean(k_fold_predictions, axis=0)
    # 최종 예측값 결정
    final_predictions = np.argmax(average_probs, axis=1)

    np.save(os.path.join(test_csv_dir, f"softvoting_all_fold_eva_large_curriculum_mlp_gelu.npy"), average_probs)

    # test_info의 복사본을 사용하여 CSV 저장
    csv_name = "softvoting_5_eva_large_curriculum_mlp_gelu.csv"
    result_info = test_info.copy()
    result_info['target'] = final_predictions 
    result_info = result_info.reset_index().rename(columns={"index": "ID"})

    save_path = os.path.join(test_csv_dir, csv_name)
    result_info.to_csv(save_path, index=False)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='cuda:(gpu)')
    
    # default 부분 수정해서 사용!
    # k_fold로 돌리기 위한 코드, 기존 코드와 달라진 부분이 있어 확인 후 사용 바람

    # method
    parser.add_argument('--model_type', type=str, default='timm', help='사용할 모델 이름 : model_selector.py 중 선택')
    parser.add_argument('--model_name', type=str, default='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', help='model/timm_model_name.txt 에서 확인, 아키텍처 확인은 "https://github.com/huggingface/pytorch-image-models/tree/main/timm/models"')
    parser.add_argument('--pretrained', type=bool, default='True', help='전이학습 or 학습된 가중치 가져오기 : True / 전체학습 : False')
    # 전이학습할 거면 꼭! (True) customize_layer.py 가서 레이어 수정, 레이어 수정 안할 거면 가서 레이어 구조 변경 부분만 주석해서 사용 (어떤 레이어 열지는 알아야함)
    # 모델 구조랑 레이어 이름 모르겠으면 위에 모델 정의 부분가서 print(model) , assert False 주석 풀어서 확인하기

    parser.add_argument('--transform', type=str, default='AlbumentationsTransform', help='transform class 선택 torchvision or albumentation / dataloader.py code 참고')
    
    # 데이터 경로
    parser.add_argument('--train_dir', type=str, default="/data/ephemeral/home/data/train", help='훈련 데이터셋 루트 디렉토리 경로') # "/data/ephemeral/home/data/train"
    parser.add_argument('--test_dir', type=str, default="/data/ephemeral/home/data/test", help='테스트 데이터셋 루트 디렉토리 경로') # "/data/ephemeral/home/data/test"
    parser.add_argument('--train_csv', type=str, default="/data/ephemeral/home/data/train.csv", help='훈련 데이터셋 csv 파일 경로') # "/data/ephemeral/home/data/train.csv"
    parser.add_argument('--test_csv', type=str, default="/data/ephemeral/home/data/test.csv", help='테스트 데이터셋 csv 파일 경로') # "/data/ephemeral/home/data/test.csv"

    parser.add_argument('--save_rootpath', type=str, default="Experiments/eva_large_curriculum_mlp_gelu", help='가중치, log, tensorboard 그래프 저장을 위한 path 실험명으로 디렉토리 구성')
    parser.add_argument('--csv_name', type=str, default="curriculum_fold1.csv", help='')
    
    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=20, help='에포크 설정')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rage')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--step_size', type=int, default=5, help='몇 번째 epoch 마다 학습률 줄일 지 선택')
    parser.add_argument('--gamma', type=float, default=0.5, help='학습률에 얼마를 곱하여 줄일 지 선택')
    parser.add_argument('--num_k_fold', type=int, default=5, help='k-fold 수 설정')

    args = parser.parse_args()

    start_time = time.time()
    train_test()
    end_time = time.time()

    print(f" End : {(end_time - start_time)/60} min")
