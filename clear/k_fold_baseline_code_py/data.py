import pandas as pd
import cv2
import os
import torch
import numpy as np
import albumentations as A

from typing import Callable, Tuple, Union
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        info_df: pd.DataFrame,
        transform: Callable,
        is_inference: bool = False
    ):
        # 데이터셋의 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블을 초기화합니다.
        self.root_dir = root_dir  # 이미지 파일들이 저장된 기본 디렉토리
        self.transform = transform  # 이미지에 적용될 변환 처리
        self.is_inference = is_inference # 추론인지 확인
        self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록

        if not self.is_inference:
            self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록

    def __len__(self) -> int:
        # 데이터셋의 총 이미지 수를 반환합니다.
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        # 주어진 인덱스에 해당하는 이미지를 로드하고 변환을 적용한 후, 이미지와 레이블을 반환합니다.
        img_path = os.path.join(self.root_dir, self.image_paths[index])  # 이미지 경로 조합
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
        image = self.transform(image)  # 설정된 이미지 변환을 적용합니다.

        if self.is_inference:
            return image
        else:
            target = self.targets[index]  # 해당 이미지의 레이블
            return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환합니다.
        
class TorchvisionTransform: # 단순한 전처리, 간편한 사용, 증강이 사용되면 제한적일 수 있음
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((224, 224)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    transforms.RandomRotation(15),  # 최대 15도 회전
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기 및 대비 조정
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환

        transformed = self.transform(image)  # 설정된 변환을 적용

        return transformed  # 변환된 이미지 반환
    
class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(448, 448),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
            ToTensorV2()  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
                ] + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed['image']  # 변환된 이미지의 텐서를 반환

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type

        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):

        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train)

        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train)

        return transform


def setup_train_directories(save_root_path):
    # 가중치, 로그, TensorBoard 경로 설정
    weight_dir = os.path.join(save_root_path, 'weights')
    log_dir = os.path.join(save_root_path, 'logs')
    tensorboard_dir = os.path.join(save_root_path, 'tensorboard')

    # 디렉토리 생성 (존재하지 않으면 생성)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    return weight_dir, log_dir, tensorboard_dir

def set_up_test_directories(save_root_path):
    save_csv_dir =  os.path.join(save_root_path, 'test_csv')
    os.makedirs(save_csv_dir, exist_ok=True)
    return save_csv_dir

def return_data_frames_and_num_classes(train_csv,test_csv):
    
    # 데이터 준비

    train_info = pd.read_csv(train_csv)
    test_info = pd.read_csv(test_csv)

    num_classes = len(train_info['target'].unique()) 

    return train_info, test_info, num_classes


def set_train_and_val_data(train_info, train_dir, transform = 'torchvision', batch_size = 16):

    # 데이터 준비
    train_df, val_df = train_test_split(train_info, test_size=0.2, stratify=train_info['target'], random_state=42) # split 은 항상 seed 42로 고정.
    transform_selector = TransformSelector(transform_type=transform)
    train_transform, val_transform = transform_selector.get_transform(is_train=True), transform_selector.get_transform(is_train=True)
    train_dataset = CustomDataset(
        root_dir=train_dir,
        info_df=train_df,
        transform=train_transform
    )

    val_dataset = CustomDataset(
        root_dir=train_dir,
        info_df=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader

def set_test_loader(test_info, test_dir, transform, batch_size = 64):
    transform_selector = TransformSelector(transform_type=transform)
    test_transform = transform_selector.get_transform(is_train=False)
   
    test_dataset = CustomDataset(
        root_dir=test_dir,
        info_df=test_info,
        transform=test_transform,
        is_inference=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return test_loader