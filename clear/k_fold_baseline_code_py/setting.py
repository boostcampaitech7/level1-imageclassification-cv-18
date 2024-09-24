
import argparse
import configparser
import os

import torch
import torch.optim as optim
import torch.nn as nn

import custom_model
import custom_train, custom_test
import data

def set_arg_parser_default():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    defaults = config['default']

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='cuda:(gpu)')
    
    # default 부분 수정해서 사용!
    # k_fold로 돌리기 위한 코드, 기존 코드와 달라진 부분이 있어 확인 후 사용 바람

    # method
    parser.add_argument('--model_type', type=str, help='사용할 모델 이름 : model_selector.py 중 선택')
    parser.add_argument('--model_name', type=str, help='model/timm_model_name.txt 에서 확인, 아키텍처 확인은 "https://github.com/huggingface/pytorch-image-models/tree/main/timm/models"')
    parser.add_argument('--pretrained', type=bool, help='전이학습 or 학습된 가중치 가져오기 : True / 전체학습 : False')

    parser.add_argument('--transform', type=str, help='transform class 선택 torchvision or albumentation / dataloader.py code 참고')
    
    # 데이터 경로
    parser.add_argument('--train_dir', type=str, default="/data/ephemeral/home/data/train", help='훈련 데이터셋 루트 디렉토리 경로') # "/data/ephemeral/home/data/train"
    parser.add_argument('--test_dir', type=str, default="/data/ephemeral/home/data/test", help='테스트 데이터셋 루트 디렉토리 경로') # "/data/ephemeral/home/data/test"
    parser.add_argument('--train_csv', type=str, default="/data/ephemeral/home/data/train.csv", help='훈련 데이터셋 csv 파일 경로') # "/data/ephemeral/home/data/train.csv"
    parser.add_argument('--test_csv', type=str, default="/data/ephemeral/home/data/test.csv", help='테스트 데이터셋 csv 파일 경로') # "/data/ephemeral/home/data/test.csv"

    parser.add_argument('--save_rootpath', type=str, default="Experiments/debug", help='가중치, log, tensorboard 그래프 저장을 위한 path 실험명으로 디렉토리 구성')
    
    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, help='에포크 설정')
    parser.add_argument('--lr', type=float, help='learning rage')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--step_size', type=int, help='몇 번째 epoch 마다 학습률 줄일 지 선택')
    parser.add_argument('--gamma', type=float, help='학습률에 얼마를 곱하여 줄일 지 선택')

    args = vars(parser.parse_args())
    result = dict(defaults)
    
    # 하이퍼 파라미터 초기화
    result['gpu'] = int(result['gpu'])
    result['epochs'] = int(result['epochs'])
    result ['lr'] = float(result['lr'])
    result['batch_size'] = int(result['batch_size'])
    result['step_size'] = int(result['step_size'])
    result ['gamma'] = float(result['gamma'])

    result.update({k: v for k, v in args.items() if v is not None}) 
    args = argparse.Namespace(**result)
    return args

def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"is_available cuda : {torch.cuda.is_available()}")
    print(f"current use : cuda({torch.cuda.current_device()})\n")
    return device

def set_trainer(args, train_loader, val_loader, num_classes, device):

    # set model       
    model_selector = custom_model.ModelSelector(
        model_type= args.model_type,
        num_classes = num_classes,
        model_name= args.model_name,
        pretrained= args.pretrained
    )

    model = model_selector.get_model()
    model = custom_model.customize_transfer_layer(model, num_classes)
    model.to(device)


    # set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.step_size,
    gamma=args.gamma
    )

    # set loss
    loss_fn = nn.CrossEntropyLoss() 

    # train
    trainer = custom_train.Trainer(
    model = model,
    model_name = args.model_name,
    pretrained=args.pretrained,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=args.epochs,
    root_log = args.save_rootpath
    )

    return trainer

def set_tester(args, test_info, test_loader, model, save_file_name = 'test.csv', device='cpu'):

    # 모델로 추론 실행
    predictions = custom_test.inference(
        model=model,
        device=device,
        test_loader=test_loader
    )

    # test_info의 복사본을 사용하여 CSV 저장
    result_info = test_info.copy()
    result_info['target'] = predictions
    result_info = result_info.reset_index().rename(columns={"index": "ID"})
    save_path = data.set_up_test_directories(args.save_rootpath)
    save_path = os.path.join(save_path, save_file_name)
    result_info.to_csv(save_path, index=False)