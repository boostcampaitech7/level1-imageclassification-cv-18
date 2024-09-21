



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

    parser.add_argument('--save_rootpath', type=str, default="Experiments/debug", help='가중치, log, tensorboard 그래프 저장을 위한 path 실험명으로 디렉토리 구성')
    
    # 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=30, help='에포크 설정')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rage')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--step_size', type=int, default=5, help='몇 번째 epoch 마다 학습률 줄일 지 선택')
    parser.add_argument('--gamma', type=float, default=0.1, help='학습률에 얼마를 곱하여 줄일 지 선택')

    args = parser.parse_args()

    start_time = time.time()
    train_test()
    end_time = time.time()

    print(f" End : {(end_time - start_time)/60} min")