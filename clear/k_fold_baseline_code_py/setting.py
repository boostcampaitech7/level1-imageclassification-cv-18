


def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"is_available cuda : {torch.cuda.is_available()}")
    print(f"current use : cuda({torch.cuda.current_device()})\n")
    return device

def set_train_and_val_data():

    # 데이터 준비
    train_data_dir = args.train_dir
    train_data_info_file = args.train_csv

    train_info = pd.read_csv(train_data_info_file)
    num_classes = len(train_info['target'].unique()) 

    train_df, val_df = train_test_split(train_info, test_size=0.2, stratify=train_info['target'], random_state=42) # split 은 항상 seed 42로 고정.
    
    if args.transform == "TorchvisionTransform":
        train_transform = TorchvisionTransform(is_train=True)
        val_transform = TorchvisionTransform(is_train=False)
    elif args.transform == "AlbumentationsTransform":
        train_transform = AlbumentationsTransform(is_train=True)
        val_transform = AlbumentationsTransform(is_train=False)

    train_dataset = CustomDataset(
    root_dir=train_data_dir,
    info_df=train_df,
    transform=train_transform
    )

    val_dataset = CustomDataset(
        root_dir=train_data_dir,
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

    return train_loader, val_loader, num_classes