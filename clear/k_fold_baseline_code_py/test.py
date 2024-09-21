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

def test():
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

    for weight_file in weights:
        model.load_state_dict(torch.load(os.path.join(weight_dir, weight_file)))
        print(weight_file)
        csv_name = os.path.basename(weight_file).replace(".pt", "") + ".csv"

        # 모델로 추론 실행
        predictions = inference(
            model=model,
            device=device,
            test_loader=test_loader
        )

        # test_info의 복사본을 사용하여 CSV 저장
        result_info = test_info.copy()
        result_info['target'] = predictions
        result_info = result_info.reset_index().rename(columns={"index": "ID"})

        save_path = os.path.join(test_csv_dir, csv_name)
        result_info.to_csv(save_path, index=False)