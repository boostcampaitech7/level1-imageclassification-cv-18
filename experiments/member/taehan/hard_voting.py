import pandas as pd
import torch

def load_csv_results(file_paths):
    '''
    csv 파일 로드
    '''
    results = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        results.append(df)
    return results

def main():
    # csv 파일 리스트로 csv 파일 경로 넣어 주면 됩니다! 
    csv_files = [
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/softvoting_one_piece.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/softvoting_curr12.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/hardvoting_one_piece.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/eva_large_curriculum_head/Experiments/eva_large_curriculum_head/test_csv/softvoting_5_eva_large_curriculum_head.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/eva_large_curriculum_mlp/Experiments/eva_large_curriculum_mlp_gelu/test_csv/softvoting_5_eva_large_curriculum_mlp_gelu.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/eva_large_curriculum_mlp/Experiments/eva_large_curriculum_mlp_gelu/test_csv/output.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/eva_large_curriculum_mlp/Experiments/eva_large_curriculum_mlp_gelu/test_csv/output_1.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/eva_large_curriculum_mlp/Experiments/eva_large_curriculum_mlp_gelu/test_csv/output_2.csv",
        "/data/ephemeral/home/chan/level1-imageclassification-cv-18/last_train/eva_large_curriculum_mlp/Experiments/eva_large_curriculum_mlp_gelu/test_csv/output_3.csv"
    ]

    model_results = load_csv_results(csv_files)
    
    num_data = len(model_results[0])
    num_classes = 500
    
    votes = torch.zeros((num_data, num_classes+1), dtype=torch.int)

    for i in range(num_data):
        for model_result in model_results:
            output_class = model_result.loc[i, "target"]  
            votes[i][output_class] += 1  

    final_preds = votes.argmax(-1)

    final_result = model_results[0].copy()

    final_result["target"] = final_preds.numpy()

    # 결과 파일 이름 or 경로 지정
    final_result.to_csv("one_piece.csv", index=False)

# 동일하게 python hard_voting.py로 실행!
if __name__ == "__main__":
    main()
