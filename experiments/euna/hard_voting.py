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
        "train 결과/cosine_annealing_LR/Experiments_15_1e-6/debug/test_csv/best_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_True_epoch_0_loss_0.5640.csv",
        "train 결과/cosine_annealing_LR/Experiments_15_1e-6/debug/test_csv/best_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_True_epoch_1_loss_0.4665.csv",
        "train 결과/cosine_annealing_LR/Experiments_15_1e-6/debug/test_csv/best_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_True_epoch_2_loss_0.4073.csv"
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
    final_result.to_csv("hard_voting_final_result.csv", index=False)

# 동일하게 python hard_voting.py로 실행!
if __name__ == "__main__":
    main()
