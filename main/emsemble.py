import numpy as np
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

def hard_hard():
    # csv 파일 리스트로 csv 파일 경로 넣어 주면 됩니다! 
    csv_files = [
        "main/csv/hard-hard/hard_fold1_eva_giant_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold2_eva_giant_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold3_eva_giant_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold4_eva_giant_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold5_eva_giant_mlp_gelu.csv",
        
        "main/csv/hard-hard/hard_fold1_eva_large_curriculum_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold2_eva_large_curriculum_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold3_eva_large_curriculum_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold4_eva_large_curriculum_mlp_gelu.csv",
        "main/csv/hard-hard/hard_fold5_eva_large_curriculum_mlp_gelu.csv",

        "main/csv/hard-hard/hard_fold1_eva_large_mlp.csv",
        "main/csv/hard-hard/hard_fold2_eva_large_mlp.csv",
        "main/csv/hard-hard/hard_fold3_eva_large_mlp.csv",
        "main/csv/hard-hard/hard_fold4_eva_large_mlp.csv",
        "main/csv/hard-hard/hard_fold5_eva_large_mlp.csv"
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

def soft_hard():
    # csv 파일 리스트로 csv 파일 경로 넣어 주면 됩니다! 
    csv_files = [
        "main/csv/soft-hard/5-fold_softvoting_eva_giant_mlp_gelu.csv",
        "main/csv/soft-hard/5-fold_softvoting_eva_large_mlp.csv",
        "main/csv/soft-hard/softvoting_5_eva_large_curriculum_mlp_gelu.csv"
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

def soft_soft():
    test_csv_path = "/data/ephemeral/home/data/test.csv" # 변경 필요 : "/data/ephemeral/home/data/train"으로 변경해야함 
    
    test_info = pd.read_csv(test_csv_path)

    prediction_list = [
        "main/score_vector/soft-soft/all_fold_eva_giant_mlp_gelu.npy",
        "main/score_vector/soft-soft/all_fold_eva_large_curriculum_mlp_gelu.npy",
        "main/score_vector/soft-soft/all_fold_eva_large_mlp.npy"
    ]

    predictions = []
    for pred in prediction_list:
        prediction = np.load(pred)
        print(prediction.shape)
        predictions.append(prediction)
        
    all_pred = np.concatenate(tuple(predictions),axis=0)
    print(all_pred.shape)
    
    all_pred = np.mean(all_pred, axis=0)
    print(all_pred.shape)
    final_predictions = np.argmax(all_pred, axis=1)
    print(final_predictions.shape)
    
    csv_name_fold = f"softvoting_one_piece_weighted_rank_01.csv" # 변경 필요 : 자신이 원하는 csv파일명으로 변경 해야함
    result_info = test_info.copy()
    result_info['target'] = final_predictions 
    result_info = result_info.reset_index().rename(columns={"index": "ID"})
    result_info.to_csv(csv_name_fold, index=False)

# 동일하게 python hard_voting.py로 실행!
if __name__ == "__main__":
    soft_soft()
    # soft_hard()
    # hard_hard()
