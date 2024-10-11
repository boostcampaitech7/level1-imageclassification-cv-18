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

def hard_voting(csv_files:list):
    '''
    csv_files : 
    soft_voting / hard_voting이 되어 있는 prediction 들이 저장된 csv_file 경로 list
    '''
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


def soft_voting(prediction_list, test_csv_path = "/data/ephemeral/home/data/test.csv", csv_name_fold = f"softvoting_one_piece_weighted_rank_01.csv"):
    
    """
    prediction_list : prediction 결과가 저장되어있는 npy 파일

    Usage:
    prediction_list = [
        "main/score_vector/soft-soft/all_fold_eva_giant_mlp_gelu.npy",
        "main/score_vector/soft-soft/all_fold_eva_large_curriculum_mlp_gelu.npy",
        "main/score_vector/soft-soft/all_fold_eva_large_mlp.npy"
    ]
    """
    
    test_info = pd.read_csv(test_csv_path)
    predictions = []
    for pred in prediction_list:
        prediction = np.load(pred)
        predictions.append(prediction)
        
    all_pred = np.concatenate(tuple(predictions),axis=0)
    all_pred = np.mean(all_pred, axis=0)
    final_predictions = np.argmax(all_pred, axis=1)
    
    result_info = test_info.copy()
    result_info['target'] = final_predictions 
    result_info = result_info.reset_index().rename(columns={"index": "ID"})
    result_info.to_csv(csv_name_fold, index=False)

# 동일하게 python hard_voting.py로 실행!
if __name__ == "__main__":

    prediction_list = [
    "main/score_vector/soft-soft/all_fold_eva_giant_mlp_gelu.npy",
    "main/score_vector/soft-soft/all_fold_eva_large_curriculum_mlp_gelu.npy",
    "main/score_vector/soft-soft/all_fold_eva_large_mlp.npy"
    ]
    
    soft_voting(prediction_list)