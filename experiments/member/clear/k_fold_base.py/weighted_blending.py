import os
import numpy as np
import pandas as pd

test_csv_path = "level1-imageclassification-cv-18\\clear\\k_fold_base.py\\Experiments\\eva_large\\test_csv"

def data_load(file_name):
    return np.load(file_name)

def return_weighted_blending(predictions_list, weights):
    if sum(weights) > 1:
        weights_sum_one = []
        for weight in weights:
            weights_sum_one.append(weight/sum(weights))
    return np.average(predictions_list, axis = 0, weights = weights)

if __name__ == "__main__":
    prediction_file_list = os.listdir(test_csv_path)
    predictions_list = []
    for file_name in prediction_file_list:
        predictions_list.append(data_load(os.path.join(test_csv_path,file_name)))
    predictions_list = np.array(predictions_list)
    weighted_blending = return_weighted_blending(predictions_list,weights = [1,1,1,1,1])

    # # 최종 예측값 결정
    # final_predictions = np.argmax(weighted_blending, axis=1)
    # test_info = pd.read_csv("data\\test.csv")
    # csv_name = "k-fold_ensemble.csv"
    # result_info = test_info.copy()
    # result_info['target'] = final_predictions 
    # result_info = result_info.reset_index().rename(columns={"index": "ID"})

    # save_path = os.path.join(test_csv_path, csv_name)
    # result_info.to_csv(save_path, index=False)