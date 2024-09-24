# 오후에 작성 예정. 하나의 weight 불러와서 test 하는 코드.
import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    datapath = "Experiments/eva_large/test_csv"
    data = np.load(os.path.join(datapath, f'fold1_predictions.npy')) # 데이터 로드. @파일명

    final_predictions = np.argmax(data, axis=1)

    # test_info의 복사본을 사용하여 CSV 저장
    test_csv_path = "./../../../data/test.csv"
    test_info = pd.read_csv(test_csv_path)

    csv_name = "test.csv"
    result_info = test_info.copy()
    result_info['target'] = final_predictions 
    result_info = result_info.reset_index().rename(columns={"index": "ID"})

    save_path = os.path.join(datapath, csv_name)
    result_info.to_csv(save_path, index=False)
