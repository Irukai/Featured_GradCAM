import os
import pandas as pd

from modules.config import host_process

def load_data(directory_path):
    rolled_one_dfs = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            key = filename.split('.')[0]  # 파일명에서 확장자를 제거하여 키로 사용
            csv_filepath = os.path.join(directory_path, filename)
            df = pd.read_csv(csv_filepath)
            rolled_one_dfs[key] = df
            print(f"Loaded {csv_filepath} into rolled_one_dfs[{key}]")

    rolled_one_labels = dict()
    for key in rolled_one_dfs.keys():
        rolled_one_labels[key] = rolled_one_dfs[key].iloc[:, 0]
        rolled_one_labels[key].astype(int)

    rolled_one_dfs[host_process] = rolled_one_dfs[host_process].iloc[:, 1:]
    features_name = rolled_one_dfs[host_process].columns
    rolled_one_dfs[host_process].columns = range(rolled_one_dfs[host_process].shape[1]) # column name을 index로 변경

    return rolled_one_dfs, rolled_one_labels, features_name

def check_data(rolled_one_dfs):
    for key, df in rolled_one_dfs.items():
        print(f"{key}: {df.shape}")
