import torch
import torch.nn as nn
import os

feature_size = 112
chunk_size = 112  # 19분 단위로 capture
train_radio = 0.6
num_chunks = 30
batch_size = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
learning_rate = 1e-4
epochs = 100
criterion = nn.CrossEntropyLoss()
alpha = 0.2
graph = {
    "train_hist_loss": [],
    "val_hist_loss": [],
    "train_hist_acc": [],
    "val_hist_acc": [],
}

host_process = 'lphost06_wls1'

# 현재 작업 디렉토리 얻기
current_path = os.getcwd()
directory_path = os.path.join(current_path, 'Preprocessed_data2')
result_path = os.path.join(current_path, 'Result')