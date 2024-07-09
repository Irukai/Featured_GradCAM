import random
from torch.utils.data import Dataset, DataLoader, random_split
from modules.config import (
    chunk_size,
    num_chunks,
    train_radio,
    batch_size,
    feature_size,
    host_process
)
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, data_collection):
        self.data = data_collection['chunk_dataset']
        self.data = self.data.unsqueeze(1)
        self.labels = data_collection['labels']
        self.timestamp = data_collection['timestamp']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        chunk = self.data[idx]
        label = self.labels[idx]
        time = self.timestamp[idx]

        return chunk, label, time 



def make_data(signal, labels, data_collection):
    global chunk_size, num_chunks
    
    anomaly_indices = []
    for _ in range(num_chunks//2):
        while True:
            idx = np.random.randint(0, len(signal[host_process]) - chunk_size)
            # 이상 데이터가 없으면 True
            if (labels[host_process][idx : idx + chunk_size].sum() == 0):
                normal_chunk = np.array(signal[host_process].iloc[idx : idx + chunk_size].T)
            
                num_err = 1 # np.random.randint(1, 4)
                pos_err = random.sample(list(range(chunk_size + 1)), num_err)
                anomaly_indices.append(pos_err)

                # 추가할 err 개수 만큼 반복
                for i in range(num_err):
                    # 이상 데이터가 있으면
                    while True:
                        idx = np.random.randint(0, len(signal[host_process]) - chunk_size)
                        if (labels[host_process][idx : idx + chunk_size].sum() > 0):
                            ano_chunk = np.array(signal[host_process].iloc[idx : idx + chunk_size].T) # (133, 133) f-t
                            normal_chunk[pos_err[i], :] = ano_chunk[pos_err[i], :]
                            break
                        
                mixed_ano_chunk = normal_chunk.copy()
                # 편집된 이상 데이터를 저장
                data_collection["chunk_dataset"].append(mixed_ano_chunk)
                data_collection["timestamp"].append(labels[host_process][idx : idx + chunk_size])
                data_collection["labels"].append([0, 1])  # 이상
                break

    while len(data_collection["labels"]) < num_chunks:  # 10~20
        idx = np.random.randint(0, len(signal[host_process]) - chunk_size)
        # 이상 데이터가 없으면
        if (labels[host_process][idx : idx + chunk_size].sum() == 0):  
            chunk = np.array(signal[host_process].iloc[idx : idx + chunk_size].T)
            data_collection["chunk_dataset"].append(chunk)
            data_collection["timestamp"].append(labels[host_process][idx : idx + chunk_size])
            data_collection["labels"].append([1, 0])  # 정상
            anomaly_indices.append(np.array([-1]))

    data_collection["chunk_dataset"] = torch.tensor(np.array(data_collection["chunk_dataset"]), dtype=torch.float32)
    data_collection["labels"] = torch.tensor(np.array(data_collection["labels"]), dtype=torch.float32)
    data_collection["timestamp"] = torch.tensor(np.array(data_collection["timestamp"]))

    return anomaly_indices

    # # 배열 요소들을 길이 4로 패딩
    # max_len = 4
    # padded_data = []
    # for arr in anomaly_indices:
    #     padded_arr = np.pad(arr, (0, max_len - len(arr)), constant_values=-1)
    #     padded_data.append(padded_arr)

    # return padded_data


def make_dataloader(dataset):
    data_size = len(dataset)
    train_size = int(data_size * train_radio)
    val_size = int(data_size * ((1 - train_radio) / 2))
    test_size = data_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    data_size_dict = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }
    return train_loader, val_loader, test_loader, data_size_dict
