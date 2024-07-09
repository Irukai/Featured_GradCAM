import torch
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import precision_score, recall_score, f1_score
from modules.config import result_path, chunk_size, device, host_process, num_chunks
import warnings
import os
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

def min_max_normalize(array, is_tensor=True):
    eps = np.finfo(float).eps  # 아주 작은 값

    if is_tensor:
        if array.dim() == 3:  # (1000, 21, 21)
            min_val = torch.amin(array, dim=(1, 2), keepdim=True)
            max_val = torch.amax(array, dim=(1, 2), keepdim=True)
        elif array.dim() == 4:  # (batch, channels, height, width)
            min_val = torch.amin(array, dim=(2, 3), keepdim=True)
            max_val = torch.amax(array, dim=(2, 3), keepdim=True)
        else:
            raise ValueError("Unexpected tensor shape. Expected a 3D or 4D tensor.")
        
        normalized_array = (array - min_val) / (max_val - min_val + eps)
    else:
        if array.ndim == 3:  # (1000, 21, 21)
            min_val = np.min(array, axis=(1, 2), keepdims=True)
            max_val = np.max(array, axis=(1, 2), keepdims=True)
        elif array.ndim == 4:  # (batch, channels, height, width)
            min_val = np.min(array, axis=(2, 3), keepdims=True)
            max_val = np.max(array, axis=(2, 3), keepdims=True)
        else:
            raise ValueError("Unexpected array shape. Expected a 3D or 4D array.")
        
        normalized_array = (array - min_val) / (max_val - min_val + eps)
        
    return normalized_array


def plot_graph(data, top_features):

    fig = plt.figure(figsize=(20, 16))

    # 데이터를 순회하면서 값이 1인 지점의 인덱스를 기록
    index_of_ones = [i for i, value in enumerate(data[0]) if abs(value - 1) < 1e-5]

    idx = 0
    for i in range(10):
        fig.add_subplot(3, 4, idx + 1)
        plt.plot(np.arange(0, len(data)), data[top_features.values[idx]])
        plt.title(top_features.values[idx])
        idx += 1
        # 값이 1인 인덱스에 세로 줄 그리기
        for j in index_of_ones:
            plt.axvspan(j, j, color="r", alpha=1)  # 세로 줄을 그림

    # 파일을 저장할 전체 경로 생성
    save_path = os.path.join(result_path, 'features_plot.jpeg')

    # Save the figure as a JPEG file
    plt.savefig(save_path, format='jpeg')


def show_data(data_collection, anomaly_indices):

    feature = data_collection['chunk_dataset']          # (batch, feature, time)
    target = data_collection['labels']
    time = data_collection['timestamp']

    for i in range(num_chunks):
        plt.figure(figsize=(16,8))
        plt.imshow(feature[i, :, :])
        
        err_point = torch.nonzero(time[i])

        if err_point.numel() != 0:
            # for e in err_point.squeeze().tolist():
            plt.hlines(y=anomaly_indices[i], xmin=0, xmax=chunk_size-1, colors="r", alpha=0.5)
            plt.vlines(x=err_point.squeeze().tolist(), ymin=0, ymax=feature[i].shape[0]-1, color="r", alpha=0.5)

        label = np.argmax(target[i])
        if label == 0:
            plt.title('Normal')
        else:
            plt.title('Anoamly')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.colorbar()
        save_path = os.path.join(result_path, f'data/data_plot_{i}.jpeg')
        plt.savefig(save_path, format='jpeg')

def transform_data(input_data):
    """
    주어진 데이터를 변환하여 [a, e], [b, e], [c, e], [d, e] 꼴의 데이터를 반환합니다.
    이때, a, b, c, d가 -1일 경우 해당 데이터는 제외합니다.
    
    Args:
        input_data (list of tuples): 각 요소는 ([a, b, c, d], e) 형식의 튜플입니다.
    
    Returns:
        torch.Tensor: 변환된 데이터를 포함하는 텐서
    """
    transformed_data = []

    for data_pair in input_data:
        trans_data = []
        data_array, e = data_pair
        # -1인 값은 제외하고 (텐서, e) 쌍을 만듬
        for element in data_array:
            if element != -1:
                trans_data.append((element, e.item()))
        transformed_data.append(trans_data)

    return transformed_data

def evaluate_cams(pred_cams, true_labels, threshold=0.5):
    """
    Evaluates the predicted CAMs against the true labels using Precision, Recall, and F1-score.
    
    Args:
    - pred_cams (torch.Tensor or np.ndarray): Predicted CAMs, shape (batch_size, 21, 21)
    - true_labels (torch.Tensor or np.ndarray): Ground truth labels, shape (batch_size, 21, 21)
    - threshold (float): Threshold to binarize the predicted CAMs

    Returns:
    - precision (float): Precision score
    - recall (float): Recall score
    - f1 (float): F1-score
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred_cams, torch.Tensor):
        pred_cams = pred_cams.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()

    # Binarize predicted CAMs based on the threshold
    bin_pred_cams = (pred_cams >= threshold).astype(int)

    # Flatten the arrays
    bin_pred_cams_flat = bin_pred_cams.flatten()
    true_labels_flat = true_labels.flatten()

    # Calculate precision, recall, and F1-score
    precision = precision_score(true_labels_flat, bin_pred_cams_flat, average='binary')
    recall = recall_score(true_labels_flat, bin_pred_cams_flat, average='binary')
    f1 = f1_score(true_labels_flat, bin_pred_cams_flat, average='binary')

    return precision, recall, f1

def evaluate_model_on_dataloader(gcam_model, test_dataLoader, criterion, device, mode, threshold=0.32):
    """
    Evaluates the model on a given dataloader and computes precision, recall, and F1-score.
    
    Args:
    - gcam_model: The trained Grad-CAM model
    - test_dataLoader: DataLoader for the test dataset
    - criterion: Loss function used during training
    - device: Device to perform computations on ('cpu' or 'cuda')
    - threshold: Threshold to binarize the predicted CAMs

    Returns:
    - precision (float): Precision score
    - recall (float): Recall score
    - f1 (float): F1-score
    """
    cam_pred = []
    cam_true = []

    for te_x, te_y, _, in test_dataLoader:
        # Binarize true CAMs based on the threshold
        bin_true_cams = (te_x >= threshold).type(torch.int16)

        te_x = te_x.to(device)
        te_y = te_y.to(device)

        if mode == 0:
            # Get activations from the Grad-CAM model
            activations = gcam_model.get_activations(te_x, te_y, criterion)
            gradcam = F.interpolate(activations.unsqueeze(1), size=(21, 21), mode='bilinear')

            # Append predictions and true labels
            cam_pred.append(gradcam.detach().cpu().numpy())
            cam_true.append(bin_true_cams.cpu().numpy())

        elif mode == 1:
            sv = np.array(gcam_model.shap_values(te_x))     # (batch, channel, height, width, class)
            shap_values = np.array(sv[..., 1])  # Store the SHAP values 
            # shap_values.resize(shap_values.shape[0], 21, 21)
            # Binarize predicted CAMs based on the threshold

            shap_values = min_max_normalize(shap_values, False)
            cam_pred.append(shap_values)
            cam_true.append(bin_true_cams.cpu().numpy())

        else:
            activations = gcam_model.get_concat_activation(te_x, te_y, criterion)
            # Binarize predicted CAMs based on the threshold

            cam_pred.append(activations.detach().cpu().numpy())
            cam_true.append(bin_true_cams.cpu().numpy())

    # Concatenate all predictions and labels
    cam_pred = np.concatenate(cam_pred, axis=0)
    cam_true = np.concatenate(cam_true, axis=0)

    # Evaluate CAMs
    precision, recall, f1 = evaluate_cams(cam_pred, cam_true, threshold=threshold)

    return precision, recall, f1