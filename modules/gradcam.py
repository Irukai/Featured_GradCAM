import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
from modules.config import chunk_size, result_path, device
from modules.utils import min_max_normalize
from PIL import Image


def generate_cross_mask(shape, center, length):
    """
    십자가 모양의 마스크를 생성한다 (중심에서 멀어질수록 약하게 강조)
    
    Args:
    - shape (tuple): 마스크의 크기 (height, width)
    - center (tuple): 십자가 중심의 (row, col) 위치
    - length (int): 십자가의 팔 길이
    
    Returns:
    - mask (numpy.ndarray): 십자가 모양의 강조 마스크
    """
    height, width = shape
    center_row, center_col = center
    mask = np.zeros((height, width))
    
    for r in range(max(center_row - length, 0), min(center_row + length + 1, height)):
        distance = abs(center_row - r)
        intensity = 1 - (distance / length)
        mask[r, center_col] = intensity
    
    for c in range(max(center_col - length, 0), min(center_col + length + 1, width)):
        distance = abs(center_col - c)
        intensity = 1 - (distance / length)
        mask[center_row, c] = intensity
    
    return mask

def generate_gradcam_labels(input_images, highlight_positions, cross_length=4):
    """
    입력 이미지를 받아 특정 위치에 십자가 모양으로 강조한 이미지를 생성하는 함수.
    
    Args:
    - input_images (numpy.ndarray): (1000, 21, 21) shape의 입력 이미지
    - highlight_positions (list): (1000, 강조할 위치 (row, col))의 리스트
    - cross_length (int): 십자가의 팔 길이
    
    Returns:
    - output_images (numpy.ndarray): 강조된 부분이 있는 이미지들과 동일한 shape
    """
    output_images = input_images.copy()
    height, width = input_images.shape[1], input_images.shape[2]  
    
    for i, img in enumerate(output_images):
        if highlight_positions[i]: # 이상 포인트가 있는 경우라면
            for center in highlight_positions[i]:
                mask = generate_cross_mask((height, width), center, cross_length)
                img += mask  # 강조 마스크를 이미지에 추가
    
    tensor_img = torch.tensor(output_images, dtype=torch.float32)
    return tensor_img



def gcam_plot(original, activations,):
    fig = plt.figure(figsize=(15, 6))

    fig.add_subplot(1, 2, 1)
    plt.imshow(original.squeeze().squeeze().detach().cpu().numpy())
    plt.title('data')
    plt.xlabel('time')
    plt.ylabel('features')

    fig.add_subplot(1, 2, 2)
    gc = F.interpolate(activations.unsqueeze(1), size=(chunk_size, chunk_size))
    plt.imshow(gc[0, 0].detach().cpu().numpy())
    plt.title('gradcam')
    plt.xlabel('time')
    plt.ylabel('features')

    # 파일을 저장할 전체 경로 생성
    save_path = os.path.join(result_path, 'gracam_result.jpeg')

    # Save the figure as a JPEG file
    plt.savefig(save_path, format='jpeg')


def fgcam_plot(original, gcam):
    fig = plt.figure(figsize=(15, 6))

    fig.add_subplot(1, 2, 1)
    plt.imshow(original.squeeze().squeeze().detach().cpu().numpy())
    plt.title('data')
    plt.xlabel('time')
    plt.ylabel('features')

    fig.add_subplot(1, 2, 2)
    plt.imshow(gcam.squeeze().squeeze())
    plt.title('Featured_gradcam')
    plt.xlabel('time')
    plt.ylabel('features')

    # 파일을 저장할 전체 경로 생성
    save_path = os.path.join(result_path, 'featured_gracam_result.jpeg')

    # Save the figure as a JPEG file
    plt.savefig(save_path, format='jpeg')


import shap

def show_attributions(model, td, tl, d):
    '''
    model : shapely values model
    td : test data      # shape : (1, 1, 21, 21)
    tl : test label     # shape : (1, 2)
    d : original data (e.g., training data for background distribution) # shape : (1, 1, 21, 21)
    '''
    # Predict the probabilities of the digits using the test images
    output = model(td.to(device))
    # Get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1] 
    # Convert to numpy only once to save time
    pred_np = pred.detach().cpu().numpy() 

    expl = shap.GradientExplainer(model, d.to(device))

    # (1, 1, 21, 21)
    shap_values = expl.shap_values(td)
    sn = [np.swapaxes(np.swapaxes(shap_values[..., i], 1, -1), 1, 2) for i in range(shap_values.shape[-1])]
    ti = np.swapaxes(np.swapaxes(td.detach().cpu().numpy(), 1, -1), 1, 2)
    
    # SHAP 값의 형식 확인
    print(f"Number of classes: {len(sn)}")
    print(f"SHAP values shape for each class: {sn[0].shape}")

    # Prepare to augment the plot
    plt.figure()
    shap.image_plot(sn, ti)

    # 파일을 저장할 전체 경로 생성
    save_path = os.path.join(result_path, 'shapely_value_result.jpeg')
    plt.savefig(save_path, format='jpeg')
    # # Clear the current figure to avoid overlap
    # plt.clf()
    return expl