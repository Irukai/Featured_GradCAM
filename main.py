import os
import numpy as np
import shap
from modules.data_loader import load_data, check_data
from modules.utils import clear_gpu_cache, plot_graph, show_data, transform_data, evaluate_model_on_dataloader
from modules.models import GradCAM, Featred_GradCAM, Shapley
from modules.dataset import CustomDataset, make_data, make_dataloader
from modules.train import train_model, train_model_ours, evaluate_model
from modules.config import (
    graph,
    device,
    feature_size,
    chunk_size,
    criterion,
    num_chunks,
    directory_path,
    result_path,
    host_process
)
from modules.gradcam import show_attributions, gcam_plot, fgcam_plot, generate_gradcam_labels

import time
from torchsummary import summary
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader
from lightgbm import LGBMClassifier, plot_importance

import warnings

warnings.filterwarnings("ignore")


def main():
    # Clear GPU cache
    clear_gpu_cache()
    print('Device :', device)

    ### Load data
    print('파일 불러오는 중...')
    rolled_one_dfs, rolled_one_labels, all_features = load_data(directory_path)
    check_data(rolled_one_dfs)
    print('파일 불러오기 완료\n\n')

    ### Select Features
    print('특징 선택 중...')

    ### Generate dataset and dataloader
    print('데이터 생성 중...')
    data_collection = {"chunk_dataset": [], "labels": [], "timestamp": []}
    # anomaly_indices
    anomaly_indices = make_data(
        rolled_one_dfs,
        rolled_one_labels,
        data_collection,
    )

    # # GradCAM Labeling을 위해 이상 지점을 저장
    # highlight_positions = []                                               
    # time_pos = np.argmax(data_collection['timestamp'], axis=1)
    # for i in range(num_chunks):
    #     highlight_positions.append([anomaly_indices[i], time_pos[i]])

    # err_pos = transform_data(highlight_positions)
    # gradcam_data = np.zeros((num_chunks, 21, 21))

    # GradCAM Label 데이터 생성
    # gradcam_labels = generate_gradcam_labels(gradcam_data, err_pos, cross_length=1)

    # 첫 번째 이미지 (GradCAM Label 데이터 중 하나) 시각화
    # plt.figure(figsize=(6, 6))
    # plt.imshow(gradcam_labels[0].numpy())
    # plt.title("GradCAM Label Data [Sample]")
    # plt.colorbar()
    # 파일을 저장할 전체 경로 생성
    # save_path = os.path.join(result_path, 'gradcamlabel.jpeg')

    # Save the figure as a JPEG file
    # plt.savefig(save_path, format='jpeg')

    dataset = CustomDataset(data_collection)
    train_loader, val_loader, test_loader, data_size_dict = make_dataloader(dataset)

    feature, target, times = next(iter(train_loader))          # chunk, label, time, index, gcam_label
    print("첫 번째 샘플의 크기:", feature.size(), target.size(), times.size())

    # gradcam 비교를 위한 샘플 데이터 선택
    x = data_collection['chunk_dataset'][0].unsqueeze(0).unsqueeze(1).to(device)
    y = data_collection['labels'][0].unsqueeze(0).to(device)

    show_data(data_collection, anomaly_indices)
    print('데이터 생성 완료\n\n')


    ### Initialize GradCAM model
    print('GradCAM 모델 생성 중...')
    gradcam = GradCAM(feature_size, chunk_size).to(device)
    summary(gradcam, (1, feature_size, chunk_size), device="cuda")

    print('\nGradCAM 모델 학습 진행')
    # Train the model
    train_model(gradcam, train_loader, val_loader, data_size_dict, graph)
    # Evaluate the model
    evaluate_model(gradcam, test_loader, data_size_dict)

    print('\n\n')

    # precision, recall, f1 = evaluate_model_on_dataloader(gradcam, test_loader, criterion, device, 0)
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1-score: {f1:.4f}')


    ### Initialize GradCAM model
    shapley_model = Shapley(feature_size, chunk_size).to(device)
    summary(shapley_model, (1, feature_size, chunk_size), device='cuda')

    print('\nShapley Value 모델 학습 진행')
    # Train the model
    train_model(shapley_model, train_loader, val_loader, data_size_dict, graph)
    # Evaluate the model
    evaluate_model(shapley_model, test_loader, data_size_dict)

    print('\n\n')

    sv_data, sv_label, _,= next(iter(train_loader))
    d = sv_data[0].unsqueeze(1)
    explainer = show_attributions(shapley_model, x, y, d)

    # expl = shap.DeepExplainer(shapley_model, d.to(device))
    
    # precision, recall, f1 = evaluate_model_on_dataloader(explainer, test_loader, criterion, device, 1)
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1-score: {f1:.4f}')


    ### Initialize Featured GradCAM model
    print('\nFeatured GradCAM 모델 생성 중...')
    f_gradcam = Featred_GradCAM(feature_size, chunk_size).to(device)
    summary(f_gradcam, (1, feature_size, chunk_size), device="cuda")
    # Train the model
    print('\nFeatured GradCAM 모델 학습 진행')
    train_model_ours(f_gradcam, train_loader, val_loader, data_size_dict, graph)
    # Evaluate the model
    evaluate_model(f_gradcam, test_loader, data_size_dict)

    print('\n\n')

    # precision, recall, f1 = evaluate_model_on_dataloader(f_gradcam, test_loader, criterion, device, 2)
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    # print(f'F1-score: {f1:.4f}')


    ### GradCAM activations map 생성 및 저장
    print('GradCAM 활성화 맵 계산 중...')
    gc_activations = gradcam.get_activations(x, y, criterion)
    ours_gcam = f_gradcam.get_concat_activation(x, y, criterion).detach().cpu().numpy()
    print('GradCAM 활성화 맵 계산 완료')
    
    gcam_plot(x, gc_activations)
    fgcam_plot(x, ours_gcam)
    print('Result 디렉터리에 결과 이미지 저장 완료')


if __name__ == "__main__":
    main()
