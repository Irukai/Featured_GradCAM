
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import min_max_normalize

class GradCAM(nn.Module):
    def __init__(self, feature_size, chunk_size):
        super(GradCAM, self).__init__()
       
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
    
        self.classifier = nn.Sequential(
            nn.Linear(32 * (feature_size-6) * (chunk_size-6), 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.gradients = None

    def forward(self, x):
        h1 = self.layer1(x)
        h1.register_hook(self.activations_hook)

        x = h1.view(h1.size(0), -1)
        x = self.classifier(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x, y, loss_gcam):
        pred_g = self.forward(x)
        loss_g = loss_gcam(pred_g, y)
        loss_g.backward()

        pooled_grad = torch.mean(self.gradients, dim=[2, 3]) # (b, 32, 15, 15) -> (b, 32)     # (neuron importance weight, a)

        activations = self.layer1(x)
        for b in range(activations.size(0)):
            for i in range(activations.size(1)): # 각 feature 마다 weight와 곱하기
                activations[b, i, :, :] *= pooled_grad[b, i]
                
        heatmap = torch.sum(activations, dim=1)
        heatmap = F.relu(heatmap)
    
        # 데이터 정규화
        # normalize_heatmap = min_max_normalize(heatmap)
        return heatmap
    
class Shapley(nn.Module):
    def __init__(self, feature_size, chunk_size):
        super(Shapley, self).__init__()

        def CBALayer(in_channels, out_channels, kernel_size, stride, padding):
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU()
            )
            return layer
        
        self.conv1 = CBALayer(1, 8, 3, 1, 0)
        self.conv2 = CBALayer(8, 16, 3, 1, 0)
        self.conv3 = CBALayer(16, 32, 3, 1, 0)
        
        self.fc1 = nn.Linear(32 * (feature_size-6) * (chunk_size-6), 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
# Featured CNN 모델 정의
class Featred_GradCAM(nn.Module):
    def __init__(self, feature_size, chunk_size):
        super(Featred_GradCAM, self).__init__()
        
        self.feature_size = feature_size
        self.chunk_size = chunk_size

        self.feature_layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3,1)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3,1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,1)),
            nn.ReLU(),
        )

        self.time_layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1,3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(1,3)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1,3)),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * (self.feature_size-6) * self.chunk_size, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.gradients1 = None
        self.gradients2 = None


    def forward(self, x):
        x1 = self.feature_layer1(x)
        
        x2 = self.time_layer1(x)

        h1 = x1.register_hook(self.activations_hook1)
        h2 = x2.register_hook(self.activations_hook2)

        x2 = x2.permute(0,1,3,2)
        x = torch.concat((x1,x2), dim=1)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def activations_hook1(self,grad):
        self.gradients1 = grad

    def activations_hook2(self,grad):
        self.gradients2 = grad

    def get_activations_gradient1(self):
        return self.gradients1

    def get_activations_gradient2(self):
        return self.gradients2

    def get_activations1(self, x, y, loss_gcam): # 가로로 긴 heatmap 생성

        pred_g = self.forward(x)
        loss_g = loss_gcam(pred_g, y)
        loss_g.backward()

        pooled_grad_v1 = torch.mean(self.gradients1, dim=[2, 3]) # (1, 32, 21, 15) -> (32)     # (neuron importance weight, a)

        fx_ac = self.feature_layer1(x)
        new_fx_ac = fx_ac.clone()  # 기존 텐서의 사본을 생성

        for b in range(new_fx_ac.size(0)):  # 각 배치마다
            for i in range(new_fx_ac.size(1)): # 각 feature 마다 weight와 곱하기
                new_fx_ac[b, i, :, :] *= pooled_grad_v1[b, i]

        f_heatmap = torch.sum(new_fx_ac, dim=1)
        f_heatmap = F.relu(f_heatmap)
        
        # 데이터 정규화
        # normalize_f_heatmap = min_max_normalize(f_heatmap)
        return f_heatmap
        
    def get_activations2(self, x, y, loss_gcam): # 세로로 긴 heatmap 생성
        pred_g = self.forward(x)
        loss_g = loss_gcam(pred_g, y)
        loss_g.backward()

        pooled_grad_v2 = torch.mean(self.gradients2, dim=[2, 3]) # (1, 32, 15, 21) -> (32)     # (neuron importance weight, a)

        tx_ac = self.time_layer1(x)
        new_tx_ac = tx_ac.clone()  # 기존 텐서의 사본을 생성

        for b in range(new_tx_ac.size(0)):  # 각 배치마다
            for i in range(new_tx_ac.size(1)): # 각 feature 마다 weight와 곱하기
                new_tx_ac[b, i, :, :] *= pooled_grad_v2[b, i]

        t_heatmap = torch.sum(new_tx_ac, dim=1)
        t_heatmap = F.relu(t_heatmap)
        
        # 데이터 정규화
        # normalize_t_heatmap = min_max_normalize(t_heatmap)
        return t_heatmap


    def get_concat_activation(self, x, y, criterion):
        feature_gc = self.get_activations1(x, y, criterion) # 가로
        time_gc = self.get_activations2(x, y, criterion) # 세로

        feature_gc = F.interpolate(feature_gc.unsqueeze(1), size=(self.feature_size, self.chunk_size), mode='bilinear')
        time_gc = F.interpolate(time_gc.unsqueeze(1), size=(self.feature_size, self.chunk_size), mode='bilinear')

        # concat_gc = min_max_normalize(time_gc+feature_gc)
        concat_gc = time_gc + feature_gc
        
        return concat_gc