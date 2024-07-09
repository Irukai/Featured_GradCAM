import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
from modules.config import device, epochs, learning_rate, alpha

def train_model(model, train_loader, val_loader, data_size, graph):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion_gc = nn.MSELoss()

    # 모델 학습
    for epoch in range(epochs+1):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        for tr_x, tr_y, _, in train_loader:
            tr_x = tr_x.to(device)
            tr_y = tr_y.to(device)
                    
            pred = model(tr_x)
            
            loss = criterion(pred, tr_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            tr_pred = torch.argmax(pred, axis=1)
            tr_label = torch.argmax(tr_y, axis=1)

            tr_pred = tr_pred.detach().cpu().numpy()
            tr_label = tr_label.detach().cpu().numpy()
            correct = sum(tr_pred == tr_label)

            train_acc += correct

        for val_x, val_y, _ in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            outputs = model(val_x)
            
            loss = criterion(outputs, val_y)
            val_loss += loss.item()

            val_pred = torch.argmax(outputs, axis=1)
            val_label = torch.argmax(val_y, axis=1)

            val_pred = val_pred.detach().cpu().numpy()
            val_label = val_label.detach().cpu().numpy()
            correct = sum(val_pred == val_label)

            val_acc += correct

        train_avg_loss = train_loss/data_size['train_size']
        graph['train_hist_loss'].append(train_avg_loss)

        val_avg_loss = val_loss/data_size['val_size']
        graph['val_hist_loss'].append(val_avg_loss)

        train_avg_acc = train_acc/data_size['train_size']
        val_avg_acc = val_acc/data_size['val_size']

        graph['train_hist_acc'].append(train_avg_acc)
        graph['val_hist_acc'].append(val_avg_acc)


        if (epoch) % 10 == 0:
            print(f'Epoch [{epoch}], Train Loss: {train_avg_loss:.5f}, \t Val Loss: {val_avg_loss:.5f}, \n \
                Train Accuracy : {train_avg_acc*100:.2f}%, \t Val Accuracy : {val_avg_acc*100:.2f}%')
            
        if val_avg_acc*100 >= 98 and epoch > 10:
            print(f'Epoch [{epoch}], Train Loss: {train_avg_loss:.5f}, \t Val Loss: {val_avg_loss:.5f}, \n \
                Train Accuracy : {train_avg_acc*100:.2f}%, \t Val Accuracy : {val_avg_acc*100:.2f}%')
            break

def train_model_ours(model, train_loader, val_loader, data_size, graph):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_class = nn.CrossEntropyLoss()
    criterion_gc = nn.MSELoss()

    # 모델 학습
    for epoch in range(epochs+1):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        for tr_x, tr_y, _, in train_loader:
            tr_x = tr_x.to(device)
            tr_y = tr_y.to(device)
            # trgc_y = trgc_y.to(device)

            pred = model(tr_x)
            # concat_gc = model.get_concat_activation(tr_x, tr_y, criterion_class)

            loss_class = criterion_class(pred, tr_y)
            # loss_gc = criterion_gc(concat_gc, trgc_y)
            # loss_total = (alpha * loss_class + (1-alpha)*loss_gc)
            loss_total = loss_class
            

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            train_loss += loss_total.item()

            tr_pred = torch.argmax(pred, axis=1)
            tr_label = torch.argmax(tr_y, axis=1)

            tr_pred = tr_pred.detach().cpu().numpy()
            tr_label = tr_label.detach().cpu().numpy()
            correct = sum(tr_pred == tr_label)

            train_acc += correct

        for val_x, val_y, _, in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            # valgc_y = valgc_y.to(device)

            outputs = model(val_x)
            
            # concat_gc = model.get_concat_activation(val_x, val_y, criterion_class)
            
            loss_class = criterion_class(outputs, val_y)
            # loss_gc = criterion_gc(concat_gc, valgc_y)
            # loss_total = (alpha * loss_class + (1-alpha)*loss_gc)
            loss_total = loss_class

            val_loss += loss_total.item()

            val_pred = torch.argmax(outputs, axis=1)
            val_label = torch.argmax(val_y, axis=1)

            val_pred = val_pred.detach().cpu().numpy()
            val_label = val_label.detach().cpu().numpy()
            correct = sum(val_pred == val_label)

            val_acc += correct

        train_avg_loss = train_loss/data_size['train_size']
        graph['train_hist_loss'].append(train_avg_loss)

        val_avg_loss = val_loss/data_size['val_size']
        graph['val_hist_loss'].append(val_avg_loss)

        train_avg_acc = train_acc/data_size['train_size']
        val_avg_acc = val_acc/data_size['val_size']

        graph['train_hist_acc'].append(train_avg_acc)
        graph['val_hist_acc'].append(val_avg_acc)


        if (epoch) % 10 == 0:
            print(f'Epoch [{epoch}], Train Loss: {train_avg_loss:.5f}, \t Val Loss: {val_avg_loss:.5f}, \n \
                Train Accuracy : {train_avg_acc*100:.2f}%, \t Val Accuracy : {val_avg_acc*100:.2f}%')
            
        if val_avg_acc*100 >= 98 and epoch > 10:
            print(f'Epoch [{epoch}], Train Loss: {train_avg_loss:.5f}, \t Val Loss: {val_avg_loss:.5f}, \n \
                Train Accuracy : {train_avg_acc*100:.2f}%, \t Val Accuracy : {val_avg_acc*100:.2f}%')
            break

def evaluate_model(model, data_loader, data_size):
    test_acc = 0
    for i, (data_, labels_, _,) in enumerate(data_loader):

        data_ = data_.to(device)
        labels_ = labels_.to(device)
        
        target_ = model(data_)
    
        pred_ = torch.argmax(target_, axis=1)
        labels_ = torch.argmax(labels_, axis=1)

        pred_ = pred_.detach().cpu().numpy()
        labels_ = labels_.detach().cpu().numpy()
        correct = sum(labels_ == pred_)

        test_acc += correct

    test_size = data_size['test_size'] 
    print(f'Test Accuracy : {((test_acc / test_size) * 100):.3f}%')