# SAIS/scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prepare_dataset import loadDataloader
import numpy as np
from tqdm import tqdm

class SkillAssessmentModel(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, num_classes=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(0, 1)  # TransformerEncoder expects [seq_len, batch_size, input_dim]
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=0)  # Global average pooling
        return self.classifier(x)

def train_model():
    # 设置设备
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    data_params = {
        'root_path': 'mock_VUA',
        'dataset_name': 'VUA_EASE',
        'data_type': 'reps',
        'batch_size': 4,
        'nclasses': 2,
        'domain': 'NH_02',
        'phases': ['train', 'val'],
        'task': 'Prototypes',
        'balance': True,
        'balance_groups': False,
        'single_group': False,
        'group_info': {},
        'importance_loss': False
    }
    
    dataloaders = loadDataloader(**data_params).load()
    
    # 初始化模型
    model = SkillAssessmentModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(dataloaders['train'], desc='Training'):
            videoname, features, flows, labels, frames_importance, domains = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc='Validation'):
                videoname, features, flows, labels, frames_importance, domains = batch
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Train Loss: {train_loss/len(dataloaders["train"]):.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(dataloaders["val"]):.3f} | Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'mock_VUA/Results/best_model.pth')

if __name__ == '__main__':
    train_model()