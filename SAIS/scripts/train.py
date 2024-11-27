import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import create_dataloaders

class SkillAssessmentModel(nn.Module):
    def __init__(self, dino_dim=384, flow_dim=10, hidden_dim=256, num_classes=2):
        super().__init__()
        
        # DINO特征处理
        self.dino_encoder = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.dino_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.3,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Flow特征处理 - 使用更简单的架构
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def _safe_mean(self, x, mask=None):
        """Safely compute mean over sequence dimension with optional mask"""
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            # Ensure we don't divide by zero by clamping the denominator
            return x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        return x.mean(dim=1)
    
    def _check_empty_sequence(self, features):
        """Check if all sequences in the batch are empty"""
        return (torch.sum(torch.abs(features).view(features.size(0), -1), dim=1) == 0).all()

    def forward(self, dino_features, flow_features):
        batch_size = dino_features.size(0)
        
        # Create padding masks
        dino_mask = torch.sum(torch.abs(dino_features), dim=-1) == 0
        flow_mask = torch.sum(torch.abs(flow_features), dim=-1) == 0
        
        # Process DINO features
        dino_encoded = self.dino_encoder(dino_features)
        if not self._check_empty_sequence(dino_features):
            dino_out = self.dino_transformer(dino_encoded, src_key_padding_mask=dino_mask)
        else:
            dino_out = dino_encoded
        dino_pooled = self._safe_mean(dino_out, dino_mask)
        
        # Process Flow features - simpler processing without transformer
        flow_encoded = self.flow_encoder(flow_features)
        flow_pooled = self._safe_mean(flow_encoded, flow_mask)
        
        # Combine features
        combined = torch.cat([dino_pooled, flow_pooled], dim=1)
        fused = self.fusion(combined)
        
        return self.classifier(fused)

def train_model():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(device)} (GPU)")
    else:
        print(f"Using device: {device} (CPU)")
    
    # 设置数据加载参数
    data_params = {
        'root_path': 'mock_VUA/Results',
        'encoder_params': 'ViT_SelfSupervised_ImageNet',
        'batch_size': 4
    }
    
    # 检查特征文件是否存在
    rgb_file = os.path.join(data_params['root_path'], f"{data_params['encoder_params']}_RepsAndLabels.h5")
    flow_file = os.path.join(data_params['root_path'], f"{data_params['encoder_params']}_Flow_RepsAndLabels.h5")
    
    print(f"\nChecking feature files:")
    print(f"DINO features: {rgb_file}")
    print(f"Flow features: {flow_file}")
    
    if not os.path.exists(rgb_file) or not os.path.exists(flow_file):
        print(f"\nWarning: Feature files not found at expected locations!")
        print("Please make sure the feature extraction step has completed successfully.")
        return None, 0
    
    # 加载数据
    dataloaders = create_dataloaders(**data_params)
    
    # 初始化模型
    model = SkillAssessmentModel(
        dino_dim=384,
        flow_dim=10,
        hidden_dim=256
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # 创建保存模型的目录
    save_dir = os.path.join('mock_VUA', 'Results', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练参数
    num_epochs = 50
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    patience = 10
    no_improve = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            pbar = tqdm(dataloaders[phase], desc=phase)
            for batch in pbar:
                # 移动数据到设备
                dino_features = batch['dino'].to(device)
                flow_features = batch['flow'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(dino_features, flow_features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * dino_features.size(0)
                running_corrects += torch.sum(preds == labels)
                total_samples += dino_features.size(0)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * torch.sum(preds == labels) / len(labels):.2f}%'
                })
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples * 100
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
            
            # 在验证集上更新学习率调度器和早停
            if phase == 'val':
                scheduler.step(epoch_acc)
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                    }, best_model_path)
                    print(f'Saved new best model with accuracy: {best_acc:.2f}%')
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f'\nEarly stopping triggered after {patience} epochs without improvement')
                        return model, best_acc
    
    print(f'\nBest validation accuracy: {best_acc:.2f}%')
    print(f'Best model saved to: {best_model_path}')
    
    return model, best_acc

if __name__ == '__main__':
    model, best_acc = train_model()
    print(f'\nTraining completed. Best validation accuracy: {best_acc:.2f}%')