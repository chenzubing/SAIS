import h5py
import os

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
    model = SkillAssessmentModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建保存模型的目录
    save_dir = os.path.join('mock_VUA', 'Results', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    num_epochs = 50
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
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
            
            for batch in tqdm(dataloaders[phase], desc=phase):
                dino_features = batch['dino'].to(device)
                labels = batch['label'].squeeze().to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(dino_features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * dino_features.size(0)
                running_corrects += torch.sum(preds == labels)
                total_samples += dino_features.size(0)
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples * 100
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved new best model with accuracy: {best_acc:.2f}%')
    
    print(f'\nBest validation accuracy: {best_acc:.2f}%')
    print(f'Best model saved to: {best_model_path}')
    
    return model, best_acc

if __name__ == '__main__':
    model, best_acc = train_model()
    print(f'\nTraining completed. Best validation accuracy: {best_acc:.2f}%')