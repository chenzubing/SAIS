import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import numpy as np
from sklearn.model_selection import train_test_split

def custom_collate_fn(batch):
    """处理不同长度序列的自定义collate函数"""
    # 获取batch中的最大序列长度
    max_dino_len = max([item['dino'].shape[0] for item in batch])
    max_flow_len = max([item['flow'].shape[0] for item in batch])
    
    # 准备batch列表
    dino_features = []
    flow_features = []
    labels = []
    video_ids = []
    
    # 填充序列到相同长度
    for item in batch:
        # 处理DINO特征
        dino_feat = item['dino']
        dino_pad_len = max_dino_len - dino_feat.shape[0]
        if dino_pad_len > 0:
            dino_feat = torch.cat([
                dino_feat,
                torch.zeros(dino_pad_len, dino_feat.shape[1], dtype=dino_feat.dtype)
            ], dim=0)
        
        # 处理Flow特征
        flow_feat = item['flow']
        flow_pad_len = max_flow_len - flow_feat.shape[0]
        if flow_pad_len > 0:
            flow_feat = torch.cat([
                flow_feat,
                torch.zeros(flow_pad_len, flow_feat.shape[1], dtype=flow_feat.dtype)
            ], dim=0)
        
        dino_features.append(dino_feat.unsqueeze(0))
        flow_features.append(flow_feat.unsqueeze(0))
        labels.append(item['label'])
        video_ids.append(item['video_id'])
    
    # 堆叠所有张量
    dino_features = torch.cat(dino_features, dim=0)
    flow_features = torch.cat(flow_features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return {
        'dino': dino_features,
        'flow': flow_features,
        'label': labels,
        'video_id': video_ids
    }

class VideoFeatureDataset(Dataset):
    def __init__(self, root_path, encoder_params, phase='train', val_split=0.2, seed=42):
        self.root_path = root_path
        self.encoder_params = encoder_params
        self.phase = phase
        
        # 加载特征文件
        rgb_file = os.path.join(root_path, f'{encoder_params}_RepsAndLabels.h5')
        flow_file = os.path.join(root_path, f'{encoder_params}_Flow_RepsAndLabels.h5')
        
        print(f"\nLoading features for {phase} phase from:")
        print(f"DINO features: {rgb_file}")
        print(f"Flow features: {flow_file}")
        
        self.hf_rgb = h5py.File(rgb_file, 'r')
        self.hf_flow = h5py.File(flow_file, 'r')
        
        # 获取所有视频ID并确定标签
        self.video_ids = list(self.hf_rgb.keys())
        self.labels = [1 if "expert" in vid.lower() else 0 for vid in self.video_ids]
        
        # 使用stratified split保持类别平衡
        train_ids, val_ids = train_test_split(
            range(len(self.video_ids)), 
            test_size=val_split,
            stratify=self.labels,
            random_state=seed
        )
        
        # 根据phase选择对应的索引
        self.indices = train_ids if phase == 'train' else val_ids
        
        # 打印数据集统计信息
        phase_labels = [self.labels[i] for i in self.indices]
        print(f"\n{phase.capitalize()} set statistics:")
        print(f"Total videos: {len(self.indices)}")
        print(f"Expert videos: {sum(phase_labels)}")
        print(f"Novice videos: {len(phase_labels) - sum(phase_labels)}")
    
    def __len__(self):
        return len(self.indices)
    
    def _augment_features(self, features, p=0.5):
        """简单的特征增强"""
        if self.phase != 'train' or np.random.random() > p:
            return features
            
        # 随机mask一些时间步
        mask = torch.rand(features.shape[0]) > 0.1
        features = features[mask]
        
        # 添加高斯噪声
        if np.random.random() < 0.5:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
            
        return features
    
    def __getitem__(self, idx):
        # 获取真实的视频索引
        video_idx = self.indices[idx]
        video_id = self.video_ids[video_idx]
        
        # 加载特征
        dino_features = self.hf_rgb[video_id][:]  # shape: (N, 384)
        flow_features = self.hf_flow[video_id][:]  # shape: (N-1, 10)
        
        # 转换为tensor并应用数据增强
        dino_features = torch.FloatTensor(dino_features)
        flow_features = torch.FloatTensor(flow_features)
        
        if self.phase == 'train':
            dino_features = self._augment_features(dino_features)
            flow_features = self._augment_features(flow_features)
        
        label = torch.LongTensor([self.labels[video_idx]])
        
        return {
            'dino': dino_features,
            'flow': flow_features,
            'label': label,
            'video_id': video_id
        }

def create_dataloaders(root_path, encoder_params, batch_size=4, num_workers=2, val_split=0.2, seed=42):
    """创建训练和验证数据加载器"""
    datasets = {
        phase: VideoFeatureDataset(
            root_path=root_path,
            encoder_params=encoder_params,
            phase=phase,
            val_split=val_split,
            seed=seed
        )
        for phase in ['train', 'val']
    }
    
    dataloaders = {
        phase: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        for phase, dataset in datasets.items()
    }
    
    return dataloaders
