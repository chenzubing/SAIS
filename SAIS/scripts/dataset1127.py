import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import numpy as np

class VideoFeatureDataset(Dataset):
    def __init__(self, root_path, encoder_params, phase='train'):
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
        
        # 获取所有视频ID
        self.video_ids = list(self.hf_rgb.keys())
        
        # TODO: 根据phase划分训练集和验证集
        if phase == 'train':
            self.video_ids = self.video_ids[:int(len(self.video_ids) * 0.8)]
        else:
            self.video_ids = self.video_ids[int(len(self.video_ids) * 0.8):]
            
        print(f"Number of {phase} videos: {len(self.video_ids)}")
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # 加载特征
        dino_features = self.hf_rgb[video_id][:]  # shape: (N, 384)
        flow_features = self.hf_flow[video_id][:]  # shape: (N-1, 10)
        
        # TODO: 添加标签
        label = 1 if "expert" in video_id.lower() else 0
        
        # 转换为tensor
        dino_features = torch.FloatTensor(dino_features)
        flow_features = torch.FloatTensor(flow_features)
        label = torch.LongTensor([label])
        
        return {
            'dino': dino_features,
            'flow': flow_features,
            'label': label,
            'video_id': video_id
        }

def create_dataloaders(root_path, encoder_params, batch_size=4):
    """创建训练和验证数据加载器"""
    datasets = {
        phase: VideoFeatureDataset(root_path, encoder_params, phase)
        for phase in ['train', 'val']
    }
    
    dataloaders = {
        phase: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=2
        )
        for phase, dataset in datasets.items()
    }
    
    return dataloaders
