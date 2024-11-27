import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2 as cv
import os
import gc
import sys
import traceback

def setup_gpu():
    """设置GPU和内存限制"""
    if torch.cuda.is_available():
        # 设置更严格的内存限制
        torch.cuda.set_per_process_memory_fraction(0.3)
        return torch.device('cuda')
    return torch.device('cpu')

def load_dino_model(device):
    """加载DINO模型"""
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model.eval()
    model.to(device)
    return model

def extract_features(model, image_path, transform, device, batch_size=2):
    """提取单张图像的DINO特征"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(img)
            
        return features.cpu().numpy()
    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None

def compute_flow(frame1, frame2):
    """使用OpenCV计算光流"""
    # 转换为灰度图
    prev = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    curr = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    
    # 计算光流
    flow = cv.calcOpticalFlowFarneback(
        prev, curr, 
        None,
        pyr_scale=0.5,  # 金字塔缩放比例
        levels=3,       # 金字塔层数
        winsize=15,     # 窗口大小
        iterations=3,   # 迭代次数
        poly_n=5,       # 多项式展开的邻域大小
        poly_sigma=1.2, # 高斯标准差
        flags=0
    )
    
    return flow

def extract_flow_features(frame1_path, frame2_path):
    """提取两帧之间的光流特征"""
    try:
        # 将Path对象转换为字符串
        frame1_path = str(frame1_path)
        frame2_path = str(frame2_path)
        
        # 读取图像
        frame1 = cv.imread(frame1_path)
        frame2 = cv.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            print(f"Error reading frames: {frame1_path} or {frame2_path}")
            return None
        
        # 调整图像大小以减少计算量
        target_size = (128, 128)
        frame1 = cv.resize(frame1, target_size)
        frame2 = cv.resize(frame2, target_size)
        
        # 计算光流
        flow = compute_flow(frame1, frame2)
        
        # 计算统计特征
        mean_flow = np.mean(flow, axis=(0, 1))  # [2]
        std_flow = np.std(flow, axis=(0, 1))    # [2]
        max_flow = np.max(np.abs(flow), axis=(0, 1))  # [2]
        
        # 计算方向直方图特征
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        bins = 4
        hist = np.histogram(ang, bins=bins, weights=mag)[0]
        hist = hist / (hist.sum() + 1e-6)  # 归一化
        
        # 合并所有特征
        flow_features = np.concatenate([
            mean_flow,  # 2维
            std_flow,   # 2维
            max_flow,   # 2维
            hist       # bins维
        ])
        
        return flow_features
        
    except Exception as e:
        print(f"Error extracting flow features: {str(e)}")
        traceback.print_exc()
        return None

def initialize_models(device):
    """初始化所有需要的模型"""
    # 只加载DINO模型
    dino_model = load_dino_model(device)
    return dino_model

def save_features_to_separate_h5(all_features, output_dir):
    """将特征分别保存到RepsAndLabels.h5和Flow_RepsAndLabels.h5"""
    try:
        # 保存DINO特征
        dino_file = output_dir / "RepsAndLabels.h5"
        with h5py.File(dino_file, 'w') as f:
            for video_name, features in all_features.items():
                if features['dino'] is not None and len(features['dino']) > 0:
                    dino_features = np.array(features['dino'])
                    # 重塑特征为(N, 384)
                    dino_features = dino_features.reshape(dino_features.shape[0], -1)
                    # 创建数据集，使用视频名作为键
                    f.create_dataset(video_name, data=dino_features)
                    print(f"Saved {video_name} DINO features with shape: {dino_features.shape}")
        
        # 保存Flow特征
        flow_file = output_dir / "Flow_RepsAndLabels.h5"
        with h5py.File(flow_file, 'w') as f:
            for video_name, features in all_features.items():
                if features['flow'] is not None and len(features['flow']) > 0:
                    flow_features = np.array(features['flow'])
                    # 创建数据集，使用视频名作为键
                    f.create_dataset(video_name, data=flow_features)
                    print(f"Saved {video_name} Flow features with shape: {flow_features.shape}")
                    
        print(f"\nFeatures saved to:")
        print(f"DINO features: {dino_file}")
        print(f"Flow features: {flow_file}")
                
    except Exception as e:
        print(f"Error saving features to h5: {str(e)}")
        traceback.print_exc()

def process_video(video_dir, device):
    """处理视频帧并提取特征"""
    try:
        # 初始化模型
        print(f"\nProcessing video: {video_dir.name}")
        dino_model = initialize_models(device)
        
        if dino_model is None:
            print("Failed to initialize DINO model. Exiting...")
            return None, None
            
        frame_paths = sorted(list(video_dir.glob('frame_*.jpg')))
        if len(frame_paths) < 2:
            print(f"Warning: Not enough frames in {video_dir}")
            return None, None
            
        # 初始化特征列表
        dino_features = []
        flow_features = []
        
        # 设置预处理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # 批处理帧
        batch_size = 32  # 可以根据GPU内存调整
        for i in tqdm(range(0, len(frame_paths), batch_size), desc="Processing frames"):
            try:
                batch_paths = frame_paths[i:i+batch_size]
                
                # 处理DINO特征
                for frame_path in batch_paths:
                    feat = extract_features(dino_model, frame_path, transform, device)
                    if feat is not None:
                        dino_features.append(feat)
                        
                # 处理Flow特征
                for j in range(len(batch_paths)):
                    if i+j >= len(frame_paths)-1:
                        break
                    flow = extract_flow_features(batch_paths[j], frame_paths[i+j+1])
                    if flow is not None:
                        flow_features.append(flow)
                
                # 清理批次内存
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                traceback.print_exc()
                continue
        
        if len(dino_features) > 0:
            dino_features = np.array(dino_features)
            print(f"DINO Features shape: {dino_features.shape}")
            
        if len(flow_features) > 0:
            flow_features = np.array(flow_features)
            print(f"Flow Features shape: {flow_features.shape}")
        
        return dino_features, flow_features
        
    except Exception as e:
        print(f"Error processing video {video_dir}: {str(e)}")
        traceback.print_exc()
        return None, None

def main():
    try:
        # 设置路径
        if len(sys.argv) > 2:
            videos_path = Path(sys.argv[1])
            output_path = Path(sys.argv[2])
        else:
            videos_path = Path("/root/SAIS/mock_VUA/Images")
            output_path = Path("/root/SAIS/mock_VUA/Results")
            
        print(f"Processing videos from: {videos_path}")
        print(f"Saving results to: {output_path}")
        
        # 确保输出目录存在
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        device = setup_gpu()
        
        # 处理视频
        video_dirs = sorted([d for d in videos_path.iterdir() if d.is_dir()])
        print(f"Found {len(video_dirs)} video directories")
        
        # 存储所有视频的特征
        all_features = {}
        
        # 使用tqdm显示总体进度
        for video_dir in tqdm(video_dirs, desc="Processing videos"):
            try:
                # 提取特征
                dino_features, flow_features = process_video(video_dir, device)
                
                # 存储特征
                if dino_features is not None or flow_features is not None:
                    all_features[video_dir.name] = {
                        'dino': dino_features,
                        'flow': flow_features
                    }
                    
            except Exception as e:
                print(f"Error processing video directory {video_dir}: {str(e)}")
                traceback.print_exc()
                continue
        
        # 保存特征到两个独立的h5文件
        save_features_to_separate_h5(all_features, output_path)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()