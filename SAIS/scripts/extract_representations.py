# SAIS/scripts/extract_representations.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_dino_model():
    """加载预训练的DINO ViT模型"""
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    return model  # 移除.cuda()

def extract_features(model, image_path, transform):
    """提取单张图像的特征"""
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # 移除.cuda()
    
    with torch.no_grad():
        features = model(img)
    
    return features.cpu().numpy()

def process_video(model, video_path, transform):
    """处理单个视频的所有帧"""
    frame_paths = sorted(Path(video_path).glob('frame_*.jpg'))
    features = []
    
    for frame_path in tqdm(frame_paths, desc=f"Processing {video_path.name}"):
        feat = extract_features(model, frame_path, transform)
        features.append(feat)
    
    return np.concatenate(features)

def main():
    # 设置设备
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 设置转换
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 加载模型
    print("Loading DINO model...")
    model = load_dino_model()
    model = model.to(device)
    
    # 设置路径
    root_path = 'mock_VUA'
    videos_path = Path(root_path) / 'Images'
    output_path = Path(root_path) / 'Results'
    output_path.mkdir(exist_ok=True)
    
    print("Starting feature extraction...")
    # 创建H5文件
    with h5py.File(output_path / 'ViT_SelfSupervised_ImageNet_RepsAndLabels.h5', 'w') as f:
        # 处理所有视频
        video_dirs = list(videos_path.iterdir())
        for video_dir in tqdm(video_dirs, desc="Processing videos"):
            if video_dir.is_dir():
                print(f"\nProcessing video: {video_dir.name}")
                features = process_video(model, video_dir, transform)
                f.create_dataset(video_dir.name, data=features)
                print(f"Features shape: {features.shape}")

if __name__ == '__main__':
    main()