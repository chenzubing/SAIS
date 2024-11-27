# SAIS/scripts/extract_representations.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import os

def load_dino_model():
    """加载预训练的DINO ViT模型"""
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    return model

def load_flow_model():
    """加载预训练的光流模型"""
    try:
        print("\n=== Flow Model Loading Process ===")
        import ptlflow
        print("Successfully imported ptlflow")
        
        # 手动下载模型权重 - 使用新的下载链接
        weights_url = 'https://github.com/hmorimitsu/ptlflow/releases/download/weights/raft-things-46.24.ckpt'
        weights_path = '/root/SAIS/mock_VUA/Results/models/raft-things.ckpt'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print(f"Created models directory at: {os.path.dirname(weights_path)}")
        
        # 如果权重文件不存在，则下载
        if not os.path.exists(weights_path):
            print(f"Downloading RAFT weights to {weights_path}...")
            try:
                # 使用requests库下载，它提供更好的错误处理
                import requests
                response = requests.get(weights_url, stream=True)
                response.raise_for_status()  # 检查下载是否成功
                
                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))
                print(f"Total file size to download: {total_size / 1024 / 1024:.2f} MB")
                
                # 写入文件
                with open(weights_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print("Download completed successfully")
                
                # 验证文件大小
                file_size = os.path.getsize(weights_path)
                print(f"Downloaded file size: {file_size / 1024 / 1024:.2f} MB")
                
                if file_size < 1000000:  # 文件小于1MB可能表示下载不完整
                    raise Exception("Downloaded file is too small, might be corrupted")
                    
            except Exception as e:
                print(f"Error during download: {str(e)}")
                if os.path.exists(weights_path):
                    os.remove(weights_path)  # 删除可能损坏的文件
                raise
        else:
            file_size = os.path.getsize(weights_path)
            print(f"Found existing RAFT weights. File size: {file_size / 1024 / 1024:.2f} MB")
        
        # 加载模型
        print("Creating RAFT model...")
        model = ptlflow.get_model('raft', pretrained_ckpt=weights_path)
        print("Model created successfully")
        
        model.eval()
        print("Model set to eval mode")
        
        # 验证模型
        print("\nValidating model parameters...")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {param_count:,}")
        
        # 尝试进行一次简单的前向传播来验证模型
        print("\nValidating model with dummy input...")
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = model.forward_features(dummy_input)
            print("Model forward pass successful")
        except Exception as e:
            print(f"Warning: Model forward pass failed: {str(e)}")
        
        print("=== Flow Model Loading Complete ===\n")
        return model
        
    except Exception as e:
        print("\n=== Flow Model Loading Failed ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        print("=== End of Error Details ===\n")
        print("Proceeding with DINO features only...")
        return None

def extract_features(model, image_path, transform, device):
    """提取单张图像的特征"""
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    img = img.to(device)  # Move input tensor to the same device as model
    
    with torch.no_grad():
        features = model(img)
    
    return features.cpu().numpy()

def process_flow_inputs(model, path1, path2):
    """处理两帧图像以准备光流计算"""
    try:
        images = []
        for path in [path1, path2]:
            img = cv.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            images.append(img)
        
        io_adapter = IOAdapter(model, images[0].shape[:2], cuda=torch.cuda.is_available())
        inputs = io_adapter.prepare_inputs(images)
        return inputs, io_adapter
    except Exception as e:
        print(f"Warning: Failed to process flow inputs: {str(e)}")
        return None, None

def obtain_flow(model, inputs, io_adapter):
    """计算光流特征"""
    try:
        if inputs is None or io_adapter is None:
            return np.zeros(2048)  # 返回零向量作为替代
            
        with torch.no_grad():
            predictions = model(inputs)
        predictions = io_adapter.unpad_and_unscale(predictions)
        flows = predictions['flows']
        flow_features = flows[0, 0].reshape(-1)  # 将光流特征展平
        
        # 确保特征维度正确
        if flow_features.shape[0] != 2048:
            print(f"Warning: Unexpected flow feature dimension: {flow_features.shape[0]}, padding to 2048")
            padded_features = np.zeros(2048)
            padded_features[:min(flow_features.shape[0], 2048)] = flow_features[:min(flow_features.shape[0], 2048)].cpu().numpy()
            return padded_features
            
        return flow_features.cpu().numpy()
    except Exception as e:
        print(f"Warning: Failed to obtain flow features: {str(e)}")
        return np.zeros(2048)  # 返回零向量作为替代

def process_video(dino_model, flow_model, video_path, transform, device):
    """处理单个视频的所有帧，提取DINO和光流特征"""
    frame_paths = sorted(Path(video_path).glob('frame_*.jpg'))
    dino_features = []
    flow_features = []
    flow_success_count = 0
    flow_fail_count = 0
    
    print(f"\n=== Processing Video: {video_path.name} ===")
    print(f"Total frames: {len(frame_paths)}")
    
    for i in tqdm(range(len(frame_paths)-1), desc=f"Processing {video_path.name}"):
        # 提取DINO特征
        feat = extract_features(dino_model, frame_paths[i], transform, device)
        dino_features.append(feat)
        
        # 提取光流特征
        if flow_model is not None:
            try:
                inputs, io_adapter = process_flow_inputs(flow_model, frame_paths[i], frame_paths[i+1])
                if inputs is not None:
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    flow_feat = obtain_flow(flow_model, inputs, io_adapter)
                    flow_features.append(flow_feat)
                    flow_success_count += 1
                else:
                    print(f"\nWarning: Failed to process flow inputs for frames {i} and {i+1}")
                    flow_features.append(np.zeros(2048))
                    flow_fail_count += 1
            except Exception as e:
                print(f"\nWarning: Failed to process flow features for frame {i}: {str(e)}")
                flow_features.append(np.zeros(2048))
                flow_fail_count += 1
    
    # 为最后一帧添加DINO特征
    feat = extract_features(dino_model, frame_paths[-1], transform, device)
    dino_features.append(feat)
    
    print(f"\n=== Video Processing Statistics ===")
    print(f"Total frames processed: {len(frame_paths)}")
    print(f"DINO features extracted: {len(dino_features)}")
    if flow_model is not None:
        print(f"Flow features successful: {flow_success_count}")
        print(f"Flow features failed: {flow_fail_count}")
        print(f"Flow success rate: {flow_success_count/(flow_success_count+flow_fail_count)*100:.2f}%")
    
    # 如果没有成功提取任何flow特征，返回None
    if flow_success_count == 0:
        print("No successful flow features extracted, returning None for flow features")
        return np.concatenate(dino_features), None
        
    return np.concatenate(dino_features), np.stack(flow_features)

def main():
    # 设置设备
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        print(f"Found {torch.cuda.device_count()} CUDA device(s)")
        device = torch.device('cuda')
        # 打印CUDA信息
        for i in range(torch.cuda.device_count()):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i)/1024**2:.1f}MB")
            print(f"  Cached: {torch.cuda.memory_reserved(i)/1024**2:.1f}MB")
    
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
    dino_model = load_dino_model()
    dino_model = dino_model.to(device)
    print("DINO model moved to", next(dino_model.parameters()).device)
    
    print("Loading Flow model...")
    flow_model = load_flow_model()
    if flow_model is not None:
        flow_model = flow_model.to(device)
        print("Flow model moved to", next(flow_model.parameters()).device)
    
    # 设置路径 - 使用绝对路径
    root_path = '/root/SAIS/mock_VUA'  # 修改为云服务器上的实际路径
    videos_path = Path(root_path) / 'Images'
    output_path = Path(root_path) / 'Results'
    output_path.mkdir(exist_ok=True)
    
    if not videos_path.exists():
        raise FileNotFoundError(f"Videos directory not found at {videos_path}")
    
    print(f"Processing videos from: {videos_path}")
    print(f"Saving results to: {output_path}")
    
    # 创建H5文件
    with h5py.File(output_path / 'ViT_SelfSupervised_ImageNet_RepsAndLabels.h5', 'w') as f_dino:
        # 只在flow_model成功加载时创建flow特征文件
        f_flow = None
        if flow_model is not None:
            f_flow = h5py.File(output_path / 'ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5', 'w')
        
        try:
            # 处理所有视频
            video_dirs = sorted(list(videos_path.iterdir()))
            print(f"Found {len(video_dirs)} video directories")
            
            for video_dir in tqdm(video_dirs, desc="Processing videos"):
                if video_dir.is_dir():
                    print(f"\nProcessing video: {video_dir.name}")
                    frame_count = len(list(video_dir.glob('frame_*.jpg')))
                    print(f"Found {frame_count} frames")
                    
                    dino_features, flow_features = process_video(dino_model, flow_model, video_dir, transform, device)
                    f_dino.create_dataset(video_dir.name, data=dino_features)
                    if f_flow is not None and flow_features is not None:
                        f_flow.create_dataset(video_dir.name, data=flow_features)
                    print(f"DINO Features shape: {dino_features.shape}")
                    if flow_features is not None:
                        print(f"Flow Features shape: {flow_features.shape}")
                    
                    # Clear CUDA cache after each video
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise
        finally:
            if f_flow is not None:
                f_flow.close()
            # 最终清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()