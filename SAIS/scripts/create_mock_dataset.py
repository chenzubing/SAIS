# SAIS/scripts/create_mock_dataset.py
import cv2
import numpy as np
import os
from tqdm import tqdm

def create_mock_surgical_video(output_path, num_frames=300, size=(480, 640)):
    """创建模拟的手术视频帧"""
    os.makedirs(output_path, exist_ok=True)
    
    # 创建模拟的手术场景
    for i in tqdm(range(num_frames)):
        # 创建基础图像
        frame = np.ones((size[0], size[1], 3), dtype=np.uint8) * 200
        
        # 添加模拟的手术工具
        cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)  # 模拟针头
        cv2.line(frame, (320, 240), (320+50, 240+50), (255, 0, 0), 2)  # 模拟缝线
        
        # 添加随机运动
        offset_x = int(np.sin(i/10) * 20)
        offset_y = int(np.cos(i/10) * 20)
        
        # 保存帧
        frame_path = os.path.join(output_path, f'frame_{i:08d}.jpg')
        cv2.imwrite(frame_path, frame)

# SAIS/scripts/create_mock_dataset.py
def create_mock_dataset():
    """创建模拟的VUA数据集"""
    root_path = 'mock_VUA'
    
    # 创建所有必需的目录
    for directory in ['Images', 'SurgicalPaths', 'Results']:
        os.makedirs(os.path.join(root_path, directory), exist_ok=True)
    
    # 创建视频数据
    for split in ['train', 'val', 'test']:
        for skill in ['high', 'low']:
            for video_id in range(5):
                video_path = os.path.join(root_path, 'Images', 
                                        f'video_{split}_{skill}_{video_id}')
                create_mock_surgical_video(video_path)
    
    # 创建标注文件
    create_mock_annotations(root_path)

def create_mock_annotations(root_path):
    """创建模拟的标注文件"""
    import pandas as pd
    
    annotations = []
    for split in ['train', 'val', 'test']:
        for skill in ['high', 'low']:
            for video_id in range(5):
                video_name = f'video_{split}_{skill}_{video_id}'
                
                # 添加手术阶段标注
                annotations.append({
                    'Video': video_name,
                    'RACE': 'Needle Handling',
                    'StartFrame': 0,
                    'EndFrame': 100,
                    'maj': 2 if skill == 'high' else 0,
                    'EASE': 'Wrist Rotation'
                })
                
    df = pd.DataFrame(annotations)
    df.to_csv(os.path.join(root_path, 'SurgicalPaths', 'EASE_Explanations.csv'))

if __name__ == '__main__':
    create_mock_dataset()