# SAIS/scripts/run_experiments.py
import os
import subprocess

def run_pipeline():
    """运行完整的实验流程"""
    # 1. 创建模拟数据集
    print("Creating mock dataset...")
    subprocess.run(["python", "SAIS/scripts/create_mock_dataset.py"])
    
    # 2. 提取特征
    print("Extracting features...")
    subprocess.run(["python", "SAIS/scripts/extract_representations.py"])
    
    # 3. 训练模型
    print("Training model...")
    subprocess.run(["python", "SAIS/scripts/train.py"])

if __name__ == '__main__':
    run_pipeline()