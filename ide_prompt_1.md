# 参考文献@website https://www.nature.com/articles/s41551-023-01010-8，行手术视频的研究
## 运用模拟数据，构建一个VUA数据集，使用DINO进行自监督学习，进行特征提取，构建一个使用模拟数据的示例流程。并分步骤实现：
## 首先创建模拟的手术视频数据集，代码为 @ SAIS/scripts/create_mock_dataset.py

## 使用DINO进行特征提取，运行代码 @ SAIS/scripts/extract_representations.py，生成特征提取的模型文件@ SAIS/mock_VUA/Results/ViT_SelfSupervised_ImageNet_RepsAndLabels.h5以及mock_VUA/文件夹

## 参考@ SAIS/scripts/extract_representations_0.py中的代码，请修改@ SAIS/scripts/extract_representations.py中的代码，使能够从模拟数据@ SAIS\scripts\create_mock_dataset.py中同时生成光流特征并保存为@ SAIS/mock_VUA/Results/ViT_SelfSupervised_ImageNet_FlowRepsAndLabels.h5文件


