# Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization

This repository provides the official implementation of [**Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through Visual Prototype and Harmonization**](https://openaccess.thecvf.com/content/CVPR2025/papers/Mao_Beyond_Single-Modal_Boundary_Cross-Modal_Anomaly_Detection_through_Visual_Prototype_and_CVPR_2025_paper.pdf).

We study cross-modal anomaly detection, where a model is trained on known modalities and tested on unseen ones. To improve generalization, our method learns Transferable Visual Prototypes directly in the visual space and uses Prototype Harmonization to adaptively combine prototypes from different source modalities. For few-shot settings, Visual Discrepancy Inference further enhances detection by comparing query images with a few normal samples from the target modality.

Experiments on ten datasets across RGB, 3D, MRI/CT, and thermal modalities show that our method achieves strong zero-shot and few-shot anomaly detection performance.

<!-- <p align="center">
  <img src="./assets/framework.png" width="800">
</p>
<p align="center">
  Overview of the proposed cross-modal anomaly detection framework.
</p> -->


## Environment Setup
The code runs on a server equipped with 10 NVIDIA GeForce RTX 3090 GPUs under the following environment:
- Python 3.10.0
- PyTorch 2.5.1 + CUDA 11.8
- torchvision 0.20.1 + CUDA 11.8
- NumPy 1.26.3
- OpenCV 4.10.0
- scikit-learn 1.5.2
- scikit-image 0.24.0
- pandas 2.2.3
- matplotlib 3.10.0
- tqdm 4.67.1
- einops 0.8.0


## Data Preparation
### Download the datasets
We use ten public datasets across four modalities for cross-modal anomaly detection, including RGB datasets (MVTec AD, VisA, BTAD, and MPDD), 
3D datasets (MVTec 3D-AD and Eyecandies), MRI/CT datasets (Brain MRI and OCT2017), and thermal datasets (WTB and Leaves).
- RGB: [MvTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), [VisA](https://github.com/amazon-science/spot-diff), [BTAD](https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection/data), [MPDD](https://github.com/stepanje/MPDD)
- 3D: [MVTec 3D-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad), [Eyecandies](https://eyecan-ai.github.io/eyecandies/download)
- MRI/CT: [Brain MRI](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), [OCT2017](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- Thermal: [WTB](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), [Leaves](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

### Convert datasets to the MVTec-style structure
After downloading the datasets, reorganize them into the standard MVTec AD format. Each category should contain train, test, and ground_truth folders, 
where normal training images are placed in train/good, test images are placed in test/good or test/<defect_type>, and pixel-level masks are placed in ground_truth/<defect_type> if available.

```text
<dataset_name>/ 
└── <category_name>/
  ├── train/good/ 
  ├── test/good/ 
  ├── test/<defect_type>/ 
  └── ground_truth/<defect_type>/ 
  ...
```

For few-shot evaluation, the normal support samples for 1, 2, 4, and 8-shot settings are randomly generated and keep fixed across different methods for fair comparison, and can be downloaded from [few-shot data]()


## Run on Zero-Shot Setting

For zero-shot training and evaluation, simply run the corresponding bash script:

```text
  bash train_zero_shot.sh
```
```text
  bash test_zero_shot.sh
```

<p align="center">
  <img src="./assets/result_zero_shot.png" width="600">
</p>
<p align="center">
  Results under zero-shot setting.
</p>


## Run on Few-Shot Setting

For few-shot evaluation, simply run the corresponding bash script:
```text
  bash test_few_shot.sh
```

<p align="center">
  <img src="./assets/result_few_shot.png" width="600">
</p>
<p align="center">
  Results under few-shot setting.
</p>




