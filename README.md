# vector_map



## 1.环境
```
conda create --name vector_map python=3.9
conda activate vector_map
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia

```

## 2.数据集
MapDr
```
git lfs install
cd dataset
git clone https://www.modelscope.cn/datasets/MIV-XJTU/MapDR-mini.git
```
OpenLaneV2
```
gdown https://drive.google.com/uc?id=1Ni-L6u1MGKJRAfUXm39PdBIxdk_ntdc6
```

