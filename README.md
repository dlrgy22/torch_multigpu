## 🏠 Description

```bash
003.churn_modeling
├── data
│   └── input                  # input data
│       ├── ...     
│       ├── ...
│       └── ...
│
├── module
│   ├── data
│   │   └── dataset.py
│   ├── model
│   │   ├── basemodel.py
│   │   └── basemodel_multigpu.py
│   └── Trainer
│       ├── basemodel.py
│       └── basemodel_multigpu.py
│
├── config                    # config file
│   └── cfg_baseline           
│
├── output                    # model output (pth)
│   └── ...
│
├── data_distribute.py  
└── model_distribute.py  
```
### data
BIRDS 450 SPECIES- IMAGE CLASSIFICATION
https://www.kaggle.com/datasets/gpiosenka/100-bird-species

### config
- train_file : 학습에 사용될 이미지의 위치가 저장되어있는 csv파일
- img_size : 학습에 사용될 이미지 사이즈
- batch_size : batch size
- epoch : train epoch
- transforms : image transforms
- model_name : timm에 저장되어있는 이미지 모델 path
- n_classes : class 개수
- save_path : 모델 파라미터 저장 위치

### how to use
single gpu
```bash
python data_distribute.py --config cfg_baseline
```
data distribute (ddp)
```bash
python -m torch.distributed.launch --nproc_per_node=4 train_script.py --config cfg_baseline
```
model distribute (ddp)
```bash
python model_distribute.py --config cfg_baseline
```