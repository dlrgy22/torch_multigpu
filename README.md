## π  Description

```bash
003.churn_modeling
βββ data
β   βββ input                  # input data
β       βββ ...     
β       βββ ...
β       βββ ...
β
βββ module
β   βββ data
β   β   βββ dataset.py
β   βββ model
β   β   βββ basemodel.py
β   β   βββ basemodel_multigpu.py
β   βββ Trainer
β       βββ basemodel.py
β       βββ basemodel_multigpu.py
β
βββ config                    # config file
β   βββ cfg_baseline           
β
βββ output                    # model output (pth)
β   βββ ...
β
βββ data_distribute.py  
βββ model_distribute.py  
```
### data
BIRDS 450 SPECIES- IMAGE CLASSIFICATION
https://www.kaggle.com/datasets/gpiosenka/100-bird-species

### config
- train_file : νμ΅μ μ¬μ©λ  μ΄λ―Έμ§μ μμΉκ° μ μ₯λμ΄μλ csvνμΌ
- img_size : νμ΅μ μ¬μ©λ  μ΄λ―Έμ§ μ¬μ΄μ¦
- batch_size : batch size
- epoch : train epoch
- transforms : image transforms
- model_name : timmμ μ μ₯λμ΄μλ μ΄λ―Έμ§ λͺ¨λΈ path
- n_classes : class κ°μ
- save_path : λͺ¨λΈ νλΌλ―Έν° μ μ₯ μμΉ

### how to use
single gpu
```bash
python data_distribute.py --config cfg_baseline
```
data distribute (ddp)
```bash
python -m torch.distributed.launch --nproc_per_node=4 data_distribute.py --config cfg_baseline
```
model distribute (ddp)
```bash
python model_distribute.py --config cfg_baseline
```
