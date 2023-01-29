import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# dataset
cfg.train_file = 'input/birds.csv'
cfg.img_size = 368
cfg.batch_size = 16

cfg.transforms = A.Compose([
    A.Resize(cfg.img_size, cfg.img_size),
    A.Normalize(),
    ToTensorV2(p=1.0)
])
cfg.epoch = 20

# model
cfg.model_name = 'tf_efficientnet_b4'
cfg.n_classes = 450
cfg.lr = 0.00005


#
cfg.save_path = "output/baseline.pth"
