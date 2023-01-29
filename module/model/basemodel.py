import timm
from torch import nn

class BirdModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.n_classes)

    def forward(self, input):
        output = self.model(input)

        return output