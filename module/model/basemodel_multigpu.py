import timm
from torch import nn

class BirdModelMultiGPU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.n_classes)
        self.gpus = cfg.device

    def to_gpus(self):
        self.model.to(self.gpus[0])
        for x in self.model.blocks[len(self.model.blocks) // 2 + 2: ]:
            x.to(self.gpus[1])

        self.model.conv_head.to(self.gpus[1])
        self.model.bn2.to(self.gpus[1])
        self.model.global_pool.to(self.gpus[1])
        self.model.classifier.to(self.gpus[1])

    def forward(self, input):
        conv_stem_output = self.model.conv_stem(input)
        layer_input = self.model.bn1(conv_stem_output)

        for i, block_layer in enumerate(self.model.blocks): 
            if i == len(self.model.blocks) // 2 + 2:
                layer_input = layer_input.to(self.gpus[1])

            layer_output = block_layer(layer_input)
            layer_input = layer_output

        output = self.model.conv_head(layer_output)
        output = self.model.bn2(output)
        output = self.model.global_pool(output)
        output = self.model.classifier(output)

        return output