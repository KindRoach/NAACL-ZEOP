# Reference: https://github.com/amazon-research/sccl


from typing import Dict

import torch

from model.base_model import unit_test, BaseModel
from model.sccl_like.sccl import ScclModel, ScclConfig


class ScclTriggerModel(ScclModel):
    def __init__(self, config: ScclConfig):
        super().__init__(config)
        self.config = config

    def encode(self, event_encoding: Dict[str, torch.Tensor]):
        return BaseModel.encode(self, event_encoding)


def main():
    config = ScclConfig()
    model = ScclTriggerModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
