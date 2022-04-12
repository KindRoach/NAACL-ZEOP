from dataclasses import dataclass

from model.base_model import BaseModel, BaseConfig


@dataclass
class PureBertConfig(BaseConfig):
    pass


class PureBertModel(BaseModel):
    """
    Not a trainable model. Only for baseline.
    """

    def __init__(self, config: PureBertConfig):
        super().__init__(config)
