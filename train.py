import random

import numpy
import torch

from data_process.data_enum import DatasetEnum
from data_process.data_reader.main_data_reader import get_seen_unseen_type_num
from hyper_param import ace, fewshot_ed
from model.base_model import BaseConfig, BaseModel
from model.bertcl_like.bert_cl import BertCLModel
from model.sccl_like.sccl_trigger import ScclTriggerModel
from model.ssvqvae_like.ssvq import SsvqModel
from model.zecl_like.zecl_pcl import ZeclPclModel
from model.zecl_like.zecl_wocl import ZeclWoclModel
from model.zope_like.zope_pcl import ZopePclModel
from model.zope_like.zope_pcl_homo import ZopePclHomoModel
from model.zope_like.zope_pcl_rewrite import ZopePclRewriteModel
from model.zope_like.zope_pcl_sim_ces import ZopePclSimcesModel
from tool.train_helper import TrainHelper

UNSEEN_TYPE_NUM = None
TRAIN_RANDOM_SEED = 2020

hyper_param_map = {
    DatasetEnum.Ace: ace,
    DatasetEnum.FewShotED: fewshot_ed,
}


def train_with_config(model_class: type, config: BaseConfig, data_set: DatasetEnum, train_tag: str = None):
    random.seed(TRAIN_RANDOM_SEED)
    numpy.random.seed(TRAIN_RANDOM_SEED)
    torch.manual_seed(TRAIN_RANDOM_SEED)
    torch.cuda.manual_seed(TRAIN_RANDOM_SEED)

    update_common_config(config, data_set)
    model: BaseModel = model_class(config)
    model.to(config.main_device)
    train_helper = TrainHelper(model, train_tag)
    train_helper.train_model()


def update_common_config(config: BaseConfig, data_set: DatasetEnum):
    config.epoch_num = hyper_param_map[data_set].epoch_num
    config.train_batch_size = 32
    config.main_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.multi_device_ids = [0, 1, 2, 3]
    config.eval_batch_size = config.train_batch_size * 4
    config.random_seed = TRAIN_RANDOM_SEED
    config.check_point_step = 500
    config.early_stop_check_point = 10
    config.data_set = data_set
    config.weight_decay = 1e-6
    config.bert_model = data_set.bert
    config.bert_fix_layers_ratio = 0

    # change dist function in contrastive loss
    # config.contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean

    seen_num, unseen_num = get_seen_unseen_type_num(data_set)
    config.unseen_type_num = UNSEEN_TYPE_NUM if UNSEEN_TYPE_NUM else unseen_num
    config.seen_type_num = seen_num


def main_train(model_class: type, data_set: DatasetEnum, train_tag: str = None):
    hp = hyper_param_map[data_set]

    default_config_map = {
        ScclTriggerModel: hp.sccl_config,
        SsvqModel: hp.ssvqvae_config,
        BertCLModel: hp.bertCL_config,
        ZeclPclModel: hp.zecl_pcl_config,
        ZopePclModel: hp.zope_pcl_config,
        ZeclWoclModel: hp.zecl_config,
        ZopePclSimcesModel: hp.zope_pcl_config,
        ZopePclRewriteModel: hp.zope_pcl_config,
        ZopePclHomoModel: hp.zope_pcl_config,
    }

    config = default_config_map[model_class]
    train_with_config(model_class, config, data_set, train_tag)


def main_exp():
    for model in [
        # Main model
        ScclTriggerModel,  # SCCL
        SsvqModel,  # SS-VQ-VAE
        BertCLModel,  # BERT-OCL
        ZeclPclModel,  # ZEO
        ZopePclModel,  # ZEOP

        # Ablation model
        ZeclWoclModel,  # ZEOP-woCL
        ZopePclSimcesModel,  # +Dropout
        ZopePclRewriteModel,  # +Rewrite
        ZopePclHomoModel,  # +Homogeneous
    ]:
        main_train(model, DatasetEnum.Ace)
        main_train(model, DatasetEnum.FewShotED)


def type_num_exp():
    for num in [
        80, 64, 48, 32
    ]:
        global UNSEEN_TYPE_NUM
        UNSEEN_TYPE_NUM = num
        main_train(ZopePclModel, DatasetEnum.Ace)


if __name__ == '__main__':
    main_exp()
