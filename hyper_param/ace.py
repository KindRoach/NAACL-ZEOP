import dataclasses

from model.base_model import DistFuncEnum
from model.bertcl_like.bert_cl import BertCLConfig
from model.sccl_like.sccl import ScclConfig
from model.ssvqvae_like.ssvqvae import SsvqvaeConfig
from model.zecl_like.zecl import ZeclConfig
from model.zope_like.zope import ZopeConfig

epoch_num = 500

zecl_config = ZeclConfig(
    learning_rate=1e-3,
    bert_learning_rate=1e-6,

    loss_weight_supervised=1.0,
    contrastive_loss_margin=1.0,
    contrastive_dist_func=DistFuncEnum.Euclidean,

    cluster_alpha=1.0,
    loss_weight_cluster=1.0,
)

zecl_pcl_config = dataclasses.replace(zecl_config)
zecl_pcl_config.contrastive_dist_func = DistFuncEnum.Wasserstein

zope_config = ZopeConfig(
    learning_rate=1e-3,
    bert_learning_rate=1e-6,

    loss_weight_el=1.0,
    loss_weight_tl=1.0,
    contrastive_loss_margin=1.0,
    contrastive_dist_func=DistFuncEnum.Euclidean,

    cluster_alpha=1.0,
    loss_weight_cluster=1.0,
)

zope_pcl_config = dataclasses.replace(zope_config)
zope_pcl_config.contrastive_dist_func = DistFuncEnum.Wasserstein

bertCL_config = BertCLConfig(
    learning_rate=1e-3,
    bert_learning_rate=1e-6,

    contrastive_loss_margin=1.0,
    contrastive_dist_func=DistFuncEnum.Euclidean,

    temperature=0.5,
    base_temperature=0.07,
)

bert_supcl_config = dataclasses.replace(bertCL_config)
bert_supcl_config.learning_rate = 1e-5

ssvqvae_config = SsvqvaeConfig(
    learning_rate=1e-3,
    bert_learning_rate=1e-6,

    hidden_factors=500,
    loss_weight_supervised=1.0,
    loss_weight_vq=0.1,
    loss_weight_vae=0.002,
    vae_hidden=1024,

    loss_weight_contrastive=1.0,
    contrastive_loss_margin=1.0,
    contrastive_dist_func=DistFuncEnum.Euclidean,

    loss_weight_cluster=1.0,
    temperature=0.5,
    base_temperature=0.07,
)

ssvqvae_supcl_config = dataclasses.replace(ssvqvae_config)
ssvqvae_supcl_config.learning_rate = 1e-5

ssvqvae_pcl_config = dataclasses.replace(ssvqvae_config)
ssvqvae_pcl_config.contrastive_dist_func = DistFuncEnum.Wasserstein

sccl_config = ScclConfig(
    learning_rate=1e-5,
    bert_learning_rate=1e-6,

    head_size=128,
    cluster_alpha=1.0,
    temperature=0.5,
    base_temperature=0.07,
    loss_weight_cluster=10,

    contrastive_loss_margin=1.0,
    contrastive_dist_func=DistFuncEnum.Euclidean
)

sccl_orderedcl_config = dataclasses.replace(sccl_config)
sccl_orderedcl_config.learning_rate = 1e-3

sccl_pcl_config = dataclasses.replace(sccl_orderedcl_config)
sccl_pcl_config.contrastive_dist_func = DistFuncEnum.Wasserstein

