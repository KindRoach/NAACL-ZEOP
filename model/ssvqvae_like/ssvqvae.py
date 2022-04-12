from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data_process.data_enum import DataTypeEnum
from dataset.ssvqvae_dataset import SsvqvaeDataset
from model.base_model import BaseModel, BaseConfig, unit_test, DistFuncEnum
from validator.classify_validator import ClassifyValidator


@dataclass
class SsvqvaeConfig(BaseConfig):
    hidden_factors: int = 500
    loss_weight_supervised: float = 1
    loss_weight_vq: float = 0.1
    loss_weight_vae: float = 0.1
    vae_hidden: int = 1024

    # For SsvqvaeCLModel
    loss_weight_contrastive: float = 1.0
    contrastive_loss_margin: float = 1.0
    contrastive_dist_func: DistFuncEnum = DistFuncEnum.Euclidean

    # For SsvqvaeSupCLModel
    loss_weight_cluster: float = 1.0
    temperature: float = 0.5
    base_temperature: float = 0.07


class VAE(torch.nn.Module):
    def __init__(self, input_dim, vae_hidden, k_num):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, vae_hidden)
        self.decoder = torch.nn.Linear(vae_hidden + k_num, input_dim)

        # distribution parameters
        self.fc_mu = torch.nn.Linear(vae_hidden, vae_hidden)
        self.fc_var = torch.nn.Linear(vae_hidden, vae_hidden)

    def forward(self, v_t, y_t):  # L(t,y)
        x_encoded = self.encoder(v_t)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # SAMPLE Z from Q(Z|x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z_t = q.rsample()

        v_t_hat = self.decoder(torch.cat([z_t, y_t], dim=1))

        # compute the probability of x under this n-dimensional distribution
        log_scale = torch.nn.Parameter(torch.Tensor([0.0])).to(v_t.device)
        scale = torch.exp(log_scale)
        dist = torch.distributions.Normal(v_t_hat, scale)
        log_ptz = dist.log_prob(v_t_hat)

        # sum across types
        log_ptz = log_ptz.sum(dim=1)

        # Calculate KL
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z_t)
        log_pz = p.log_prob(z_t)

        # 3. calculate the KL
        kl = (log_qzx - log_pz).sum(dim=1)
        L_t_y = kl - log_ptz

        return L_t_y


class SsvqvaeModel(BaseModel):
    def __init__(self, config: SsvqvaeConfig):
        super().__init__(config)
        self.config = config
        self.hidden_layer = torch.nn.Linear(self.bert_model.embeddings.word_embeddings.embedding_dim, self.config.hidden_factors)
        self.predict_layer = torch.nn.Linear(self.config.hidden_factors, self.config.k_num)
        self.vae = VAE(self.bert_model.embeddings.word_embeddings.embedding_dim, config.vae_hidden, config.k_num)

    def forward(self, self_event, is_seen, self_label):
        bert_emb = self.encode(self_event)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        cs_loss = self.config.loss_weight_supervised * self.cs_loss(soft, self_label, is_seen)
        vq_loss = self.config.loss_weight_vq * self.vq_loss(hidden, soft)
        vae_loss = self.config.loss_weight_vae * self.vae_loss(bert_emb, soft, is_seen)
        loss = cs_loss + vq_loss + vae_loss
        return loss

    def classify(self, event_encoding):
        bert_emb = self.encode(event_encoding)
        hidden = torch.sigmoid(self.hidden_layer(bert_emb))
        soft = torch.softmax(self.predict_layer(hidden), dim=-1)
        return soft

    def cs_loss(self, soft, label, is_seen):
        is_seen = is_seen.squeeze(1)
        label = label.squeeze(1)

        target = torch.zeros_like(soft)
        for i in range(label.shape[0]):
            target[i, label[i]] = 1
        supervise_loss = is_seen * torch.sum(-torch.log(soft) * target, dim=-1)
        seen_score = torch.max(soft[:, :self.config.seen_type_num], dim=-1).values
        unseen_score = torch.max(soft[:, self.config.seen_type_num:], dim=-1).values
        unsupervised_loss = (1 - is_seen) * (seen_score - unseen_score)
        loss = supervise_loss + unsupervised_loss

        return loss

    def vq_loss(self, hidden, soft):
        predict_label = torch.argmax(soft, dim=1)
        e = self.predict_layer.weight[predict_label, :]
        vq_a = torch.norm(hidden.detach() - e, p=2, dim=1)
        vq_b = torch.norm(hidden - e.detach(), p=2, dim=1)
        return vq_a + vq_b

    def vae_loss(self, bert_emb, soft, is_seen):
        is_seen = is_seen.squeeze(1)
        seen_loss = self.vae(bert_emb, soft)
        unseen_loss = torch.zeros_like(seen_loss)
        for yi in range(self.config.k_num):
            q_y_t = soft[:, yi]
            loss_i = q_y_t * (torch.log(q_y_t) + seen_loss)
            unseen_loss += loss_i
        loss = is_seen * seen_loss + (1 - is_seen) * unseen_loss
        return loss

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return SsvqvaeDataset(self.config.data_set, data_type, self.config.bert_model)

    def get_validator(self, datatype: DataTypeEnum):
        return ClassifyValidator(self, datatype)


def main():
    config = SsvqvaeConfig(main_device="cuda:0", train_batch_size=16)
    model = SsvqvaeModel(config)
    unit_test(model)


if __name__ == '__main__':
    main()
