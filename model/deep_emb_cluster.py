import torch


class DeepEmbCluster(torch.nn.Module):

    def __init__(self, cluster_num: int, hidden_size: int, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.cluster_loss = torch.nn.KLDivLoss(reduction="none")
        self.cluster_centers = torch.nn.Parameter(torch.randn([cluster_num, hidden_size]))

    def cal_loss(self, cluster_p):
        weight = (cluster_p ** 2) / (torch.sum(cluster_p, 0) + 1e-9)
        target = (weight.t() / torch.sum(weight, 1)).t().detach()
        cluster_loss = self.cluster_loss((cluster_p + 1e-08).log(), target).sum(dim=1)
        return cluster_loss

    def cal_cluster_prob(self, embeddings):
        # embeddings = [batch_size, bert_hidden_size]
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = (self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)
