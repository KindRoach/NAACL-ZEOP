from typing import Callable

import torch


class OrderedConLoss(torch.nn.Module):

    def __init__(self, margin: float, dist_func: Callable):
        super().__init__()
        self.margin = margin
        self.dist_func = dist_func

    def cal_loss_2(self, emb_self, emb_dropout):
        """
        Args:
            emb_self: hidden vector of shape [bsz, emb size].
            emb_dropout: hidden vector of shape [bsz, emb size].
        Returns:
            Loss in shape [batch size,].
        """

        return self.dist_func(emb_self, emb_dropout)

    def cal_loss_3(self, emb_self, emb_dropout, emb_pos):
        """
        Args:
            emb_self: hidden vector of shape [batch size, emb size].
            emb_dropout: hidden vector of shape [batch size, emb size].
            emb_pos: hidden vector of shape [batch size, emb size].
        Returns:
            Loss in shape [batch size,].
        """

        dist_dropout = self.dist_func(emb_self, emb_dropout)
        dist_pos = self.dist_func(emb_self, emb_pos)
        diff_pos = self.margin - (dist_pos - dist_dropout)
        loss_pos = torch.max(torch.zeros_like(diff_pos), diff_pos)
        return dist_dropout + loss_pos

    def cal_loss_4(self, emb_self, emb_dropout, emb_pos, emb_neu):
        """
        Args:
            emb_self: hidden vector of shape [batch size, emb size].
            emb_dropout: hidden vector of shape [batch size, emb size].
            emb_pos: hidden vector of shape [batch size, emb size].
            emb_neu: hidden vector of shape [batch size, emb size].
        Returns:
            Loss in shape [batch size,].
        """

        dist_dropout = self.dist_func(emb_self, emb_dropout)

        dist_pos = self.dist_func(emb_self, emb_pos)
        diff_pos = self.margin - (dist_pos - dist_dropout)
        loss_pos = torch.max(torch.zeros_like(diff_pos), diff_pos)

        dist_neu = self.dist_func(emb_self, emb_neu)
        diff_neu = self.margin - (dist_neu - dist_pos)
        loss_neu = torch.max(torch.zeros_like(diff_neu), diff_neu)

        return dist_dropout + loss_pos + loss_neu

    def cal_loss_5(self, emb_self, emb_dropout, emb_pos, emb_neu, emb_neg, is_seen):
        """
        Args:
            emb_self: hidden vector of shape [batch size, emb size].
            emb_dropout: hidden vector of shape [batch size, emb size].
            emb_pos: hidden vector of shape [batch size, emb size].
            emb_neu: hidden vector of shape [batch size, emb size].
            emb_neg: hidden vector of shape [batch size, emb size].
            emb_neg: hidden vector of shape [batch size,].
        Returns:
            Loss in shape [batch size,].
        """

        dist_dropout = self.dist_func(emb_self, emb_dropout)

        dist_pos = self.dist_func(emb_self, emb_pos)
        diff_pos = self.margin - (dist_pos - dist_dropout)
        loss_pos = torch.max(torch.zeros_like(diff_pos), diff_pos)

        dist_neu = self.dist_func(emb_self, emb_neu)
        diff_neu = self.margin - (dist_neu - dist_pos)
        loss_neu = torch.max(torch.zeros_like(diff_neu), diff_neu)

        dist_neg = self.dist_func(emb_self, emb_neg)
        diff_neg_seen = self.margin - (dist_neg - dist_neu)
        loss_neg_seen = torch.max(torch.zeros_like(diff_neg_seen), diff_neg_seen)
        loss_neg_seen *= is_seen.squeeze(1)

        diff_neg_unseen = self.margin - (dist_neg - dist_pos)
        loss_neg_unseen = torch.max(torch.zeros_like(diff_neg_unseen), diff_neg_unseen)
        loss_neg_unseen *= 1 - is_seen.squeeze(1)

        return dist_dropout + loss_pos + loss_neu + loss_neg_seen + loss_neg_unseen

    def cal_loss_pos_neg(self, emb_self, emb_pos, emb_neg):
        """
        Args:
            emb_self: hidden vector of shape [batch size, emb size].
            emb_pos: hidden vector of shape [batch size, emb size].
            emb_neg: hidden vector of shape [batch size, emb size].
        Returns:
            Loss in shape [batch size,].
        """

        dist_pos = self.dist_func(emb_self, emb_pos)
        dist_neg = self.dist_func(emb_self, emb_neg)
        diff = self.margin - dist_neg
        loss = torch.max(torch.zeros_like(diff), diff)
        return dist_pos + loss
