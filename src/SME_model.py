import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SMEModel(nn.Module):
    def __init__(self, scale, min_label,
                 simple_linear_scale=False, input_size=4608, dim_reduction_len=1, reduced_size=128, hidden_size=32, use_gru=True):
        super(SMEModel, self).__init__()
        self.hidden_size = hidden_size
        self.use_gru = use_gru
        dimemsion_reduction = list()
        curr_reduced_size = reduced_size * np.power(2, dim_reduction_len-1)
        for _ in range(dim_reduction_len):
            if curr_reduced_size < input_size:
                dimemsion_reduction.append(nn.Linear(input_size, curr_reduced_size))
                input_size = curr_reduced_size
            curr_reduced_size = int(curr_reduced_size / 2)
        self.dimemsion_reduction = nn.Sequential(*dimemsion_reduction)
        if self.use_gru:
            self.feature_aggregation = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.regression = nn.Linear(hidden_size, 1)
        self.bound = nn.Sigmoid()
        self.nlm = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1))  # 4 parameters
        self.lm = nn.Linear(1, 1)

        torch.nn.init.constant_(self.nlm[0].weight, 2 * np.sqrt(3))
        torch.nn.init.constant_(self.nlm[0].bias, -np.sqrt(3))
        torch.nn.init.constant_(self.nlm[2].weight, 1)
        torch.nn.init.constant_(self.nlm[2].bias, 0)
        for p in self.nlm[2].parameters():
            p.requires_grad = False
        torch.nn.init.constant_(self.lm.weight, scale)
        torch.nn.init.constant_(self.lm.bias, min_label)

        if simple_linear_scale:
            for p in self.lm.parameters():
                p.requires_grad = False

    def forward(self, input):
        x, x_len = input
        x = self.dimemsion_reduction(x)
        if self.use_gru:
            x, _ = self.feature_aggregation(x, self._get_initial_state(x.size(0), x.device))
        q = self.regression(x)
        relative_score = torch.zeros_like(q[:, 0])
        mapped_score = torch.zeros_like(q[:, 0])
        aligned_score = torch.zeros_like(q[:, 0])
        for i in range(q.shape[0]):
            relative_score[i] = self._sitp(q[i, :x_len[i]])
        relative_score = self.bound(relative_score)
        mapped_score = self.nlm(relative_score)
        for i in range(q.shape[0]):
            aligned_score[i] = self.lm(mapped_score[i])

        return relative_score, mapped_score, aligned_score

    def _sitp(self, q, tau=12, beta=0.5):
        """subjectively-inspired temporal pooling"""
        q = torch.unsqueeze(torch.t(q), 0)
        qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)
        qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)
        l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
        m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
        n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
        m = m / n
        q_hat = beta * m + (1 - beta) * l
        return torch.mean(q_hat)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0
