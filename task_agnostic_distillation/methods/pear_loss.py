import torch.nn as nn
import torch


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(-1).unsqueeze(-1),
                             b - b.mean(-1).unsqueeze(-1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()



class Dist_att(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0):
        super(Dist_att, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = z_s.softmax(dim=-1)
        y_t = z_t.softmax(dim=-1)
        inter_token_1 = inter_class_relation(y_s, y_t)
        inter_token_2 = inter_class_relation(y_s.transpose(2, 3), y_t.transpose(2, 3))
        inter_head = inter_class_relation(y_s.transpose(1, 3), y_t.transpose(1, 3))
        inter_sentence = inter_class_relation(y_s.transpose(0, 3), y_t.transpose(0, 3))
        return inter_token_1, inter_token_2, inter_head, inter_sentence