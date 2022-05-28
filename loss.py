import torch.nn.functional as F
import torch

def negative_crossentropy(f, labels):
    out = 1 - F.softmax(f, 1)
    out = F.softmax(out, 1)
    labels = labels.long()
    return F.nll_loss(out.log(), labels.long())  # Equation(8) in paper

def entropy_loss(p):
    p = F.softmax(p, dim=1)
    epsilon = 1e-5
    return -1 * torch.sum(p * torch.log(p + epsilon)) / p.size(0)