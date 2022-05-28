import random
import torch.nn.functional as F
import numpy as np
import torch
from loss import negative_crossentropy, entropy_loss
from scipy import stats
from config import config

args = config()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h
def get_embedding(model, input, device, type=False):
    batch_size = 64
    if input.shape[0] > batch_size:
        embed = []
        i = 0
        while i <= input.shape[0]-1:
            embed.append(
                model(input[i:i+batch_size].to(device), return_feat=True).detach().cpu())
            i += batch_size
        embed = torch.cat(embed)
    else:
        embed = model(input.to(device), return_feat=True).detach().cpu()
    assert embed.shape[0] == input.shape[0]
    if type:return embed
    return embed.numpy()
def get_wrong(unlabel_targets, pseudo, index, type):
    gts = unlabel_targets[index]
    num = len(index)
    count = 0
    wrong_idx=[]
    wrong_list = []
    for idx, item in enumerate(pseudo):
        gt = gts[idx]
        if type=='nl':
            if gt==item:
                count+=1
                wrong_idx.append(idx)
                wrong_list.append(item)
        else:
            if gt!=item:
                count+=1
                wrong_idx.append(idx)
                wrong_list.append(item)
    wrong_rate = count/(num+1e-6)
    return wrong_rate, count, num, wrong_idx
def train(model, input, targets, epoch, optim, criterion, test_embeddings, test_targets):
    for epc in range(epoch):
        out = model(input)
        _out = F.softmax(out, dim=1)
        loss = criterion(out, targets) + entropy_loss(_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epc % 10 == 0:
            acc = get_metrics(model, test_embeddings.to('cuda:0'), test_targets)
            print('train_loss: {:.4f}, test_acc: {:.4f}'.format(loss.item(), acc))
def get_metrics(model, inputs, targets):
    out = model(inputs)
    out = F.softmax(out, 1)
    preds = torch.argmax(out, 1).detach().cpu().numpy()
    acc = (preds == targets).mean()
    return acc
def get_preds(out):
    # out = F.softmax(out)
    preds = torch.argmin(out).detach().cpu().numpy()
    return preds, preds
def get_negative_labels(unlabel_out, position, _postion, thres=0.2):
    length = len(position)
    r = []
    un_idx = []
    for idx in range(length):
        pos = position[idx]
        _pos = _postion[idx]
        _out = unlabel_out[idx][pos]
        out = F.softmax(_out)
        if len(pos)==1:
            r.append(_pos[-1])
            un_idx.append(idx)
            continue
        conf = torch.min(out)
        if conf>thres:
            un_idx.append(idx)
            if len(_pos)==0:
                r.append(torch.argmin(out).detach().cpu().numpy())
            else:
                r.append(_pos[-1])
            continue
        t, _ = get_preds(out)
        a = pos[t]
        _postion[idx].append(a)
        position[idx].remove(a)
        r.append(a)
    return np.asarray(r), un_idx
def train_NL(model, pseudo_nl_embeddings, nl_label, epoch, optimizer, test_embeddings, test_targets, train_targets):
    for epc in range(epoch):
        out = model(pseudo_nl_embeddings)
        loss_neg = negative_crossentropy(out, torch.from_numpy(nl_label).to(('cuda:0')))
        loss_entropy = entropy_loss(out)
        loss = loss_neg + loss_entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        best_acc=0
        if epc % 1 == 0:
            train_acc = get_metrics(model, pseudo_nl_embeddings, train_targets)
            if train_acc>best_acc:
                best_acc=train_acc
                torch.save(model.state_dict(), args.tmp_model)
def get_wrong(unlabel_targets, pseudo, index, type):
    gts = unlabel_targets[index]
    num = len(index)
    count = 0
    wrong_idx=[]
    wrong_list = []
    for idx, item in enumerate(pseudo):
        gt = gts[idx]
        if type=='nl':
            if gt==item:
                count+=1
                wrong_idx.append(idx)
                wrong_list.append(item)
        else:
            if gt!=item:
                count+=1
                wrong_idx.append(idx)
                wrong_list.append(item)
    wrong_rate = count/(num+1e-6)
    return wrong_rate, count, num, wrong_idx
def get_metrics(model, inputs, targets):
    out = model(inputs)
    out = F.sigmoid(out)
    preds = torch.argmax(out, 1).detach().cpu().numpy()
    acc = (preds == targets).mean()
    return acc



