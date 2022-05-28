import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import CategoriesSampler, DataSet
from utils import *
from models import linear_model

def meta_evaluate(args):
    import warnings
    warnings.filterwarnings('ignore')
    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredimagenet':
        num_classes = 351
    elif args.dataset == 'cifar_fs':
        num_classes = 64
    else:
        num_classes = 64
    from models.resnet12 import resnet12
    model = resnet12(num_classes).to(args.device)
    if args.resume is not None:
        state_dict = torch.load(args.resume)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k == 'clasifier.weight' or k == 'clasifier.bias':
                new_state_dict[name] = v
            else:
                name = k[5:]
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model = nn.DataParallel(model)

    model.to(args.device)
    model.eval()
    data_root = os.path.join(args.folder)
    dataset = DataSet(data_root, 'test', args.img_size)
    sampler = CategoriesSampler(dataset.label, args.num_batches,
                                args.num_test_ways, (args.num_shots, 15, args.unlabel))
    testloader = DataLoader(dataset, batch_sampler=sampler,
                            shuffle=False, num_workers=1)
    k = args.num_shots * args.num_test_ways
    loader = tqdm(testloader, ncols=0)
    accPL = [[],[]]
    accNL = [[] for _ in range(10)]
    countNL = [[] for _ in range(10)]
    countPL = [[] for _ in range(10)]
    erNL = [[] for _ in range(10)]
    erPL = [[] for _ in range(10)]
    numPL = [[] for _ in range(10)]
    numNL = [[] for _ in range(10)]
    for ii, (data, indicator, path) in enumerate(loader):
        targets = torch.arange(args.num_test_ways).repeat(args.num_shots+15+args.unlabel).long()[
            indicator[:args.num_test_ways*(args.num_shots+15+args.unlabel)] != 0]
        data = data[indicator != 0].to(args.device)
        train_targets = targets[:k].to(args.device)
        test_targets = targets[k:k+15*args.num_test_ways].cpu().numpy()
        unlabel_targets = targets[k+15*args.num_test_ways:].cpu().numpy()
        train_embeddings = get_embedding(model, data[:k], args.device, type=True).to(args.device)
        test_embeddings = get_embedding(model, data[k:k+15*args.num_test_ways], args.device, type=True)
        if args.unlabel != 0:
            unlabel_embeddings = get_embedding(
                model, data[k+15*args.num_test_ways:], args.device, type=True).to(args.device)
        else:
            unlabel_embeddings = None

        _POSITION = [[] for _ in range(250)]
        POSITION = [[0, 1, 2, 3, 4] for _ in range(250)]

        crieterion = nn.CrossEntropyLoss()
        clf = linear_model().to(args.device)
        optimizer = torch.optim.SGD(clf.parameters(),
                                     lr=0.1, momentum=0.9, weight_decay=0.0005)
        train(clf, train_embeddings, train_targets, 100, optimizer, crieterion, test_embeddings, test_targets)
        acc_ = get_metrics(clf, test_embeddings.to(args.device), test_targets)
        accPL[0].append(acc_)
        print('PL ACC: {:.2f}'.format(acc_))
        ia =0
        optimizer_NL = torch.optim.SGD(clf.parameters(),
                                       lr=0.02, weight_decay=0.0005, momentum=0.9)
        while True:
            print('********************************************NL')
            unlabel_out = clf(unlabel_embeddings)
            nl_pred, unselect_idx = get_negative_labels(unlabel_out, POSITION, _POSITION, thres=args.nl_thres)
            select_idx = [x for x in range(250) if x not in unselect_idx]
            if len(select_idx)<1: break
            nl_logits = unlabel_embeddings[select_idx]
            nl_label = unlabel_targets[select_idx]
            nl_pred = nl_pred[select_idx]
            wr_nl, cnt_nl, num_nl, nl_idx = get_wrong(unlabel_targets, nl_pred, select_idx, type='nl')
            erNL[ia].append(wr_nl)
            countNL[ia].append(cnt_nl)
            numNL[ia].append(num_nl)
            train_NL(clf, nl_logits, nl_pred, 10, optimizer_NL, test_embeddings, test_targets, nl_label)
            clf.load_state_dict(torch.load(args.tmp_model))
            acc_nl = get_metrics(clf, test_embeddings.to(args.device), test_targets)
            accNL[ia].append(acc_nl)
            print('NL ACC: {:.2f}'.format(acc_nl))
            ia += 1
        pseudo_label = []
        index_pl = []
        for idx in range(len(POSITION)):
            item = POSITION[idx]
            if len(item)==1:
                pseudo_label.append(item[-1])
                index_pl.append(idx)
        pseudo_label = np.asarray(pseudo_label)
        wr_pl, cnt_pl, num_pl, plidx = get_wrong(unlabel_targets, pseudo_label, index_pl, type='pl')
        erPL[0].append(wr_pl)
        countPL[0].append(cnt_pl)
        numPL[0].append(num_pl)
        inputs = unlabel_embeddings[index_pl].to(args.device)
        inputs = torch.cat((train_embeddings, inputs), dim=0)
        pseudo_label = torch.Tensor(pseudo_label).long().to(args.device)
        pseudo_label = torch.cat((train_targets, pseudo_label), dim=0)
        train(clf, inputs, pseudo_label, 100, optimizer, crieterion, test_embeddings, test_targets)
        acc_ = get_metrics(clf, test_embeddings.to(args.device), test_targets)
        accPL[1].append(acc_)
        print('PL ACC: {:.2f}'.format(acc_))
    print(sum(countPL)/len(countPL))
    print(sum(erPL)/len(erPL))
    print(sum(numPL)/len(numPL))
    for nlaccs in accNL:
        print(mean_confidence_interval(nlaccs)[0], mean_confidence_interval(nlaccs)[1])
    for placcs in accPL:
        print(mean_confidence_interval(placcs)[0], mean_confidence_interval(placcs)[1])

def main(args):
    setup_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    if args.mode == 'evaluate':
        meta_evaluate(args)
    else:
        raise NameError

if __name__ == '__main__':
    args = config()
    main(args)
