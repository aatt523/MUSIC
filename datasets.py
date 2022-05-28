import os
import os.path as osp
import pickle
import csv
import collections

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class DataSet(Dataset):
    def __init__(self, data_root, setname, img_size):
        self.img_size = img_size
        csv_path = osp.join('./data', setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        lb = -1
        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(data_root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)
        self.data = data
        self.label = label
        if setname=='test' or setname=='val':
            self.transform = transforms.Compose([
                                               transforms.Resize((img_size, img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])

        else:
            self.transform = transforms.Compose([
                                            transforms.Resize((img_size, img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if i == -1:
            return torch.zeros([3, self.img_size, self.img_size]), 0
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, 1, path

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch #num_batches
        self.n_cls = n_cls # test_ways
        self.n_per = np.sum(n_per) # num_per_class
        self.number_distract = n_per[-1]

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            indicator_batch = []
            classes = torch.randperm(len(self.m_ind))
            trad_classes = classes[:self.n_cls]
            for c in trad_classes:
                # episode class
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                # episode data
                cls_batch = l[pos]
                cls_indicator = np.zeros(self.n_per)
                cls_indicator[:cls_batch.shape[0]] = 1
                if cls_batch.shape[0] != self.n_per:
                    cls_batch = torch.cat([cls_batch, -1*torch.ones([self.n_per-cls_batch.shape[0]]).long()], 0)

                batch.append(cls_batch)
                indicator_batch.append(cls_indicator)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

filenameToPILImage = lambda x: Image.open(x).convert('RGB')

def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels


