from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from numpy.core.numeric import True_
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.gradcheck import zero_gradients
from utils import label_smoothing, one_hot_tensor,softCrossEntropy
import torchvision
import math
from torch.autograd import Variable
from utils import clamp
import pdb

class TOPO_Module(nn.Module):
    def __init__(self, basic_net, args, aux_net=None):
        super(TOPO_Module, self).__init__()
        self.basic_net = basic_net
        self.aux_net = aux_net
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.num_steps = 10
        self.step_size = 2.0/255
        self.epsilon = 8.0/255
        self.restarts = 1
        self.norm = "l_inf"
        self.early_stop = False
        self.args = args
        self.num_classes = args.num_classes
        self.momentum = 0.9
        self.feat_dim = args.num_classes
        self.stdv = 1. / math.sqrt(self.feat_dim / 3)
        self.register_buffer('memory_feat', torch.rand(self.num_classes, self.feat_dim).mul_(2 * self.stdv).add_(-self.stdv).cuda())

    def train(self, epoch, inputs, targets, index, optimizer):
        #### generating adversarial examples stage
        self.basic_net.eval()
        batch_size = len(inputs)
        x_adv = inputs.detach() + 0.001 * torch.randn(inputs.shape).cuda().detach()
        logits_nat = self.basic_net(x_adv)
        if self.norm == 'l_inf':
            for _ in range(self.num_steps): 
                x_adv.requires_grad_()
                with torch.enable_grad():
                    logits_adv = self.basic_net(x_adv)
                    loss_adv = self.criterion_kl(F.log_softmax(logits_adv, dim=1),
                                        F.softmax(logits_nat, dim=1))
                    # loss_adv = F.cross_entropy(logits_adv, targets)
                loss_adv = loss_adv 
                grad = torch.autograd.grad(loss_adv, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon), inputs + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        adv_inputs = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        #### adversarial tarining stage
        self.basic_net.train()
        self.basic_net.zero_grad()
        optimizer.zero_grad()

        logits_nat = self.basic_net(inputs)
        feat_norm_nat = F.normalize(logits_nat, p=2, dim=1)
        loss_nat = self.criterion(logits_nat, targets)

        index_nat = torch.where(logits_nat.max(1)[1] == targets)[0]
        feat_nat  = feat_norm_nat[index_nat].detach()
        targets_nat = targets[index_nat]

        logits_adv = self.basic_net(adv_inputs)
        feat_adv = F.normalize(logits_adv, p=2, dim=1)
        loss_adv = self.criterion(logits_adv, targets)

        ### 构建自然特征
        self.curt_feat_nat = torch.rand_like(self.memory_feat)
        self.curt_feat_adv = torch.rand_like(self.memory_feat)
        for class_ in range(self.num_classes):
            class_index = torch.where(targets_nat == class_)[0]
            if len(class_index) == 0:
                continue
            class_feat_nat = feat_nat[class_index,:].mean(0)
            class_feat_adv = feat_adv[class_index,:].mean(0)
            self.curt_feat_nat[class_] = class_feat_nat
            self.curt_feat_adv[class_] = class_feat_adv

        alpha = math.exp(-5 * (1-epoch/self.args.max_epoch)**2)
        self.memory_feat = self.curt_feat_nat* (1 - alpha) +  self.memory_feat*alpha

        weight =  math.exp(- 10 * (1-epoch/self.args.max_epoch)**2)
        loss_graph = ATA_loss(self.curt_feat_adv, self.memory_feat) 
        loss_graph = weight*loss_graph
        loss_match = (6.0/batch_size) * self.criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1))

        loss = loss_adv + loss_graph + loss_match
        loss.backward()
        optimizer.step()

        return logits_nat.detach(), logits_adv.detach(), loss.item(), loss_nat.item(), loss_adv.item()

    def test(self, inputs, targets, adversary=None):
        if adversary is not None:
            inputs = adversary.attack(inputs, targets).detach()

        self.basic_net.eval()
        logits = self.basic_net(inputs)
        loss = self.criterion(logits, targets)

        return logits.detach(), loss.item()

def ATA_loss(adv_norm_feat, nat_norm_feat):
    nat_rank = torch.mm(nat_norm_feat.detach(), (nat_norm_feat.detach().transpose(1, 0)))
    adv_rank = torch.mm(adv_norm_feat, (adv_norm_feat.transpose(1, 0)))
    x,y = nat_rank.shape

    mref = torch.mean(nat_rank, 1)
    mcur = torch.mean(adv_rank, 1)

    refm = nat_rank - mref.repeat(y).reshape(y,x).transpose(1, 0)
    curm = adv_rank - mcur.repeat(y).reshape(y,x).transpose(1, 0)
    r_num = torch.sum(refm*curm, 1)
    r_den = torch.sqrt(torch.sum(torch.pow(refm,2),1)*torch.sum(torch.pow(curm,2),1))
    r = 1 - (r_num / r_den)
    cor = torch.mean(r)
    return cor