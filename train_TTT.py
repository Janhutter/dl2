import os
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
# from robustbench.model_zoo.enums import ThreatModel
# from robustbench.utils import load_model

from core.eval import evaluate_ori, evaluate_ood, clean_accuracy_loader
from core.calibration import calibration_ori
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger, train_base
from core.model import build_model_wrn2810bn, build_model_res18bn, build_model_res50gn, build_vit
from core.setada import *
from core.optim import setup_optimizer
from core.data import load_dataloader
from torch.optim import lr_scheduler
from core.checkpoint import load_checkpoint
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# for rotation function from TTT original code
from ttt_core.ttt_eval import rotate_batch

logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # construct model
    base_model = None
    if cfg.MODEL.ARCH == 'WRN2810_BN':
        base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    else:
        raise NotImplementedError
    assert base_model is not None, "Base model should be initialized before training."

    # pretrain the model
    train(cfg, base_model, device)

    # old code, kept for reference!
    # configure base model
    # if 'BN' in cfg.MODEL.ARCH:
    #     if (cfg.CORRUPTION.DATASET == 'cifar10' and cfg.MODEL.ARCH == 'WRN2810_BN'):
    #         base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    #         # need to pretrain it
    #         train(cfg, base_model, device)

    #     elif cfg.CORRUPTION.DATASET == 'cifar100' or cfg.CORRUPTION.DATASET == 'tin200':
    #         base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    #         # need to pretrain it
    #         train(cfg, base_model, device)

    #     elif cfg.CORRUPTION.DATASET == 'pacs' or cfg.CORRUPTION.DATASET == 'mnist' :
    #         base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    #         # need to pretrain it
    #         train(cfg, base_model, device)
    #     else:
    #         raise NotImplementedError

    # else:
    #     raise NotImplementedError


def train(cfg, base_model, device):
    logger = set_logger(cfg)

    net, ext, head, ssh = build_model_TTT(base_model)
    parameters = list(net.parameters())+list(head.parameters())

    train_dataset, test_dataset, train_loader, test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=True, logger=logger)
    optimizer = setup_optimizer(parameters, cfg, logger)

    save_every = 15
    # epochs = 200
    # scheduler = lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[60, 120, 160],
    #     gamma=0.2,
    # )
    # julian: dit is op basis van script.sh in ttt_cifar repo (BN versie)
    epochs = 75
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 65],
        gamma=0.1,
    )

    for epoch in tqdm(range(1, epochs + 1), mininterval=5, desc='Training', unit='epoch'):
        net.train()
        ssh.train()
        correct_net = 0
        correct_ssh = 0
        total_loss_net = 0
        total_loss_ssh = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
            outputs_cls = net(inputs_cls)
            loss = F.cross_entropy(outputs_cls, labels_cls)
            total_loss_net += loss.item()

            inputs_ssh, labels_ssh = rotate_batch(inputs_cls, 'expand')
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            outputs_ssh = ssh(inputs_ssh)
            loss_ssh = F.cross_entropy(outputs_ssh, labels_ssh)
            loss += loss_ssh
            total_loss_ssh += loss_ssh.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pred_net = outputs_cls.argmax(dim=1, keepdim=True)  
            correct_net += pred_net.eq(labels_cls.view_as(pred_net)).sum().item()
            pred_ssh = outputs_ssh.argmax(dim=1, keepdim=True)
            correct_ssh += pred_ssh.eq(labels_ssh.view_as(pred_ssh)).sum().item()

        avg_loss_net = total_loss_net / len(train_loader.dataset)
        avg_loss_ssh = total_loss_ssh / len(train_loader.dataset)
        acc_net = 100. * correct_net / len(train_loader.dataset)
        acc_ssh = 100. * correct_ssh / len(train_loader.dataset)
        logger.info('Train epoch: {}, loss_net: {:.6f}, acc_net: {:.2f}%, loss_ssh: {:.6f}, acc_ssh: {:.2f}%'.format(
            epoch, avg_loss_net, acc_net, avg_loss_ssh, acc_ssh))

        if epoch % save_every == 0:
            logger.info("epoch: {}".format(epoch))
            eval_without_reset(net, cfg, logger, device, test_loader)
            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET))
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH))

            # was eerst: head: ssh.state_dict(), maar voor consistency aangepast
            torch.save({'state_dict': net.state_dict(), 'head':head.state_dict()}, os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH, "epoch_TTT_100_{}.pth".format(epoch)))

    torch.save({'state_dict': net.state_dict(), 'head':head.state_dict()}, os.path.join('ckpt', cfg.CORRUPTION.DATASET, f'{cfg.MODEL.ARCH}_TTT_100.pth'))


def eval_without_reset(net, cfg, logger, device, test_loader):
    acc = clean_accuracy_loader(net, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
    logger.info("Test set Accuracy: {}".format(acc))


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))

# changed
def extractor_from_layer3(net):
    layers = [net.conv1, net.block1, net.block2, net.block3, net.bn1, net.relu, nn.AvgPool2d(8), ViewFlatten()]
    return nn.Sequential(*layers)

# changed
def extractor_from_layer2(net):
    layers = [net.conv1, net.block1, net.block2]
    return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
    head = copy.deepcopy([net.block3, net.bn1, net.relu, nn.AvgPool2d(8)])
    head.append(ViewFlatten())
    head.append(nn.Linear(64 * width, classes))
    return nn.Sequential(*head)

def build_model_TTT(base_model):
    aux_classes = 4
    net = base_model
    ext = extractor_from_layer2(net)
    aux_head = head_on_layer2(net, 10, aux_classes)
    ssh = ExtractorHead(ext, aux_head).cuda()

    return net, ext, aux_head, ssh


if __name__ == '__main__':
    main()
