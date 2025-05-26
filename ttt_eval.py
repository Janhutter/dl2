import copy
import torch
import torch.optim as optim
from core.data import load_data
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from core.model import build_model_wrn2810bn

""" Load TTT model """
# exact copy from original TTT repo
class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

""" Load TTT model """
# exact copy from original TTT repo
class ExtractorHead(nn.Module):
    def __init__(self, ext, head):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head

    def forward(self, x):
        return self.head(self.ext(x))

""" Load TTT model """
def extractor_from_layer2(net):

    # changed layers compared to original TTT repo
    ####################
    layers = [net.conv1, net.block1, net.block2]
    ####################

    return nn.Sequential(*layers)

""" Load TTT model """
# exact copy from original TTT repo
def head_on_layer2(net, width, classes):
    head = copy.deepcopy([net.block3, net.bn1, net.relu, nn.AvgPool2d(8)])
    head.append(ViewFlatten())
    head.append(nn.Linear(64 * width, classes))
    return nn.Sequential(*head)

""" Load TTT model """
def build_model_TTT(base_model):

    # changed, hardcoded aux_classes
    aux_classes = 4
    net = base_model
    ext = extractor_from_layer2(net)

    # only take head_on_layer2
    aux_head = head_on_layer2(net, 10, aux_classes)
    ssh = ExtractorHead(ext, aux_head).cuda()

    return net, ext, aux_head, ssh



""" Rotation adaptation code from TTT """
# exact copy from original TTT repo
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

""" Rotation adaptation code from TTT """
# exact copy from original TTT repo
def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		if label == 1:
			img = tensor_rot_90(img)
		elif label == 2:
			img = tensor_rot_180(img)
		elif label == 3:
			img = tensor_rot_270(img)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

""" Rotation adaptation code from TTT """
# exact copy from original TTT repo
# Obtain rotation labels and prediction
def rotate_batch(batch, label):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	elif label == 'expand':
		labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3])
		batch = batch.repeat((4,1,1,1))
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels



""" Changed Evaluation code from TEA for TTT """
# Calculating accuracy on test set, while adapting with TTT rotation task
def clean_accuracy_TTT(ssh, ext, net, x, y, batch_size = 100, logger=None, device = None, ada=None, if_adapt=True, if_vis=False):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    
    # adapation TTT
    ####################
    # tried both, slow gets better results ofc
    # opted for fast : lr=0.001 and 1 loops
    # opted for slow : lr=0.001 and 10 loops
    optimizer_ssh = optim.SGD(ext.parameters(), lr=0.001)
    criterion_ssh = nn.CrossEntropyLoss().cuda()
    ssh.eval()
    ext.train()
    for counter in range(n_batches):
        x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
        for iteration in range(10):
            inputs_ssh, labels_ssh = rotate_batch(x_curr, 'rand')
            inputs_ssh, labels_ssh = inputs_ssh.cuda(), labels_ssh.cuda()
            optimizer_ssh.zero_grad()
            outputs_ssh = ssh(inputs_ssh)
            loss_ssh = criterion_ssh(outputs_ssh, labels_ssh)
            loss_ssh.backward()
            optimizer_ssh.step()
    net.eval()
    ####################

    with torch.no_grad():
        energes_list=[]
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = net(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
        
    return acc.item() / x.shape[0]

""" Changed Evaluation code from TEA for TTT """
# Evaluation on CORRUPTED images from test dataset Cifar10
def evaluate_ood_TTT(cfg, logger, device):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):
        res = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        for c in range(len(cfg.CORRUPTION.TYPE)):
            for s in range(len(cfg.CORRUPTION.SEVERITY)):

                # changed
                ####################
                # load TTT model (reset model)
                base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
                net, ext, head, ssh = build_model_TTT(base_model)
                ckpt = torch.load('/home/jhutter/dl2/ckpt/cifar10/WRN2810_BN_TTT.pth',  weights_only=False)
                net.load_state_dict(ckpt['state_dict'])
                head.load_state_dict(ckpt['head'])
                ####################

                x_test, y_test = load_data(cfg.CORRUPTION.DATASET+'c', cfg.CORRUPTION.NUM_EX,
                                            cfg.CORRUPTION.SEVERITY[s], cfg.DATA_DIR, True,
                                            [cfg.CORRUPTION.TYPE[c]])
                x_test, y_test = x_test.to(device), y_test.to(device)

                # changed
                ####################
                acc = clean_accuracy_TTT(ssh, ext, net, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger)           
                ####################

                logger.info(f"acc % [{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]: {acc:.2%}")
                res[s, c] = acc

        frame = pd.DataFrame({i+1: res[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))
    
    else:
        raise NotImplementedError

""" Changed Evaluation code from TEA for TTT """
# Evaluation on clean images from test dataset Cifar10
def evaluate_ori_TTT(ssh, ext, net, cfg, logger, device):

        if 'cifar' in cfg.CORRUPTION.DATASET:
            x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR, model_arch=cfg.MODEL.ARCH)
            x_test, y_test = x_test.to(device), y_test.to(device)

            # Only clean_accuracy_TTT changed and some code removed
            ####################
            out = clean_accuracy_TTT(ssh, ext, net, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
            ####################

            if cfg.MODEL.ADAPTATION == 'energy':
                acc = out
            else:
                acc = out
            logger.info("Test set Accuracy: {}".format(acc))
