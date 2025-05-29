import numpy as np
import pandas as pd
import math
import os

import torch
import torch.nn.functional as F
from autoattack import AutoAttack
from torchvision.utils import save_image

from core.data import load_data, load_dataloader


# authors did not have code for mce for the normal results
def calc_mce(pred, y, num_classes=10, bins=10):
    mce = calibration_error(pred, y, norm='max', task='multiclass', num_classes=num_classes, n_bins=bins)
    return mce

def clean_accuracy(cfg, model, x, y, n_epochs=1, batch_size = 100, logger=None, device = None, ada=None, if_adapt=True, if_vis=True, shuffle=True):
    if device is None:
        device = x.device

    n_samples = x.shape[0]
    n_batches = math.ceil(n_samples / batch_size)

    with torch.no_grad():
        for epoch_idx in range(n_epochs):

            acc = 0.

            if shuffle:
                indices = torch.randperm(n_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled = x
                y_shuffled = y
            
            for counter in range(n_batches):

                logger.warning(f"batch_{counter=}")

                x_curr = x_shuffled[counter * batch_size:(counter + 1) *
                        batch_size].to(device)
                y_curr = y_shuffled[counter * batch_size:(counter + 1) *
                        batch_size].to(device)

                if ada == 'source':
                    output = model(x_curr)

                else:
                    output = model(x_curr, if_adapt=if_adapt, counter=counter, if_vis=if_vis)

                batch_accuracy = (output.max(1)[1] == y_curr).float().sum()
                batch_accuracy = batch_accuracy / x_curr.shape[0]
                logger.warning(f"batch_acc: {batch_accuracy.item()}")

                acc += batch_accuracy
        
            epoch_acc = acc.item() / n_batches

            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET))
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH))

            torch.save(model.state_dict(), os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH, f"visz_pt_epoch_{epoch_idx}_bn.pth"))


    return epoch_acc

def evaluate_visz(model, cfg, logger, device):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):

        logger.warning("not resetting model")

        x_test, y_test = load_data(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.NUM_EX, None, cfg.DATA_DIR) # NB: shuffle not passed here, because it's not passed to RobustBench's load_cifar10!
        x_test, y_test = x_test.to(device), y_test.to(device)
        acc = clean_accuracy(cfg, model, x_test, y_test, n_epochs=cfg.OPTIM.N_EPOCHS, batch_size=cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=True)           
        logger.info(f"acc: {acc:.2%}")

