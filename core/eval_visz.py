import numpy as np
import pandas as pd
import math
import wandb
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

    steps = {0:1, 1:2, 2:2, 3:3, 4:3, 5:3, 6:3, 7:4, 8:4, 9:5}
    # lr_epch = {0:0.0001, 1:0.0001, 2:0.0001, 3:0.00005, 4:0.00005, 5:0.00003, 6:0.00003, 7:0.00003, 8:0.00001, 9:0.00001}
    
    with torch.no_grad():
        for epoch_idx in range(n_epochs):

            curr_steps = steps[epoch_idx]
            model.steps = curr_steps
            # model.optimizer.lr = lr_epch[epoch_idx]

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
                wandb.log({'batch_acc (will be jittery)': batch_accuracy.item()})
                logger.warning(f"batch_acc: {batch_accuracy.item()}")

                acc += batch_accuracy
        
            epoch_acc = acc.item() / n_batches
            wandb.log({'epoch_acc': epoch_acc})

            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET))
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH))

            torch.save(model.state_dict(), os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH, f"visz_pt_epoch_{epoch_idx}_bn.pth"))


    return epoch_acc

# def clean_accuracy_loader(model, test_loader, logger=None, device=None, ada=None, if_adapt=True, if_vis=False):
#     test_loss = 0
#     correct = 0
#     index = 1
#     total_step = math.ceil(len(test_loader.dataset) / test_loader.batch_size)
#     with torch.no_grad():
#         for counter, (data, target) in enumerate(test_loader):
#             logger.info("Test Batch Process: {}/{}".format(index, total_step))
#             data, target = data.to(device), target.to(device)
#             with torch.no_grad():
#                 if ada == 'source':
#                     output = model(data)
#                 else:
#                     output = model(data, if_adapt=if_adapt)

#             test_loss += F.cross_entropy(output, target).item() 
#             pred = output.argmax(dim=1, keepdim=True)  
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             index = index + 1
            
#     test_loss /= len(test_loader.dataset)
#     acc = 100. * correct / len(test_loader.dataset)
#     return acc

def evaluate_visz(model, cfg, logger, device):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):
        # try:
        #     model.reset()
        #     logger.info("resetting model")
        # except:
        #     logger.warning("not resetting model")
        logger.warning("not resetting model")

        x_test, y_test = load_data(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.NUM_EX, None, cfg.DATA_DIR) # NB: shuffle not passed here, because it's not passed to RobustBench's load_cifar10!
        x_test, y_test = x_test.to(device), y_test.to(device)
        acc = clean_accuracy(cfg, model, x_test, y_test, n_epochs=cfg.OPTIM.N_EPOCHS, batch_size=cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=True)           
        logger.info(f"acc: {acc:.2%}")

    
    # elif cfg.CORRUPTION.DATASET == 'mnist':
    #     _, _, _, test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
    #     acc = clean_accuracy_loader(model, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION,  if_adapt=True, if_vis=True)
    #     logger.info("Test set Accuracy: {}".format(acc))
    
    # elif cfg.CORRUPTION.DATASET == 'pacs':
    #     accs=[]
    #     for target_domain in cfg.CORRUPTION.TYPE:
    #         x_test, y_test = load_data(data = cfg.CORRUPTION.DATASET, data_dir=cfg.DATA_DIR, shuffle = True, corruptions=target_domain)
    #         x_test, y_test = x_test.to(device), y_test.to(device)
    #         out = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=True)
    #         if cfg.MODEL.ADAPTATION == 'energy':
    #             acc = out
    #         else:
    #             acc = out
    #         logger.info(f"acc % [{target_domain}]: {acc:.2%}")
    #         accs.append(acc)
    #     acc_mean=np.array(accs).mean()
    #     logger.info(f"mean acc:%: {acc_mean:.2%}")
    # else:
    #     raise NotImplementedError

