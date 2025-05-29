import os
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
os.environ["ROBUSTBENCH_DATA"] = "~/ai-dl2/tea/save"


from core.eval import evaluate_ori, evaluate_ood, clean_accuracy_loader
from core.calibration import calibration_ori
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger, train_base
from core.model import build_model_wrn2810bn, build_model_res18bn, build_model_res50gn, build_vit
from core.setada import *
from core.optim import setup_optimizer, setup_energy_optimizer
from core.data import load_dataloader
from torch.optim import lr_scheduler
from core.checkpoint import load_checkpoint
from transformers import get_cosine_schedule_with_warmup

# from ttt_cifar_release.utils.rotation import rotate_batch
from core.adazoo.energy import init_random, sample_q, EnergyModel
from core.param import collect_params

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.eval import clean_accuracy
from core.data import load_data

logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # construct model
    base_model = None
    if 'WRN2810' in cfg.MODEL.ARCH:
        base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    elif 'VIT' in cfg.MODEL.ARCH:
        base_model = build_vit(num_classes=cfg.CORRUPTION.NUM_CLASSES, dropout_rate=0).to(device)
    else:
        raise NotImplementedError
    assert base_model is not None, "Base model should be initialized before training."

    # pretrain the model
    train(cfg, base_model, device)


def make_loss_acc_plot(train_total_losses, train_cls_losses, train_energy_losses, eval_accs, eval_interval):
    """
    Make a plot of the training loss and evaluation accuracy over epochs.
    Args:
        train_losses (list): List of training losses for each epoch.
        eval_accs (list): List of evaluation accuracies for each epoch.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss', color=color)
    epochs = np.arange(1, len(train_total_losses) + 1)
    ax1.plot(epochs, train_total_losses, color=color, label='Total loss')
    ax1.plot(epochs, train_cls_losses, color='tab:orange', label='Classifier loss')
    ax1.plot(epochs, train_energy_losses, color='tab:green', label='Energy loss')
    ax1.tick_params(axis='y', labelcolor=color)
    # add grid lines in the plot
    ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax2 = ax1.twinx()
    # plot a datapoint at every i epochs
    x_points = np.arange(len(eval_accs)) * eval_interval + eval_interval
    y_points = eval_accs
    color = 'tab:blue'
    ax2.set_ylabel('Eval accuracy', color=color)
    ax2.plot(x_points, y_points, color=color, label='Eval accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.legend()
    fig.tight_layout()
    # multiply dpi by 2
    fig.set_dpi(200)
    # increase figure size by 200%
    fig.set_size_inches(10, 5)
    # save in logs/TET/${SLURM_JOB_ID}.png
    os.makedirs('logs/TET', exist_ok=True)
    file_dest = f'logs/TET/{os.environ["SLURM_JOB_ID"]}_loss_acc.png' if 'SLURM_JOB_ID' in os.environ else 'test_plot.png'
    plt.savefig(file_dest)


def train(cfg, base_model, device):
    # torch.autograd.set_detect_anomaly(True)
    logger = set_logger(cfg, silent=True)

    train_dataset, test_dataset, train_loader, test_loader = load_dataloader(
          root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, 
          if_shuffle=True, logger=logger, test_batch_size=cfg.OPTIM.TEST_BATCH_SIZE, model_arch=cfg.MODEL.ARCH)
    length = len(train_loader) 

    if 'vit' in cfg.MODEL.ARCH.lower():
        net = setup_energy(base_model, cfg, logger, setup_energy_optimizer)
        cls_params = net.parameters()
        cls_optimizer = setup_optimizer(cls_params, cfg, logger)
        eval_every = 1

        cls_scheduler = get_cosine_schedule_with_warmup(
            cls_optimizer,
            num_warmup_steps=500,
            num_training_steps=cfg.OPTIM.N_EPOCHS * len(train_loader),
        )
    else:
        net = setup_energy(base_model, cfg, logger, setup_optimizer)
        # cls_params = net.parameters()

        optimizer = net.optimizer

        eval_every = 1
        cls_scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[m * length for m in cfg.OPTIM.SCHEDULER_MILESTONES],
            gamma=cfg.OPTIM.SCHEDULER_GAMMA,
        )

    epochs_no_improvement = 0
    best_eval_acc = float('-inf')
    train_total_losses = []
    train_energy_losses = []
    train_cls_losses = []
    eval_accs = []

    energy_lambda_threshold = 0  # start increasing lambda_energy after test acc >= 0.85

    start_energy_lambda = 0.00001
    curr_energy_lambda = start_energy_lambda
    base_energy_lambda = cfg.OPTIM.LAMBDA_ENERGY
    energy_lambda_gamma = 1.1
    eval_acc = 0
    energy_loss = torch.tensor(0)

    for epoch in tqdm(range(1, cfg.OPTIM.N_EPOCHS + 1), desc="Training", unit="epoch", mininterval=5):
        # train_base(epoch, model, train_loader, optimizer, scheduler, cfg)

        net.train()
        epoch_total_loss = 0
        epoch_energy_loss = 0
        epoch_cls_loss = 0
        start_warmup = True
        curr_step = 0

        # if eval_acc >= energy_lambda_threshold:
        #     if curr_energy_lambda == 0:
        #         curr_energy_lambda = start_energy_lambda
        #         start_warmup = True
        #     else:
        #         curr_energy_lambda *= energy_lambda_gamma
        #         start_warmup = False

        print(f"curr_energy_lambda: {curr_energy_lambda}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if start_warmup:
                # exponential warmup from 0 to base_energy_lambda
                curr_energy_lambda = start_energy_lambda + (base_energy_lambda - start_energy_lambda) * (curr_step / 500)
                if curr_energy_lambda > base_energy_lambda:
                    curr_energy_lambda = base_energy_lambda
                    start_warmup = False
                curr_step += 1

            optimizer.zero_grad()

            inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
            # adapts the model, still needs to step this loss:

            if curr_energy_lambda > 0:
                outputs, energy_loss = net(inputs_cls, if_adapt=True, tet=True) 
                cls_loss = F.cross_entropy(outputs, labels_cls)
                final_loss = cfg.OPTIM.LAMBDA_CLS * cls_loss +  curr_energy_lambda* energy_loss
            else:
                outputs = net(inputs_cls, if_adapt=False, tet=True)
                cls_loss = F.cross_entropy(outputs, labels_cls)
                final_loss = cfg.OPTIM.LAMBDA_CLS * cls_loss

            final_loss.backward()
            optimizer.step()
            cls_scheduler.step()


            total_loss = final_loss.item()


            epoch_total_loss += total_loss
            epoch_energy_loss += energy_loss.item() * curr_energy_lambda
            epoch_cls_loss += cls_loss.item() * cfg.OPTIM.LAMBDA_CLS

        train_total_losses.append(epoch_total_loss / len(train_loader))
        train_energy_losses.append(epoch_energy_loss / len(train_loader))
        train_cls_losses.append(epoch_cls_loss / len(train_loader))


        # eval and saving
        if epoch % eval_every == 0:
            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET))
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH))
            
            # torch.save(net.state_dict(), os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH, f"TET_epoch_{epoch}.pth"))
            net.eval()
            eval_acc = eval_without_reset(net, cfg, logger, device, test_loader)
            eval_accs.append(eval_acc)
            # ckpt = torch.load(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH, f"TET_epoch_{epoch}.pth"))
            # net.load_state_dict(ckpt)
            net.train()

            # early stopping logic
            if epoch >= cfg.EARLY_STOP_BEGIN:
                # check if model improved
                if eval_acc > best_eval_acc:
                    epochs_no_improvement = 0
                    best_eval_acc = eval_acc

                else:
                    epochs_no_improvement += eval_every

                    logger.info(f"Model did not improve (eval acc: {eval_acc}, best: {best_eval_acc},"
                        f"{epochs_no_improvement}/{cfg.EARLY_STOP_PATIENCE} in a row)")

                    if epochs_no_improvement >= cfg.EARLY_STOP_PATIENCE:
                        logger.info(f"Early stop after {epochs_no_improvement} epochs")
                        break
            
            # make intermediate plot
            try:
                make_loss_acc_plot(train_total_losses, train_cls_losses, train_energy_losses, eval_accs, eval_interval=eval_every)
            except Exception:
                logging.exception(f"Error while plotting, {train_total_losses=}, {train_cls_losses=}, {train_energy_losses=}, {eval_accs=}")
    
    # make final plot
    try:
        make_loss_acc_plot(train_total_losses, train_cls_losses, train_energy_losses, eval_accs, eval_interval=eval_every)
    except Exception:
        logging.exception(f"Error while plotting, {train_total_losses=}, {train_cls_losses=}, {train_energy_losses=}, {eval_accs=}")
    
    torch.save(net.state_dict(), os.path.join('ckpt', cfg.CORRUPTION.DATASET, f'{cfg.MODEL.ARCH}.pth'))


def eval_without_reset(net, cfg, logger, device, test_loader):
    x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR, model_arch=cfg.MODEL.ARCH, shuffle=True)
    x_test, y_test = x_test.to(device), y_test.to(device)
    acc = clean_accuracy(net, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=False, if_vis=False)
    logger.info("Test set Accuracy: {}".format(acc))
    return acc


# class ViewFlatten(nn.Module):
# 	def __init__(self):
# 		super(ViewFlatten, self).__init__()

# 	def forward(self, x):
# 		return x.view(x.size(0), -1)

# class ExtractorHead(nn.Module):
# 	def __init__(self, ext, head):
# 		super(ExtractorHead, self).__init__()
# 		self.ext = ext
# 		self.head = head

# 	def forward(self, x):
# 		return self.head(self.ext(x))

# changed
# def extractor_from_layer3(net):
# 	layers = [net.conv1, net.block1, net.block2, net.block3, net.bn1, net.relu, nn.AvgPool2d(8), ViewFlatten()]
# 	return nn.Sequential(*layers)

# # changed
# def extractor_from_layer2(net):
# 	layers = [net.conv1, net.block1, net.block2]
# 	return nn.Sequential(*layers)

# def head_on_layer2(net, width, classes):
# 	head = copy.deepcopy([net.block3, net.bn1, net.relu, nn.AvgPool2d(8)])
# 	head.append(ViewFlatten())
# 	head.append(nn.Linear(64 * width, classes))
# 	return nn.Sequential(*head)

# def build_model_TET(base_model):
#     # aux_classes = 10
#     net = base_model
#     ext = extractor_from_layer3(net)
    
#     # dit is op basis van "energy & classification gebruikt zelfde (laatste) layer logits" idee
#     # head = head_on_layer2(net, 10, classes)
#     # head = nn.Linear(64 * 10, classes)

#     # ssh = ExtractorHead(ext, head).cuda()
#     # ssh = EnergyModel(ExtractorHead(ext, head).cuda())
    
#     return net, ext, head, ssh


if __name__ == '__main__':
    main()


#     # EBM subtask
#     # save the results somewhere
#     pos_sample = inputs_cls
#     random_sample = init_random(cfg.OPTIM.BATCH_SIZE)
#     neg_sample, _ = sample_q(net, random_sample, n_steps=20, sgld_std=0.01, sgld_lr= 0.1 ,reinit_freq=0.05, batch_size=cfg.OPTIM.BATCH_SIZE, im_sz=32, n_ch=3, device=device,  y=None)

#     loss.backward()
#     cls_optimizer.step()
#     cls_scheduler.step()

#     out_real = ssh_energy(pos_sample)
#     energy_real = out_real[0].mean()
#     energy_fake = ssh_energy(neg_sample)[0].mean()
#     loss_ssh = (- (energy_real - energy_fake)) 
#     loss += loss_ssh