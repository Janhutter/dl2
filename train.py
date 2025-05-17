import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
os.environ["ROBUSTBENCH_DATA"] = "~/ai-dl2/tea/save"


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

logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # configure base model
    if 'BN' in cfg.MODEL.ARCH:
        if (cfg.CORRUPTION.DATASET == 'cifar10' and cfg.MODEL.ARCH == 'WRN2810_BN'):
            # use robustbench
            print("load model from robustbench. No need to pretrain it!")
        elif cfg.CORRUPTION.DATASET == 'cifar100' or cfg.CORRUPTION.DATASET == 'tin200':
            base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            # need to pretrain it
            train(cfg, base_model, device)

        elif cfg.CORRUPTION.DATASET == 'pacs' or cfg.CORRUPTION.DATASET == 'mnist' :
            base_model = build_model_res18bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            # need to pretrain it
            train(cfg, base_model, device)
        else:
            raise NotImplementedError
    elif 'GN' in cfg.MODEL.ARCH:
        group_num=int(cfg.MODEL.ARCH.split("_")[-1])
        base_model = build_model_res50gn(group_num, cfg.CORRUPTION.NUM_CLASSES).to(device)
        # need to pretrain it
        train(cfg, base_model, device)

    elif 'VIT' in cfg.MODEL.ARCH:

        base_model = build_vit(num_classes=cfg.CORRUPTION.NUM_CLASSES, dropout_rate=0).to(device)


        # need to pretrain it
        train(cfg, base_model, device)


def train(cfg, model, device):
    logger = set_logger(cfg)

    train_dataset, test_dataset, train_loader, test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=True, logger=logger, model_arch=cfg.MODEL.ARCH)
    optimizer = setup_optimizer(model.parameters(), cfg, logger)

    if 'vit' in cfg.MODEL.ARCH.lower():
        epochs = 20
        save_every = 1
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_loader) * epochs,
        )
    else:
        epochs = 200
        save_every = 20
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2,
        )

    for epoch in tqdm(range(1, epochs + 1), mininterval=5, desc="Training", unit="epoch"):
        train_base(epoch, model, train_loader, optimizer, scheduler, cfg)

        if epoch % save_every == 0:
            logger.info("epoch: {}".format(epoch))
            model.eval()
            eval_without_reset(model, cfg, logger, device, test_loader)
            model.train()
            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET))
            if not os.path.exists(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)):
                os.makedirs(os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH))

            torch.save({'state_dict': model.state_dict()}, os.path.join('ckpt', cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH, "epoch_{}.pth".format(epoch)))

    torch.save({'state_dict': model.state_dict()}, os.path.join('ckpt', cfg.CORRUPTION.DATASET, f'{cfg.MODEL.ARCH}.pth'))


def eval_without_reset(model, cfg, logger, device, test_loader):
    acc = clean_accuracy_loader(model, test_loader, logger=logger, device=device, ada=cfg.MODEL.ADAPTATION, if_adapt=True, if_vis=False)
    logger.info("Test set Accuracy: {}".format(acc))

if __name__ == '__main__':
    main()
