import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from core.eval_visz import evaluate_visz
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger
from core.model import build_model_wrn2810bn, build_model_res18bn, build_model_res50gn, build_vit
from core.setada import *


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
            model = 'Standard'
            base_model = load_model(model, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).to(device)
        elif cfg.CORRUPTION.DATASET == 'cifar100' or cfg.CORRUPTION.DATASET == 'tin200':
            base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        elif cfg.CORRUPTION.DATASET == 'pacs' or cfg.CORRUPTION.DATASET == 'mnist' :
            base_model = build_model_res18bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        else:
            raise NotImplementedError
    elif 'GN' in cfg.MODEL.ARCH:
        group_num=int(cfg.MODEL.ARCH.split("_")[-1])
        base_model = build_model_res50gn(group_num, cfg.CORRUPTION.NUM_CLASSES).to(device)
        ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
        base_model.load_state_dict(ckpt['state_dict'])
    elif cfg.MODEL.ARCH == 'VIT_16':
        image_size = (cfg.CORRUPTION.IMG_SIZE, cfg.CORRUPTION.IMG_SIZE)
        patch_size = (16, 16)
        base_model = build_vit(image_size=image_size,
                patch_size=patch_size,
                emb_dim=768,
                mlp_dim=3072,
                num_heads=12,
                num_layers=12,
                num_classes=cfg.CORRUPTION.NUM_CLASSES,
                attn_dropout_rate=0.0,
                dropout_rate=0.1,
                head = 'both',
                feat_dim=128,
                contrastive=False,
                timm=True).to(device)
        ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET, cfg.MODEL.ARCH)))
        base_model.load_state_dict(ckpt['state_dict'])
    else:
        raise NotImplementedError

    # configure tta model
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy_visz(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    evaluate_visz(model, cfg, logger, device)


if __name__ == '__main__':
    main()
