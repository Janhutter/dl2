import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
os.environ["ROBUSTBENCH_DATA"] = "~/ai-dl2/tea/save"

from core.eval_visz import evaluate_visz
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger
from core.model import build_model_wrn2810bn
from core.setada import *

logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # only implemented for wideresnet
    if cfg.MODEL.ARCH != 'WRN2810_BN':
        raise NotImplementedError

    # configure base model
    logger.info(f"loading model checkpoint from {cfg.MODEL.CHECKPOINT_PTH}")
    base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    ckpt = torch.load(cfg.MODEL.CHECKPOINT_PTH, weights_only=False)

    # remove prefix from model component names (if model was saved in this codebase)
    state_dict = {k.replace('energy_model.f.', ''): v for k, v in ckpt.items()}
    base_model.load_state_dict(state_dict, strict=False)
        
    if cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy_visz(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    evaluate_visz(model, cfg, logger, device)


if __name__ == '__main__':
    main()
