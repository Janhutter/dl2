import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
os.environ["ROBUSTBENCH_DATA"] = "~/ai-dl2/tea/save"


from core.eval_visz import evaluate_visz
# from core.eval_visz import evaluate_ood, evaluate_ori
from core.calibration import calibration_ori
from core.config import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger
from core.model import build_model_wrn2810bn, build_model_res18bn, build_model_res50gn, build_vit
from core.setada import *
from core.checkpoint import load_checkpoint


logger = logging.getLogger(__name__)

def main():
    load_cfg_fom_args()
    set_seed(cfg)
    set_logger(cfg)
    device = torch.device('cuda:0')

    # configure base model
    logger.info(f"loading model checkpoint from {cfg.MODEL.CHECKPOINT_PTH}")
    base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    # ckpt = torch.load(cfg.MODEL.CHECKPOINT_PTH)

    ckpt = torch.load(cfg.MODEL.CHECKPOINT_PTH, weights_only=False)

    # Strip prefix in one line
    state_dict = {k.replace('energy_model.f.', ''): v for k, v in ckpt.items()}

    # Debug: print available keys
    logger.info(f"Available keys in checkpoint: {list(ckpt.keys())}")
    logger.info(f"Available keys in checkpoint: {list(state_dict.keys())}")

    base_model.load_state_dict(state_dict, strict=False)


    # Then load the correct key - common alternatives:
    # if 'state_dict' in ckpt:
    #     base_model.load_state_dict(ckpt['state_dict'])
    #     logger.info(f"hello1")

    # elif 'model' in ckpt:
    #     base_model.load_state_dict(ckpt['model'])
    #     logger.info(f"hello2")

    # elif 'model_state_dict' in ckpt:
    #     base_model.load_state_dict(ckpt['model_state_dict'])
    #     logger.info(f"hello3")

    # else:
    #     # Sometimes the checkpoint IS the state dict directly
    #     base_model.load_state_dict(ckpt)
    #     logger.info(f"hello4")

    # base_model.load_state_dict(ckpt['state_dict'])


    
        
    # # configure tta model
    # if cfg.MODEL.ADAPTATION == "source":
    #     logger.info("test-time adaptation: NONE")
    #     model = setup_source(base_model, cfg, logger)
    if cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy_visz(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    # evaluate on each severity and type of corruption in turn
    # evaluate_ood(model, cfg, logger, device)
    # calibration_ori(model, cfg, logger, device)
    evaluate_visz(model, cfg, logger, device)


if __name__ == '__main__':
    main()
