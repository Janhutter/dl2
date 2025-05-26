import os
import logging
import wandb

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

    wandb.init(project="TET",
        name="visz-pt-model",
        config={
        "model.ADAPTATION": cfg.MODEL.ADAPTATION,
        "model.ARCH": cfg.MODEL.ARCH,
        "model.ADA_PARAM": cfg.MODEL.ADA_PARAM,
        "EBM.UNCOND": cfg.EBM.UNCOND,
        "EBM.STEPS": cfg.EBM.STEPS,
        "EBM.SGLD_LR": cfg.EBM.SGLD_LR,
        "EBM.SGLD_STD": cfg.EBM.SGLD_STD,
        "EBM.BUFFER_SIZE": cfg.EBM.BUFFER_SIZE,
        "EBM.REINIT_FREQ": cfg.EBM.REINIT_FREQ,
        "OPTIM.BATCH_SIZE": cfg.OPTIM.BATCH_SIZE,
        "OPTIM.TEST_BATCH_SIZE": cfg.OPTIM.TEST_BATCH_SIZE,
        "OPTIM.METHOD": cfg.OPTIM.METHOD,
        "OPTIM.LR": cfg.OPTIM.LR,
        "OPTIM.WD": cfg.OPTIM.WD,
        "OPTIM.DAMPENING": cfg.OPTIM.DAMPENING,
        "OPTIM.NESTEROV": cfg.OPTIM.NESTEROV,
        "OPTIM.STEPS": cfg.OPTIM.STEPS,
        "OPTIM.MOMENTUM": cfg.OPTIM.MOMENTUM,
        "OPTIM.LAMBDA_ENERGY": cfg.OPTIM.LAMBDA_ENERGY,
        "OPTIM.WARMUP_START_LR": cfg.OPTIM.WARMUP_START_LR,
        "OPTIM.WARMUP_STEPS": cfg.OPTIM.WARMUP_STEPS,
        "N_EPOCHS": cfg.OPTIM.N_EPOCHS
        }
    )

    # configure base model
    logger.info(f"loading model checkpoint from {cfg.MODEL.CHECKPOINT_PTH}")
    base_model = build_model_wrn2810bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
    # ckpt = torch.load(cfg.MODEL.CHECKPOINT_PTH)

    ckpt = torch.load(cfg.MODEL.CHECKPOINT_PTH, weights_only=False)

    # Strip prefix in one line
    state_dict = {k.replace('energy_model.f.', ''): v for k, v in ckpt.items()}
    base_model.load_state_dict(state_dict)

    # # Debug: print available keys
    # logger.info(f"Available keys in checkpoint: {list(ckpt.keys())}")

    # # Then load the correct key - common alternatives:
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
