import os
import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch.nn.utils as nn_utils

from logging.handlers import RotatingFileHandler
from core.adazoo.energy import Energy


def set_seed(cfg):
    os.environ['PYTHONHASHSEED'] =str(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

# def set_logger(cfg):
#     os.makedirs(cfg.SAVE_DIR,exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO,
#         format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
#         datefmt="%y/%m/%d %H:%M:%S",
#         handlers=[
#             logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
#             logging.StreamHandler()
#         ])

#     logger = logging.getLogger(__name__)
#     version = [torch.__version__, torch.version.cuda,
#                torch.backends.cudnn.version()]
#     logger.info(
#         "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
#     logger.info(cfg)

#     return logger


def set_logger(cfg, silent=False):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    log_handlers = []

    log_path = os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)

    try:
        # Use rotating log files to prevent unbounded growth
        file_handler = RotatingFileHandler(
            filename=log_path,
            mode='a',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8',
            delay=False
        )
        log_handlers.append(file_handler)
    except OSError as e:
        print(f"[WARNING] Could not create log file '{log_path}': {e}. Logging to console only.")

    # Always include StreamHandler (console)
    stream_handler = logging.StreamHandler()
    log_handlers.append(stream_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=log_handlers
    )

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]

    if not silent:
        logger.info("PyTorch Version: torch=%s, cuda=%s, cudnn=%s", *version)
        logger.info(cfg)

    return logger

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def train_base(epoch, model, train_loader, optimizer, scheduler=None, cfg=None):
    clip_norm = cfg.OPTIM.CLIP_NORM
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)

        optimizer.zero_grad(set_to_none=True)

        loss = F.cross_entropy(output, target)
        loss.backward()
        if clip_norm:
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


