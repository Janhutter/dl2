import math
import torch
import torch.optim as optim

def setup_optimizer(params, cfg, logger):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.MODEL.ADAPTATION.lower() == 'sar':
        base_optimizer = optim.SGD
        return SAM(params, base_optimizer,lr=cfg.OPTIM.LR, momentum=0.9)
    else:
        if cfg.OPTIM.METHOD.lower() == 'adam':
            return optim.Adam(params,
                        lr=cfg.OPTIM.LR,
                        betas=(cfg.OPTIM.BETA, 0.999),
                        weight_decay=cfg.OPTIM.WD)
        elif cfg.OPTIM.METHOD.lower() == 'sgd':
            return optim.SGD(params,
                    lr=cfg.OPTIM.LR,
                    momentum=cfg.OPTIM.MOMENTUM,
                    dampening=cfg.OPTIM.DAMPENING,
                    weight_decay=cfg.OPTIM.WD,
                    nesterov=cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError


def setup_optimizer_with_warmup(params, cfg, logger):
    optimizer = setup_optimizer(params, cfg, logger)
    
    if cfg.OPTIM.METHOD.lower() == 'adam':
        # Store initial LR and warmup parameters
        optimizer.initial_lr = cfg.OPTIM.LR
        optimizer.step_count = 0
        
        # Warmup configuration
        warmup_steps = getattr(cfg.OPTIM, 'WARMUP_STEPS', 20)  # Default 1000 steps
        warmup_start_lr = getattr(cfg.OPTIM, 'WARMUP_START_LR', 1e-7)  # Very small starting LR
        
        # Add a custom step method that includes warmup scheduling
        original_step = optimizer.step
        def step_with_warmup(*args, **kwargs):
            result = original_step(*args, **kwargs)
            optimizer.step_count += 1
            
            if optimizer.step_count <= warmup_steps:
                # Linear warmup phase
                warmup_factor = optimizer.step_count / warmup_steps
                new_lr = warmup_start_lr + (optimizer.initial_lr - warmup_start_lr) * warmup_factor
                logger.info(f"Warmup step {optimizer.step_count}/{warmup_steps}, LR: {new_lr:.6f}")
            else:
                # # Post-warmup: keep constant or apply decay
                # # Option 1: Constant LR after warmup
                # new_lr = optimizer.initial_lr
                
                # # Option 2: Cosine decay after warmup (uncomment if desired) NOT CORRECT
                # remaining_steps = cfg.OPTIM.STEPS - warmup_steps
                # decay_step = optimizer.step_count - warmup_steps
                # new_lr = 0.5 * optimizer.initial_lr * (1 + math.cos(math.pi * decay_step / remaining_steps))
                
                # Option 3: Exponential decay after warmup (uncomment if desired)
                decay_rate = 0.9975 # getattr(cfg.OPTIM, 'DECAY_RATE', 0.99)
                new_lr = optimizer.initial_lr * (decay_rate ** (optimizer.step_count - warmup_steps))
                logger.info(f"step {optimizer.step_count - warmup_steps}, LR: {new_lr:.6f}")
            

            # Apply the new learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
                
            return result
            
        optimizer.step = step_with_warmup
        
        logger.info(f"Warmup scheduler configured: {warmup_steps} steps, "
                   f"start_lr: {warmup_start_lr}, target_lr: {optimizer.initial_lr}")
    
    return optimizer

def setup_energy_optimizer(params, cfg, logger):
    if cfg.MODEL.ADAPTATION.lower() == 'sar':
        base_optimizer = optim.SGD
        return SAM(params, base_optimizer, lr=cfg.OPTIM_ENERGY.LR, momentum=0.9)
    else:
        if cfg.OPTIM_ENERGY.METHOD.lower() == 'adam':
            return optim.Adam(params,
                        lr=cfg.OPTIM_ENERGY.LR,
                        betas=(cfg.OPTIM_ENERGY.BETA, 0.999),
                        weight_decay=cfg.OPTIM_ENERGY.WD)
        elif cfg.OPTIM_ENERGY.METHOD.lower() == 'sgd':
            return optim.SGD(params,
                    lr=cfg.OPTIM_ENERGY.LR,
                    momentum=cfg.OPTIM_ENERGY.MOMENTUM,
                    dampening=cfg.OPTIM_ENERGY.DAMPENING,
                    weight_decay=cfg.OPTIM_ENERGY.WD,
                    nesterov=cfg.OPTIM_ENERGY.NESTEROV)
        else:
            raise NotImplementedError


"""
from https://github.com/davda54/sam
"""
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups