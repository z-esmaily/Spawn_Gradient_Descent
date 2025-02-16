import torch
from .optimizer import Optimizer, required

class SpawnGD(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SpawnGD, self).__init__(params, defaults)

    def step(self, epoch, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # Store last two points (moved outside conditions)
                param_state = self.state[p]
                if 'prev_params_buffers' not in param_state:
                    param_state['prev_params_buffers'] = [torch.clone(p.data).detach(), torch.clone(p.data).detach()]
                else:
                    param_state['prev_params_buffers'] = [param_state['prev_params_buffers'][1], torch.clone(p.data).detach()]
                
                # updates based on epoch
                if epoch == 0 or epoch % 2 == 0:  # Pure SGD
                    p.data = p.data - group['lr'] * d_p
                else:  # SGD + Spawn
                    # SGD update
                    p.data = p.data - group['lr'] * d_p

                    # Spawn step
                    if not param_state['prev_params_buffers'][0].equal(param_state['prev_params_buffers'][1]):
                        diff_params = p.data - param_state['prev_params_buffers'][0]
                        length, next_sign = torch.abs(diff_params), torch.sign(diff_params)
                        p.data = p.data + next_sign * (length * torch.rand_like(p.data))
        return loss
