import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DLR_Loss(nn.Module):
    def __init__(self):
        super(DLR_Loss, self).__init__()

    def forward(self, logits, y_true, y_target=None):
        
        sorted_logits, indices = torch.sort(logits, dim=1, descending=True)
        z_pi_1 = sorted_logits[:, 0]
        z_pi_3 = sorted_logits[:, 2]

        if y_target is None:

            scale = z_pi_1 - z_pi_3 + 1e-12
            z_y = logits.gather(1, y_true.view(-1, 1)).view(-1)
            
            # 获取除了 y 之外最大的 logit
            logits_clone = logits.clone()
            logits_clone.scatter_(1, y_true.view(-1, 1), -float('inf'))
            z_other_max = logits_clone.max(dim=1).values
            
            loss = - (z_y - z_other_max) / scale
            
        else:
            z_pi_4 = sorted_logits[:, 3]
            scale = z_pi_1 - (z_pi_3 + z_pi_4) / 2.0 + 1e-12
            
            # 获取真实类别 (z_y) 和 目标类别 (z_t) 的 logits
            z_y = logits.gather(1, y_true.view(-1, 1)).view(-1)
            z_t = logits.gather(1, y_target.view(-1, 1)).view(-1)
            
            loss = - (z_y - z_t) / scale
            
        return loss

def apgd_attack(model, x, y, loss_fn, eps, n_iter=100, device='cpu'):
    """
    Auto-PGD (APGD) 实现 (L-inf 范数)
    """
    x = x.to(device)
    y = y.to(device)
    x_orig = x.clone().detach()
    batch_size = x.shape[0]

    alpha_momentum = 0.75 
    rho = 0.75            
    
    # 动态生成检查点 W
    p = [0, 0.22]
    while p[-1] < 1:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    w = [int(n_iter * p_i) for p_i in p[:-1]]
    w.append(n_iter)
    W = set(w)
    
    eta = torch.full((batch_size, 1, 1), 2.0 * eps, device=device)
    
    noise = torch.zeros_like(x).uniform_(-eps, eps)
    x_k = torch.clamp(x + noise, -1, 1).detach().requires_grad_(True)
    
    x_k_minus_1 = x_k.clone().detach()
    x_max = x_k.clone().detach()       
    
    with torch.no_grad():
        logits = model(x_k)
        f_max = loss_fn(logits, y)
        
    last_loss = f_max.clone() 
    
    count_success = torch.zeros(batch_size, device=device)
    f_max_old = f_max.clone() 
    w_j_minus_1 = 0 

    for k in range(n_iter):
        x_k.requires_grad_()
        
        logits = model(x_k)
        loss = loss_fn(logits, y)
        
        model.zero_grad()
        loss.sum().backward()
        grad = x_k.grad.data
        
        with torch.no_grad():
            idx_improved = loss > f_max
            f_max[idx_improved] = loss[idx_improved]
            x_max[idx_improved] = x_k[idx_improved]
            
            idx_step_improved = loss > last_loss
            count_success[idx_step_improved] += 1
            
            last_loss = loss.clone()

        # 梯度更新
        grad_sign = grad.sign()

        # PGD Step
        z_k_plus_1 = x_k + eta * grad_sign
        z_k_plus_1 = torch.clamp(z_k_plus_1, x_orig - eps, x_orig + eps)
        z_k_plus_1 = torch.clamp(z_k_plus_1, -1, 1)
        
        # Momentum Step
        x_k_plus_1 = x_k + alpha_momentum * (z_k_plus_1 - x_k) + (1 - alpha_momentum) * (x_k - x_k_minus_1)
        x_k_plus_1 = torch.clamp(x_k_plus_1, x_orig - eps, x_orig + eps)
        x_k_plus_1 = torch.clamp(x_k_plus_1, -1, 1).detach()
        
        x_k_minus_1 = x_k.clone().detach()
        x_k = x_k_plus_1.clone().detach().requires_grad_(True)
        
        # 检查点逻辑
        if (k + 1) in W:
            period_len = (k + 1) - w_j_minus_1
            cond1 = count_success < (rho * period_len)
            cond2 = (f_max == f_max_old)
            to_decay = cond1 | cond2
            
            # 执行衰减 & 重启
            if to_decay.any():
                eta[to_decay] /= 2.0
                x_k.data[to_decay] = x_max[to_decay]
                x_k_minus_1[to_decay] = x_max[to_decay]
                
                last_loss[to_decay] = f_max[to_decay] 
            
            count_success = torch.zeros(batch_size, device=device)
            w_j_minus_1 = k + 1
            f_max_old = f_max.clone()
            
    return x_max.detach()

def apgd_attack_targeted(model, x, y_true, y_target, loss_fn, eps, n_iter=100, device='cpu'):
    """
    Auto-PGD (APGD) 实现 (L-inf 范数, 定向)
    """
    x = x.to(device)
    y_true = y_true.to(device)
    y_target = y_target.to(device)
    x_orig = x.clone().detach()
    batch_size = x.shape[0]

    alpha_momentum = 0.75 
    rho = 0.75            
    
    p = [0, 0.22]
    while p[-1] < 1:
        p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
    w = [int(n_iter * p_i) for p_i in p[:-1]]
    w.append(n_iter) 
    W = set(w)
    
    eta = torch.full((batch_size, 1, 1), 2.0 * eps, device=device)
    
    noise = torch.zeros_like(x).uniform_(-eps, eps)
    x_k = torch.clamp(x + noise, -1, 1).detach().requires_grad_(True)
    
    x_k_minus_1 = x_k.clone().detach()
    x_max = x_k.clone().detach() 
    
    with torch.no_grad():
        logits = model(x_k)
        f_max = loss_fn(logits, y_true, y_target)
        
    last_loss = f_max.clone() 

    count_success = torch.zeros(batch_size, device=device)
    f_max_old = f_max.clone() 
    w_j_minus_1 = 0

    for k in range(n_iter):
        x_k.requires_grad_()
        
        logits = model(x_k)
        loss = loss_fn(logits, y_true, y_target)
        
        model.zero_grad()
        loss.sum().backward()
        grad = x_k.grad.data
        
        with torch.no_grad():
            idx_improved = loss > f_max
            f_max[idx_improved] = loss[idx_improved]
            x_max[idx_improved] = x_k[idx_improved]
            
            
            idx_step_improved = loss > last_loss
            count_success[idx_step_improved] += 1
            last_loss = loss.clone()
        
        grad_sign = grad.sign()

        z_k_plus_1 = x_k + eta * grad_sign
        z_k_plus_1 = torch.clamp(z_k_plus_1, x_orig - eps, x_orig + eps)
        z_k_plus_1 = torch.clamp(z_k_plus_1, -1, 1)
        
        x_k_plus_1 = x_k + alpha_momentum * (z_k_plus_1 - x_k) + (1 - alpha_momentum) * (x_k - x_k_minus_1)
        x_k_plus_1 = torch.clamp(x_k_plus_1, x_orig - eps, x_orig + eps)
        x_k_plus_1 = torch.clamp(x_k_plus_1, -1, 1).detach() 
        
        x_k_minus_1 = x_k.clone().detach()
        x_k = x_k_plus_1.clone().detach().requires_grad_(True)
        
        if (k + 1) in W:
            period_len = (k + 1) - w_j_minus_1
            cond1 = count_success < (rho * period_len)
            cond2 = (f_max == f_max_old)
            to_decay = cond1 | cond2

            if to_decay.any():
                eta[to_decay] /= 2.0  
                x_k.data[to_decay] = x_max[to_decay]
                x_k_minus_1[to_decay] = x_max[to_decay]
                
                last_loss[to_decay] = f_max[to_decay]
            
            count_success = torch.zeros(batch_size, device=device)
            w_j_minus_1 = k + 1 
            f_max_old = f_max.clone()
            
    return x_max.detach() 

class WhiteBoxAttacker:
    def __init__(self, model, epsilon=8/255, n_iter=100, device='cuda'):
        self.model = model
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.device = device
        self.model.to(device)
        self.model.eval()

    def batch_attack(self, x, y, method='apgd', loss_type='dlr', target=None):
        if loss_type == 'ce':
            loss_fn = nn.CrossEntropyLoss(reduction='none') 
        elif loss_type == 'dlr':
            loss_fn = DLR_Loss()
        
        if method == 'apgd':
            if target is None:
                return apgd_attack(self.model, x, y, loss_fn, self.epsilon, self.n_iter, self.device)
            else:
                return apgd_attack_targeted(self.model, x, y, target, loss_fn, self.epsilon, self.n_iter, self.device)
        
        elif method == 'pgd':
            # PGD L-inf Per-Sample
            x_adv = x.clone().detach()
            x_orig = x.clone().detach()
            
            noise = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv + noise, -1, 1).requires_grad_(True)
            
            alpha = self.epsilon / 4.0
            
            for _ in range(self.n_iter):
                logits = self.model(x_adv)
                loss = loss_fn(logits, y).sum()
                
                self.model.zero_grad()
                loss.backward()
                grad = x_adv.grad.data
                
                x_adv.data = x_adv.data + alpha * grad.sign()
                x_adv.data = torch.max(torch.min(x_adv.data, x_orig + self.epsilon), x_orig - self.epsilon)
                x_adv.data = torch.clamp(x_adv.data, -1, 1)
                x_adv.grad.zero_()
            
            return x_adv.detach()
