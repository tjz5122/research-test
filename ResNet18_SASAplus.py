# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 02:14:15 2021

@author: thzha
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import stats
import numpy as np
import torch.optim as optim
import torchvision
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QHM(Optimizer):
    r"""
    Stochastic gradient method with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k)
        x(k+1) = x(k) - \alpha * d(k)
    "Quasi-hyperbolic momentum and Adam for deep learning"
        by Jerry Ma and Denis Yarats, ICLR 2019
    optimizer = QHM(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0)
    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range[0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range[0,1] (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum \beta and dampened gradient (1-\beta)
            \nu = \beta: SGD with "Nesterov momentum" \beta
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    Example:
        >>> optimizer = torch.optim.QHM(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0):
        # nu can take values outside of the interval [0,1], but no guarantee of convergence?
        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        super(QHM, self).__init__(params, defaults)

        # extra_buffer == True only in SSLS with momentum > 0 and nu != 1
        self.state['allocate_step_buffer'] = False

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.qhm_direction()
        self.qhm_update()

        return loss

    def add_weight_decay(self):
        # weight_decay is the same as adding L2 regularization
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if weight_decay > 0:
                    p.grad.data.add_(weight_decay, p.data)

    def qhm_direction(self):

        for group in self.param_groups:
            momentum = group['momentum']
            qhm_nu = group['qhm_nu']

            for p in group['params']:
                if p.grad is None:
                    continue
                x = p.data  # Optimization parameters
                g = p.grad.data  # Stochastic gradient

                # Compute the (negative) step directoin d and necessary momentum
                state = self.state[p]
                if abs(momentum) < 1e-12 or abs(qhm_nu) < 1e-12:  # simply SGD if beta=0 or nu=0
                    d = state['step_buffer'] = g
                else:
                    if 'momentum_buffer' not in state:
                        h = state['momentum_buffer'] = torch.zeros_like(x)
                    else:
                        h = state['momentum_buffer']
                    # Update momentum buffer: h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
                    h.mul_(momentum).add_(1 - momentum, g)

                    if abs(qhm_nu - 1) < 1e-12:  # if nu=1, then same as SGD with momentum
                        d = state['step_buffer'] = h
                    else:
                        if self.state['allocate_step_buffer']:  # copy from gradient
                            if 'step_buffer' not in state:
                                state['step_buffer'] = torch.zeros_like(g)
                            d = state['step_buffer'].copy_(g)
                        else:  # use gradient buffer
                            d = state['step_buffer'] = g
                        # Compute QHM momentum: d(k) = (1 - \nu) * g(k) + \nu * h(k)
                        d.mul_(1 - qhm_nu).add_(qhm_nu, h)

    def qhm_update(self):
        """
        Perform QHM update, need to call compute_qhm_direction() before calling this.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(-group['lr'], self.state[p]['step_buffer'])
                    
                    
class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        '''
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all, 
               2 - keep most recent 1/2, 
               3 - keep most recent 1/3,
               ... 
        fixed_len: fixed length to keep, ratio >=1 becomes irrelevant
        '''
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
        self.lowCriteria = 0
        self.highCriteria = 0
        self.meanCriteria = 0
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total_count = 0

    def double_size(self):
        self.size *= 2
        self.buffer.resize_(self.size)

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            self.double_size()                      # double the size of buffer

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total_count % self.ratio == 0:  # if leaky_count is multiple of ratio
                self.count += 1                         # increase count in queue by one
            else:                                   # otherwise leak and keep same count
                self.start += 1                         # increase start index by one

        self.total_count += 1                   # always increase total_count by one

        # reset start index to 0 and end index to count to save space
        if self.start >= self.count:
            self.buffer[0:self.count] = self.buffer[self.start:self.end]
            self.start = 0
            self.end = self.count

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item()

        if mode == 'bm':        # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=b_n).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n /(len(Yks)-1))*torch.norm(diffs).item()
            dof = b_n - 1
        elif mode == 'olbm':    # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=1).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(diffs).item()
            dof = self.count - b_n
        else:                   # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start:self.end]).item()
            dof = self.count - 1

        return mean, std, dof

    def stats_test(self, sigma, mode='bm', composite_test=False):
        mean, std, dof = self.mean_std(mode=mode)

        # confidence interval
        t_sigma_dof = stats.t.ppf(1-sigma/2., dof)
        half_width = std * t_sigma_dof / math.sqrt(self.count)
        lower = mean - half_width
        upper = mean + half_width
        # The simple confidence interval test    
        # stationarity = lower < 0 and upper > 0

        # A more stable test is to also check if two half-means are of the same sign
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        self.lowCriteria = lower
        self.highCriteria = upper
        self.meanCriteria = mean1 * mean2
        
        stationarity = (lower < 0 and upper > 0) and (mean1 * mean2 > 0)
        

        if composite_test:
            # Use two half tests to avoid false positive caused by crossing 0 in transient phase
            lb1 = mean1 - half_width
            ub1 = mean1 + half_width
            lb2 = mean2 - half_width
            ub2 = mean2 + half_width
            self.lowCriteria = lb1 * ub1
            self.highCriteria = lb2 * ub2
            self.meanCriteria = mean1 * mean2
            stationarity = (lb1 * ub1 < 0) and (lb2 * ub2 < 0) and (mean1 * mean2 > 0)

        return stationarity, mean, lower, upper  

    # method to test if average loss after line search is no longer decreasing 
    def rel_reduction(self):
        if self.count < 4:
            return 0.5
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        return (mean1 - mean2) / mean1
        
    # method to test if average loss after line search is no longer decreasing 
    def is_decreasing(self, min_cnt=1000, dec_rate=0.01):
        if self.count < min_cnt:
            return True
        half_point = self.start + int(math.floor(self.count / 2))
        mean1 = torch.mean(self.buffer[self.start : half_point]).item()
        mean2 = torch.mean(self.buffer[half_point : self.end]).item()
        return (mean1 - mean2) / mean1 > dec_rate

    def linregress(self, sigma, mode='linear'):
        """
        calculate a linear regression
        sigma: the confidence of the one-side test
            H0: slope >= 0 vs H1: slope < 0
        mode: whether log scale the x axis
        """
        TINY = 1.0e-20
        x = torch.arange(self.total_count-self.count, self.total_count,
                         dtype=self.buffer.dtype, device=self.buffer.device)
        if mode == 'log':
            x = torch.log(x)
        # both x and y has dimension (self.count,)
        xy = torch.cat([x.view(1, -1),
                        self.buffer[self.start:self.end].view(1, -1)],
                       dim=0)
        # compute covariance matrix
        fact = 1.0 / self.count
        xy -= torch.mean(xy, dim=1, keepdim=True)
        xyt = xy.t()
        cov = fact * xy.matmul(xyt).squeeze()
        # compute the t-statistics
        r_num = cov[0, 1].item()
        r_den = torch.sqrt(cov[0, 0]*cov[1, 1]).item()
        if r_den == 0.0:
            r = 0.0
        else:
            r = r_num / r_den
            # test for numerical error propagation
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0

        df = self.count - 2
        t = r * math.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
        # one-sided test for decreasing
        prob = stats.t.cdf(t, df)
        is_decreasing = prob < sigma
        # slop
        slope = r_num / cov[0, 0].item()
        return is_decreasing, slope, prob
    

class SASA(QHM):
    r"""
    Statistical Adaptive Stochastic Approximation (SASA+) with master condition.
    optimizer = SASA(params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0, 
                     warmup=0, drop_factor=2, significance=0.02, var_mode='bm', 
                     leak_ratio=4, minN_stats=400, testfreq=100, logstats=0)
    Stochastic gradient with Quasi-Hyperbolic Momentum (QHM):
        h(k) = (1 - \beta) * g(k) + \beta * h(k-1)
        d(k) = (1 - \nu) * g(k) + \nu * h(k) 
        x(k+1) = x(k) - \alpha * d(k)   
    Stationary criterion: 
        E[ <x(k),   d(k)>] - (\alpha / 2) * ||d(k)||^2 ] = 0
    or equivalently,
        E[ <x(k+1), d(k)>] + (\alpha / 2) * ||d(k)||^2 ] = 0
    Args:
        params (iterable): iterable params to optimize or dict of param groups
        lr (float): learning rate, \alpha in QHM update (default:-1 need input)
        momentum (float, optional): \beta in QHM update, range(0,1) (default:0)
        qhm_nu (float, optional): \nu in QHM update, range(0,1) (default: 1)
            \nu = 0: SGD without momentum (\beta is ignored)
            \nu = 1: SGD with momentum and dampened gradient
            \nu = \beta: SGD with "Nesterov momentum"
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        warmup (int, optional): number of steps before testing (default: 100)
        dropfactor (float, optional): factor of drop learning rate (default: 10)
        significance (float, optional): test significance level (default:0.05)  
        var_mode (string, optional): variance computing mode (default: 'mb')
        leak_ratio (int, optional): leaky bucket ratio to kept (default: 8)
        minN_stats (int, optional): min number of samples for test (default: 1000)
        testfreq (int, optional): number of steps between testing (default:100)
        logstats (int, optional): number of steps between logs (0 means no log)
    Example:
        >>> optimizer = torch.optim.SASA(model.parameters(), lr=0.1, momentum=0.9, 
        >>>                              weight_decay=0.0005)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=-1, momentum=0, qhm_nu=1, weight_decay=0, 
                 warmup=1000, drop_factor=10, significance=0.05, var_mode='mb',
                 leak_ratio=8, minN_stats=1000, testfreq=100, logstats=0):

        if lr <= 0:
            raise ValueError("Invalid value for learning rate (>0): {}".format(lr))
        if momentum < 0 or momentum > 1:
            raise ValueError("Invalid value for momentum [0,1): {}".format(momentum))
        if weight_decay < 0:
            raise ValueError("Invalid value for weight_decay (>=0): {}".format(weight_decay))
        if drop_factor < 1:
            raise ValueError("Invalid value for drop_factor (>=1): {}".format(drop_factor))
        if significance <= 0 or significance >= 1:
            raise ValueError("Invalid value for significance (0,1): {}".format(significance))
        if var_mode not in ['mb', 'olbm', 'iid']:
            raise ValueError("Invalid value for var_mode ('mb', 'olmb', or 'iid'): {}".format(var_mode))
        if leak_ratio < 1:
            raise ValueError("Invalid value for leak_ratio (int, >=1): {}".format(leak_ratio))
        if minN_stats < 100:
            raise ValueError("Invalid value for minN_stats (int, >=100): {}".format(minN_stats))
        if warmup < 0:
            raise ValueError("Invalid value for warmup (int, >1): {}".format(warmup))
        if testfreq < 1:
            raise ValueError("Invalid value for testfreq (int, >=1): {}".format(testfreq))

        super(SASA, self).__init__(params, lr=lr, momentum=momentum, qhm_nu=qhm_nu, weight_decay=weight_decay)
        # New Python3 way to call super()
        # super().__init__(params, lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay)

        # State initialization: leaky bucket belongs to global state.
        p = self.param_groups[0]['params'][0]
        if 'bucket' not in self.state:
            self.state['bucket'] = LeakyBucket(1000, leak_ratio, p.dtype, p.device)

        self.state['lr'] = float(lr)
        self.state['drop_factor'] = drop_factor
        self.state['significance'] = significance
        self.state['var_mode'] = var_mode
        self.state['minN_stats'] = int(minN_stats)
        self.state['warmup'] = int(warmup)
        self.state['testfreq'] = int(testfreq)
        self.state['logstats'] = int(logstats)
        self.state['composite_test'] = True     # first drop use composite test
        self.state['nSteps'] = 0                # steps counter +1 every iteration

        # statistics to monitor
        self.state['stats_x1d'] = 0
        self.state['stats_ld2'] = 0
        self.state['stats_val'] = 0
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['stats_mean'] = 0
        self.state['stats_lb'] = 0
        self.state['stats_ub'] = 0
        self.state["lowCriteria"] = 0
        self.state['highCriteria'] = 0
        self.state['meanCriteria'] = 0

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates model and returns loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.add_weight_decay()
        self.qhm_direction()
        self.qhm_update()
        self.state['nSteps'] += 1
        self.stats_adaptation()

        return loss

    def stats_adaptation(self):

        # compute <x(k+1), d(k)> and ||d(k)||^2 for statistical test
        self.state['stats_x1d'] = 0.0
        self.state['stats_ld2'] = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                xk1 = p.data.view(-1)
                dk = self.state[p]['step_buffer'].data.view(-1)     # OK after super().step()
                self.state['stats_x1d'] += xk1.dot(dk).item()
                self.state['stats_ld2'] += dk.dot(dk).item()
        self.state['stats_ld2'] *= 0.5 * self.state['lr']

        # Gather flat buffers can take too much memory for large models
        # Compute <x(k+1), d(k)> and ||d(k)||^2 for statistical test
        # dk = self._gather_flat_buffer('step_buffer')
        # xk1 = self._gather_flat_param() 
        # self.state['stats_x1d'] = xk1.dot(dk).item()
        # self.state['stats_ld2'] = (0.5 * self.state['lr']) * (dk.dot(dk).item())

        # add statistic to leaky bucket
        self.state['stats_val'] = self.state['stats_x1d'] + self.state['stats_ld2']
        bucket = self.state['bucket']
        bucket.add(self.state['stats_val'])

        # check statistics and adjust learning rate
        self.state['stats_test'] = 0
        self.state['stats_stationary'] = 0
        self.state['lowCriteria'] = 0
        self.state['highCriteria'] = 0
        self.state['meanCriteria'] = 0
        if bucket.count > self.state['minN_stats'] and self.state['nSteps'] % self.state['testfreq'] == 0:
            stationary, mean, lb, ub = bucket.stats_test(self.state['significance'], 
                                                     self.state['var_mode'], 
                                                     self.state['composite_test'])
            self.state['stats_test'] = 1
            self.state['stats_stationary'] = int(stationary)
            
            #Microsoft stat
            self.state['stats_mean'] = mean
            self.state['stats_lb'] = lb
            self.state['stats_ub'] = ub
            #Tianhao added
            self.state['lowCriteria'] = self.state['bucket'].lowCriteria
            self.state['highCriteria'] = self.state['bucket'].highCriteria
            self.state['meanCriteria'] = self.state['bucket'].meanCriteria
            
            # perform statistical test for stationarity
            if self.state['nSteps'] > self.state['warmup'] and self.state['stats_stationary'] == 1:
                self.state['lr'] /= self.state['drop_factor']
                for group in self.param_groups:
                    group['lr'] = self.state['lr']
                self._zero_buffers('momentum_buffer')
                self.state['composite_test'] = False
                bucket.reset()

        # Log statistics only for debugging. Therefore self.state['stats_test'] remains False     
        if self.state['logstats'] and not self.state['stats_test']:
            if bucket.count > bucket.ratio and self.state['nSteps'] % self.state['logstats'] == 0:
                stationary, mean, lb, ub = bucket.stats_test(self.state['significance'], 
                                                              self.state['var_mode'],
                                                              self.state['composite_test'])
                self.state['stats_stationary'] = int(stationary)
                self.state['stats_mean'] = mean
                self.state['stats_lb'] = lb
                self.state['stats_ub'] = ub


    # methods for gather flat parameters
    def _gather_flat_param(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    # method for gathering/initializing flat buffers that are the same shape as the parameters
    def _gather_flat_buffer(self, buf_name):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name not in state:  # init buffer
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = state[buf_name].data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _zero_buffers(self, buf_name):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if buf_name in state:
                    state[buf_name].zero_()
        return None
      
      
      
  
### Design the Mgnet Network
use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out  
    
      
      
### Implementation
minibatch_size = 128
num_epochs = 360
lr = 1

# Step 1: Define a model
my_model = ResNet(BasicBlock, [2,2,2,2], num_classes=10) #ResNet18
if use_cuda:
    my_model = my_model.cuda()

# Step 2: Define a loss function and training algorithm
criterion = nn.CrossEntropyLoss()

# Step 3: load dataset
normalize = torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.ToTensor(),
                                                  normalize])

transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=False)



optimizer = SASA(my_model.parameters(), lr=1.0, momentum=0.9, weight_decay=5e-4, testfreq=min(1000, len(trainloader)))


lowCriteria_list = []
highCriteria_list = []
meanCriteria_list = []
train_accuracy_list = []
test_accuracy_list =[]
lr_list = []
avg_loss_list = []

start = timer()
for epoch in range(num_epochs):
    # Reset accumulative running loss at beginning or each epoch
    running_loss = 0
    my_model.train()
    for i, (images, labels) in enumerate(trainloader):
        if use_cuda:
          images = images.cuda()
          labels = labels.cuda()

        # Forward pass to get the loss
        outputs = my_model(images)   # We need additional 0 input for u in MgNet
        loss = criterion(outputs, labels)
        # Backward and compute the gradient
        optimizer.zero_grad()
        loss.backward()  #backpropragation
        running_loss += loss.item()
        optimizer.step() #update the weights/parameters
    avg_loss_list.append(running_loss)
        
 # Training accuracy
    my_model.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        with torch.no_grad():
          if use_cuda:
              images = images.cuda()
              labels = labels.cuda()  
          outputs = my_model(images) 
          p_max, predicted = torch.max(outputs, 1) 
          total += labels.size(0)
          correct += (predicted == labels).sum()
    training_accuracy = float(correct)/total
    train_accuracy_list.append(training_accuracy)     
    
    # Test accuracy
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(testloader):
        with torch.no_grad():
          if use_cuda:
              images = images.cuda()
              labels = labels.cuda()
          outputs = my_model(images)      # We need additional 0 input for u in MgNet
          p_max, predicted = torch.max(outputs, 1) 
          total += labels.size(0)
          correct += (predicted == labels).sum()
    test_accuracy = float(correct)/total
    test_accuracy_list.append(test_accuracy)
    current_lr = optimizer.state['lr']
    lr_list.append(current_lr)
    lowCriteria_list.append(optimizer.state['lowCriteria'])
    highCriteria_list.append(optimizer.state['highCriteria'])
    meanCriteria_list.append(optimizer.state['meanCriteria'])
   
end = timer()
print("total computational time is", end - start)

file = open("ResNet18SASAplus.txt","x")
file.write("train_accuracy_list = {}\n".format(str(train_accuracy_list)))
file.write("test_accuracy_list = {}\n".format(str(test_accuracy_list)))
file.write("lr_list = {}\n".format(str(lr_list)))
file.write("lowCriteria_list = {}\n".format(str(lowCriteria_list)))
file.write("highCriteria_list = {}\n".format(str(highCriteria_list)))
file.write("meanCriteria_list = {}\n".format(str(meanCriteria_list)))
file.write("avg_loss_list = {}\n".format(str(avg_loss_list)))
file.write("total_time = {}\n".format(str(end - start)))
file.close()
print("complete")

