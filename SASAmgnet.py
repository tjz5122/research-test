# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:16:18 2021

@author: thzha
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import stats
from collections import defaultdict, Iterable
from copy import deepcopy
from itertools import chain
from tensorboardX import SummaryWriter
from torch.optim import Optimizer
import numpy as np
import torchvision
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Added by Lian
required = object()

# Added by Lian
class Optimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups] #return a list that contains dicts
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

# keep the most recent half of the added items
class HalfQueue(object):
    def __init__(self, maxN, like_tens):
        self.q = torch.zeros(maxN, dtype=like_tens.dtype,
                             device=like_tens.device)
        self.n = 0 #number filled
        self.remove = False  
        self.maxN = maxN

    def double(self):
        newqueue = torch.zeros(self.maxN * 2, dtype=self.q.dtype,
                               device=self.q.device)
        newqueue[0:self.maxN][:] = self.q
        self.q = newqueue
        self.maxN *= 2

    def add(self, val):   #remove 2 element when add one element to get 1/2 of the list
        if self.remove is True:
            # remove 1
            self.q[:-1] = deepcopy(self.q[1:]) # probably slow but ok for now
        else:
            self.n += 1  #add one
        self.q[self.n - 1] = val
        if self.n == self.maxN:
            self.double()
        self.remove = not self.remove  # or self.n == self.maxN)

    def mean_std(self, mode='bm'): #default algorithm is batch mean variance
        gbar = torch.mean(self.q[:self.n])  #gbar = z_bar_n = mean for the all z 
        std_dict = {} #standard deviation 
        df_dict = {}  #degree of freedom

        # sample variance for iid samples.
        std = torch.std(self.q[:self.n]) #sample variance for all z
        std_dict['iid'] = std
        df_dict['iid'] = self.n - 1

        # batch mean variance
        b_n = int(math.floor(math.sqrt(self.n)))   #a_n = len(Yks) / b_n(batch number) take n^0.5
        Yks = F.avg_pool1d(self.q[:self.n].unsqueeze(0).unsqueeze(0),
                           kernel_size=b_n, stride=b_n).view(-1) # Yks = bar_y_s(batch mean)
        
        diffs = Yks - gbar   # bar_y_s - bar_z_n
        std = math.sqrt(b_n / (len(Yks) - 1)) * torch.norm(diffs) # hat_theta_BM = variance of batchs
        std_dict['bm'] = std
        df_dict['bm'] = b_n - 1

        # overlapping batch mean
        Yks = F.avg_pool1d(self.q[:self.n].unsqueeze(0).unsqueeze(0),
                            kernel_size=b_n, stride=1).view(-1)
        diffs = Yks - gbar
        std = math.sqrt(
            b_n * self.n / (len(Yks) * (len(Yks) - 1))) * torch.norm(diffs)
        std_dict['olbm'] = std
        df_dict['olbm'] = self.n - b_n

        return gbar, std_dict[mode], df_dict[mode], std_dict, df_dict
        # total mean / variance of batchs / degree of freedom of batches

    def reset(self):
        self.n = 0
        self.remove = False
        self.q.zero_()



# returns True if |u-v| < delta*u with signif level sigma.
def test_onesamp(z, v, sigma, delta, mode='bm', verbose=True):
    z_mean, z_std, z_df, stds, dfs = z.mean_std(mode=mode)
    v_mean, _, _, _, _ = v.mean_std()

    rhs = delta * v_mean

    K = z.n  # number of samples

    t_sigma_df = stats.t.ppf(1 - sigma / 2., z_df)
    z_upperbound = z_mean + z_std.mul(t_sigma_df / math.sqrt(K))
    z_lowerbound = z_mean - z_std.mul(t_sigma_df / math.sqrt(K))


    return (z_upperbound < rhs and z_lowerbound > -rhs), z_mean, z_upperbound, z_lowerbound, rhs, stds, dfs


class SASA(Optimizer):

    def step_onesamp(self, closure=None):
        # assert len(self.param_groups) == 1 # same as lbfgs
        # before gathering the gradient, add weight decay term
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            if weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.add_(weight_decay, p.data)

        group = self.param_groups[0]
        minN = group['minN']
        maxN = group['maxN']
        zeta = group['zeta']   #decrease rate of lr (lr = lr *zeta)
        momentum = group['momentum']  # beta
        delta = group['delta'] # test if |bar_z_n / bar_v_n| < delta
        sigma = group['sigma']  #confidence level

        # like in LBFGS, set global state to be state of first param.
        state = self.state[self._params[0]]


        if len(state) == 0:
            state['step'] = 0
            state['K'] = 0  # how many samples we have
            state['z'] = HalfQueue(maxN, self._params[0])
            state['v'] = HalfQueue(maxN, self._params[0])


        g_k = self._gather_flat_grad()   #tensor
        x_k = self._gather_flat_param()    #tensor  #theta(parameter)
        d = self._gather_flat_buf('momentum_buffer')  # d is d in pdf/return a tensor

        uk = g_k.dot(x_k)   # u_k is <x^k, g^k>
        vk = d.dot(d).mul(
            0.5 * group['lr'] * (1.0 + momentum) / (1.0 - momentum)) # v_k is a/2 * (1+b)/(1-b) E[<d,d>]

        state['z'].add(uk - vk)  # define z_k
        state['v'].add(vk)

        if closure is not None:
            u = uk.item()  # return one value
            v = vk.item()  
            z = u - v
            closure([u], [v], [z], [], [], [], [], [])
        
        
        self.lowercriteria = 0
        self.uppercriteria = 0
        self.tolerance = 0

        if state['K'] >= minN and state['K'] % group['testfreq'] == 0:
            u_equals_v, z_mean, z_upperbound, z_lowerbound, rhs, stds, dfs = test_onesamp(
                state['z'], state['v'], sigma, delta, mode=self.mode)
            self.lowercriteria = z_lowerbound
            self.uppercriteria = z_upperbound
            self.tolerance = rhs
            
            if closure is not None:
                closure([], [], [], [z_mean.item()], [z_upperbound.item()], [z_lowerbound.item()], [rhs.item()],
                        stds, dfs)
            if state['step'] > self.warmup and u_equals_v:

                group['lr'] = group['lr'] * zeta  # decrease lr by zeta 
                state['K'] = 0  # need to collect at least minN more samples.
                # should reset the queues here; bad if samples from before corrupt what you have now.
                state['z'].reset()
                state['v'].reset()
        elif self.logstats:
            if state['z'].n >= 4 and state['K'] % self.logstats == 0:
                u_equals_v, z_mean, z_upperbound, z_lowerbound, rhs, stds, dfs = test_onesamp(
                    state['z'], state['v'], sigma, delta, mode=self.mode,
                    verbose=False)
                if closure is not None:
                    closure([], [], [], [z_mean.item()], [z_upperbound.item()], [z_lowerbound.item()],
                            [rhs.item()], stds, dfs)

        state['K'] += 1

        for p in self._params:
            if p.grad is None:
                continue
            param_state = self.state[p]
            g_k = p.grad.data
            # get momentum buffer.
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                buf.mul_(momentum).add_(1.0 - momentum, g_k)
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(1.0 - momentum, g_k)

            # now do update.
            p.data.add_(-group['lr'], buf)

        state['step'] += 1

    def __init__(self, params, lr=required, weight_decay=0, momentum=0,
                 warmup=0, minN=100, maxN=1000, zeta=0.1, sigma=0.2, delta=0.02,
                 testfreq=500, onesamp=True, mode='bm', logstats=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, minN=minN, maxN=maxN,
                        zeta=zeta, momentum=momentum, sigma=sigma, delta=delta,
                        testfreq=testfreq)

        super(SASA, self).__init__(params, defaults)
        if onesamp:
            self.step_fn = self.step_onesamp
        else:
            self.step_fn = self.step_twosamp
        # self._params = self.param_groups[0]['params']
        self._params = []
        for param_group in self.param_groups:
            self._params += param_group['params']
        self.warmup = warmup  # todo: warmup in state?
        self.mode = mode  # using which variance estimator
        self.lowercriteria = 0
        self.uppercriteria = 0
        self.tolerance = 0
        print("using variance estimator: ", mode)
        self.logstats = logstats
        print("logging stats every {} steps".format(logstats))

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1) 
            else:
                view = p.grad.data.view(-1) 
            views.append(view)
        return torch.cat(views, 0) 


    def _gather_flat_buf(self, buf_name):
        views = []
        for p in self._params:
            param_state = self.state[p]
            if buf_name not in param_state:  # init buffer
                view = p.data.new(p.data.numel()).zero_()  
            else:
                view = param_state[buf_name].data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_param(self):
        views = []
        for p in self._params:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def step(self, closure=None):
        self.step_fn(closure=closure)


### Design the Mgnet Network
use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)

class MgIte(nn.Module): 
    def __init__(self, A, S):
        super().__init__()
        
        self.A = A
        self.S = S

        self.bn1 =nn.BatchNorm2d(A.weight.size(0)) 
        self.bn2 =nn.BatchNorm2d(S.weight.size(0)) 
    
    def forward(self, out):
        u, f = out 
        u = u + F.relu(self.bn2(self.S(F.relu(self.bn1((f-self.A(u))))))) 
        out = (u, f)
        return out



class MgRestriction(nn.Module):
    def __init__(self, A_old, A, Pi, R):
        super().__init__()

        self.A_old = A_old
        self.A = A
        self.Pi = Pi
        self.R = R

        self.bn1 = nn.BatchNorm2d(Pi.weight.size(0))   
        self.bn2 = nn.BatchNorm2d(R.weight.size(0))    

    def forward(self, out):
        u_old, f_old = out 
        u = F.relu(self.bn1(self.Pi(u_old)))                              
        f = F.relu(self.bn2(self.R(f_old-self.A_old(u_old)))) + self.A(u)        
        out = (u,f)
        return out


class MgNet(nn.Module):
    def __init__(self, num_channel_input, num_iteration, num_channel_u, num_channel_f, num_classes):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.conv1 = nn.Conv2d(num_channel_input, num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel_f)        

        
        A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3,stride=1, padding=1, bias=False)
        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            for i in range(num_iteration_l):
                layers.append(MgIte(A, S)) 
            setattr(self, 'layer'+str(l), nn.Sequential(*layers))

            if l < len(num_iteration)-1:
                A_old = A 
                
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3,stride=1, padding=1, bias=False)
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3,stride=1, padding=1, bias=False)

                Pi = nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3,stride=2, padding=1, bias=False)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False)
                
                
                layers= [MgRestriction(A_old, A, Pi, R)] 
        
        self.pooling = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(num_channel_u ,num_classes) 

    def forward(self, u, f):
        f = F.relu(self.bn1(self.conv1(f)))                
        if use_cuda:                                        
            u = torch.zeros(f.size(0),self.num_channel_u,f.size(2),f.size(3), device=torch.device('cuda')) 
        else:
            u = torch.zeros(f.size(0),self.num_channel_u,f.size(2),f.size(3))        
       
        
        out = (u, f) 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out)

        
        u, f = out       
        u = self.pooling(u) #do avg pooling
        u = u.view(u.shape[0], -1)  #reshape u batch_Size to vector
        u = self.fc(u)
        return u
    
    
    
    
    
    
    
### Implementation
minibatch_size = 128
num_epochs = 360
lr = 1
degree = 256
num_channel_input = 3 # since cifar10
num_channel_u = degree # usaually take channel u and f same, suggested value = 64,128,256,512.....
num_channel_f = degree
num_classes = 10
num_iteration = [2,2,2,2] # for each layer do 1 iteration or you can change to [2,2,2,2] or [2,1,1,1]

# Step 1: Define a model
my_model = MgNet(num_channel_input, num_iteration, num_channel_u, num_channel_f, num_classes)

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


optimizer = SASA(my_model.parameters(), lr=1, weight_decay=0.0001,momentum=0.9, testfreq=len(testloader))


train_accuracy_list = []
test_accuracy_list = []
lr_list = []
avg_loss_list = []

lowercriteria_list = []
uppercriteria_list = []
tolerance_list = []

start = timer()
#Step 4: Train the NNs
# One epoch is when an entire dataset is passed through the neural network only once.
for epoch in range(num_epochs):
    running_loss = 0
    my_model.train()
    for i, (images, labels) in enumerate(trainloader):
        if use_cuda:
          images = images.cuda()
          labels = labels.cuda()

        # Forward pass to get the loss
        outputs = my_model(0,images)   # We need additional 0 input for u in MgNet
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
          outputs = my_model(0,images) 
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
          outputs = my_model(0,images)      # We need additional 0 input for u in MgNet
          p_max, predicted = torch.max(outputs, 1) 
          total += labels.size(0)
          correct += (predicted == labels).sum()
    test_accuracy = float(correct)/total
    test_accuracy_list.append(test_accuracy)
    current_lr = optimizer.state['lr']
    lr_list.append(current_lr)
    lowercriteria_list.append(optimizer.lowercriteria)
    uppercriteria_list.append(optimizer.uppercriteria)
    tolerance_list.append(optimizer.tolerance)
   
end = timer()
print("total computational time is", end - start)
file = open("SASAmgnet.txt","x")
file.write("train_accuracy_list = {}\n".format(str(train_accuracy_list)))
file.write("test_accuracy_list = {}\n".format(str(test_accuracy_list)))
file.write("lr_list = {}\n".format(str(lr_list)))
file.write("lowercriteria_list = {}\n".format(str(lowercriteria_list)))
file.write("uppercriteria_list = {}\n".format(str(uppercriteria_list)))
file.write("tolerance_list = {}\n".format(str(tolerance_list)))
file.write("avg_loss_list = {}\n".format(str(avg_loss_list)))
file.write("total_time = {}\n".format(str(end - start)))
file.close()
print("complete")
