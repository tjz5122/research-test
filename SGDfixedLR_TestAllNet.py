import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from timeit import default_timer as timer
import torch.optim as optim
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

use_cuda = torch.cuda.is_available()
print('Use GPU?', use_cuda)


###Mgnet
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
    
    
###pre-act resnet
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])



### Implementation

# cifar 10
num_channel_input = 3
num_classes = 10
normalizedmean = (0.4914, 0.4822, 0.4465)
normalizedstd = (0.2023, 0.1994, 0.2010)

#training hyperparameter
num_epochs = 120
lr = 1
num_iteration = [2,2,2,2] # for each layer do 1 iteration or you can change to [2,2,2,2] or [2,1,1,1]
minibatch_size = 128
wd = 0.0005 
momentum = 0.6
 

# Step 1: Define a model
mgnet128 = MgNet(num_channel_input, num_iteration, 128, 128, num_classes)
mgnet256 = MgNet(num_channel_input, num_iteration, 256, 256, num_classes)
resnet18 = torchvision.models.resnet18(num_classes=num_classes)
resnet34 = torchvision.models.resnet34(num_classes=num_classes)
preactresnet18 = PreActResNet18()
preactresnet34 = PreActResNet34()
densenet121 = models.densenet121()
densenet161 = models.densenet161()
efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

modeldic  = {"mgnet128":mgnet128, 
             "mgnet256":mgnet256,
             "resnet18":resnet18, 
             "resnet34":resnet34, 
             "preactresnet18":preactresnet18, 
             "preactresnet34": preactresnet34,
             "densenet121":densenet121,
             "densenet161":densenet161,
             "efficientnet":efficientnet}

if use_cuda:
    for i in modeldic:
        modeldic[i] = modeldic[i].cuda()

# Step 2: Define a loss function and training algorithm
criterion = nn.CrossEntropyLoss()

# Step 3: load dataset
normalize = torchvision.transforms.Normalize(mean=normalizedmean, std=normalizedstd)
transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.ToTensor(),
                                                  normalize])
transform_test  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize])
# cifar 10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)
# cifar 10
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=False)



#Step 4: Train the NNs
# One epoch is when an entire dataset is passed through the neural network only once. 
f = open("SGDfixedLR_TestAllNet", 'w')

for my_model in modeldic:
    
    test_accuracy_list = []
    lr_list = []
    statistic_list = []
    avg_loss_list = []
    max_test_accuarcy = 0
    best_parameter = 0
    peak_epoch = 0
    
    optimizer = optim.SGD(modeldic[my_model].parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    start = timer()
    for epoch in range(num_epochs):

        running_loss = 0
        modeldic[my_model].train()
        for i, (images, labels) in enumerate(trainloader):
            if use_cuda:
              images = images.cuda()
              labels = labels.cuda()
    
            # Forward pass to get the loss
            if my_model == "mgnet128" or my_model == "mgnet256":
                outputs = modeldic[my_model](0,images)   # We need additional 0 input for u in MgNet
            else:
                outputs = modeldic[my_model](images) 
            loss = criterion(outputs, labels)
            # Backward and compute the gradient
            optimizer.zero_grad()
            loss.backward()  #backpropragation
            running_loss += loss.item()
            optimizer.step() #update the weights/parameters
        
      # Training accuracy
        modeldic[my_model].eval()
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(trainloader):
            with torch.no_grad():
              if use_cuda:
                  images = images.cuda()
                  labels = labels.cuda()  
              if my_model == "mgnet128" or my_model == "mgnet256":
                  outputs = modeldic[my_model](0,images)   # We need additional 0 input for u in MgNet
              else:
                  outputs = modeldic[my_model](images) 
              p_max, predicted = torch.max(outputs, 1) 
              total += labels.size(0)
              correct += (predicted == labels).sum()
        training_accuracy = float(correct)/total
        
        # Test accuracy
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(testloader):
            with torch.no_grad():
              if use_cuda:
                  images = images.cuda()
                  labels = labels.cuda()
              if my_model == "mgnet128" or my_model == "mgnet256":
                  outputs = modeldic[my_model](0,images)   # We need additional 0 input for u in MgNet
              else:
                  outputs = modeldic[my_model](images) 
              p_max, predicted = torch.max(outputs, 1) 
              total += labels.size(0)
              correct += (predicted == labels).sum()
              
        test_accuracy = float(correct)/total
        current_lr = lr
        test_accuracy_list.append(test_accuracy)
        lr_list.append(current_lr)
        avg_loss_list.append(running_loss)
        
        # update parameter
        if test_accuracy > max_test_accuarcy:
            max_test_accuarcy = test_accuracy
            best_parameter = optimizer.param_groups[0]['params']
            peak_epoch = epoch
    
    
    end = timer()
    time = end - start
    
    f.write("SGDfixedlr_"+ my_model +"\n")
    f.write("SGDfixedlr_"+ my_model +"_testacculist = {}\n".format(test_accuracy_list))
    f.write("SGDfixedlr_"+ my_model +"_lrlist = {}\n".format(lr_list))
    f.write("SGDfixedlr_"+ my_model +"_losslist = {}\n".format(avg_loss_list))
    f.write("SGDfixedlr_"+ my_model +"_time = {}\n".format(time))
    f.write("SGDfixedlr_"+ my_model +"_maxtestaccu = {}\n".format(max_test_accuarcy))
    f.write("SGDfixedlr_"+ my_model +"_peakepoch = {}\n".format(peak_epoch))
    f.write("SGDfixedlr_"+ my_model +"_bestparam = {}\n".format(best_parameter))
    f.write("\n")

f.close()
