import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class ViTModel(nn.Module):
    def __init__(self, vit_model_type = 'dino_vitb16'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', vit_model_type)
        self.model.head = nn.Identity()

    def forward(self, x):
        feats = []
        
        def hook(module, input, output):
            feats.append(output)
        
        id = self.model.blocks[-1].register_forward_hook(hook)
        _ = self.model(x)
        id.remove()

        return feats[-1]
    

class Decoder(nn.Module):
    def __init__(self,in_channels,output_resolution=224, out_channels = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 2),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, 2),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, 2),
            nn.ConvTranspose2d(in_channels // 8, in_channels // 16, 3, 2),
            nn.ConvTranspose2d(in_channels // 16, 32, 3, 2),
            nn.ConvTranspose2d(32, 3, 3, 2),
            nn.Upsample(size=(output_resolution,output_resolution), align_corners=True, mode='bilinear')
        )

    def forward(self, x):
        return self.decoder(x)
    
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd).apply(x)

class Discriminator(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.out_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(output_dim, output_dim//2),
            nn.Linear(output_dim//2, output_dim//4),
            nn.Linear(output_dim//4, output_dim//8),
            nn.Linear(output_dim//8,output_dim//4),
            nn.Linear(output_dim//4,output_dim//2),
            nn.Linear(output_dim//2,output_dim)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse = False):
        if reverse:
            x = grad_reverse(x, self.lambd)
            out = self.network(x)
        else:
            out = self.network(x)
            
        return out
    
# Discriminator(768, 40)(torch.randn(2,768)).shape

# Discriminator(40)(torch.rand(14,40)).shape
