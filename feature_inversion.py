from __future__ import print_function
import matplotlib.pyplot as plt

import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from models import *

import torch
import torch.optim

from utils.feature_inversion_utils import *
from utils.perceptual_loss.perceptual_loss import get_pretrained_net
from utils.common_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
fname = r'C:\Users\user\Desktop\code\deep-image-prior-master\data\feature_inversion/building.jpg'

pretrained_net = 'alexnet_caffe' # 'vgg19_caffe'
layers_to_use = 'fc6' # comma-separated string of layer names e.g. 'fc6,fc7'

cnn = get_pretrained_net(pretrained_net).type(dtype)

opt_content = {'layers': layers_to_use, 'what':'features'}

# Remove the layers we don't need 
keys = [x for x in cnn._modules.keys()]
max_idx = max(keys.index(x) for x in opt_content['layers'].split(','))
for k in keys[max_idx+1:]:
    cnn._modules.pop(k)
    
print(cnn)

# Target imsize 
imsize = 227 if pretrained_net == 'alexnet' else 224

# Something divisible by a power of two
imsize_net = 256

# VGG and Alexnet need input to be correctly normalized
preprocess, deprocess = get_preprocessor(imsize), get_deprocessor()

img_content_pil, img_content_np  = get_image(fname, imsize)
img_content_prerocessed = preprocess(img_content_pil)[None,:].type(dtype)

img_content_pil

matcher_content = get_matcher(cnn, opt_content)

matcher_content.mode = 'store'
cnn(img_content_prerocessed);

INPUT = 'noise'
pad = 'zero' # 'refection'
OPT_OVER = 'net' #'net,input'
OPTIMIZER = 'adam' # 'LBFGS'
LR = 0.001

num_iter = 3100

input_depth = 32
net_input = get_noise(input_depth, INPUT, imsize_net).type(dtype).detach()

net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [4, 4, 4, 4, 4, 4],   
                           filter_size_down = [7, 7, 5, 5, 3, 3], filter_size_up = [7, 7, 5, 5, 3, 3], 
                           upsample_mode='nearest', downsample_mode='avg',
                           need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(dtype)

# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)
def closure():
    global i      
    out = net(net_input)[:, :, :imsize, :imsize]
    cnn(vgg_preprocess_var(out))
    total_loss =  sum(matcher_content.losses.values())
    total_loss.backward()
    print('flag12')
    print ('Iteration %05d    Loss %.3f' % (i, total_loss.item()), '\r', end='')
    if PLOT and i % 200 == 0:
        out_np = np.clip(torch_to_np(out), 0, 1)
        plot_image_grid([out_np], 3, 3);
    i += 1
    return total_loss

matcher_content.mode = 'match'
p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out = net(net_input)[:, :, :imsize, :imsize]
plot_image_grid([torch_to_np(out)], 3, 3);
print('plot finish')