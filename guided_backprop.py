#%%
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request


# Get pretrained VGG19
weights = models.VGG19_Weights.DEFAULT
vgg = models.vgg19(weights=weights)


#%% Download Image
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/American_Eskimo_Dog_1.jpg/1200px-American_Eskimo_Dog_1.jpg'
urllib.request.urlretrieve(url, 'test2.jpg')

input_img = Image.open('test2.jpg')

# Show image
plt.imshow(input_img)
plt.show()

#%%
class Guided_Backprop():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.visualization = []
        
activations = []
grad_vis = []

preprocess = weights.transforms()
input_tensor = preprocess(input_img).unsqueeze(0)

def fwd_hook_fn(m, i, o):
    activations.append(o) 

def back_hook_fn(m, g_in, g_out):
    activation = activations.pop()
    # activation = activation.clamp(min=0)
    act_mask = activation > 0
    
    grad_positive = g_out[0] > 0

    # Mask the gradients with the activation
    grad = g_out[0] * act_mask * grad_positive
    grad_vis.append(grad)

for name, layer in vgg.named_modules():
  if isinstance(layer, nn.ReLU):

    layer.register_forward_hook(fwd_hook_fn)
    layer.register_backward_hook(back_hook_fn)
  
out = vgg(input_tensor)
grad_target_map = torch.zeros(out.shape, dtype=torch.float)
grad_target_map[0][out.argmax().item()] = 1
out.backward(grad_target_map)


#%% Visualize activations
plt.imshow(activations[10][0][15].detach())
plt.show()


#%% Visualize gradients
plt.imshow(grad_vis[10][0][0].detach())
plt.show()
