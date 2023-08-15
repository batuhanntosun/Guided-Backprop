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
urllib.request.urlretrieve(url, 'test.jpg')

input_img = Image.open('test.jpg')

# Show image
plt.imshow(input_img)
plt.show()

        
activations = []
grad_vis = []
image_reconstruction = None

preprocess = weights.transforms()
input_tensor = preprocess(input_img).unsqueeze(0)


def first_layer_hook_fn(m, g_in, g_out):
    global image_reconstruction
    image_reconstruction = g_in[0]
    
def fwd_hook_fn(m, i, o):
    activations.append(o) 

def back_hook_fn(m, g_in, g_out):
    print(g_out[0])
    activation = activations.pop()
    # activation = activation.clamp(min=0)
    act_mask = activation > 0
    grad_positive = g_out[0] > 0

    # Mask the gradients with the activation
    grad = g_out[0] * act_mask * grad_positive
    grad_vis.append(grad)

    return (grad,)

for name, layer in vgg.named_modules():
  if isinstance(layer, nn.ReLU):

    layer.register_forward_hook(fwd_hook_fn)
    layer.register_backward_hook(back_hook_fn)


vgg.features[0].register_backward_hook(first_layer_hook_fn)

out = vgg(input_tensor.requires_grad_())
activation_maps = activations.copy()

grad_target_map = torch.zeros(out.shape, dtype=torch.float)
grad_target_map[0][out.argmax().item()] = 1
out.backward(grad_target_map)


#%% Visualize activations
plt.imshow(activation_maps[20][0][10].detach())
plt.show()


#%% Visualize input gradient
plt.imshow(image_reconstruction[0].permute(1,2,0).detach())
plt.show()
