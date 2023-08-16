import gradio as gr
import requests
from PIL import Image
from torchvision import transforms

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
    activation = activations.pop()
    # activation = activation.clamp(min=0)
    act_mask = activation > 0
    grad_positive = g_out[0] > 0

    # Mask the gradients with the activation
    grad = g_out[0] * act_mask * grad_positive
    grad_vis.append(grad)

    return (grad,)

# Get all vgg layer names
layer_names = []
relu_count = 0

for name, layer in vgg.named_modules():
  if isinstance(layer, nn.ReLU):
    layer_names.append(f'conv_layer{relu_count}')
    layer.register_forward_hook(fwd_hook_fn)
    layer.register_backward_hook(back_hook_fn)
    relu_count += 1

vgg.features[0].register_backward_hook(first_layer_hook_fn)

out = vgg(input_tensor.requires_grad_())
activation_maps = activations.copy()
activation_maps_dict = dict(zip(layer_names, activation_maps))
grad_target_map = torch.zeros(out.shape, dtype=torch.float)
grad_target_map[0][out.argmax().item()] = 1
out.backward(grad_target_map)


def predict(inp):
  inp = preprocess(inp).unsqueeze(0)
  vgg.eval()
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(vgg(inp)[0], dim=0)
    confidences = {weights.meta["categories"][i]: float(prediction[i]) for i in range(1000)}
  return confidences


def return_total_layer_num(layer_name):
   return activation_maps_dict[layer_name].shape[1]


def return_act_map(layer_name):
    act_im = activation_maps_dict[layer_name][0][10].detach().cpu().numpy()
    return act_im/(act_im.max()+1e-8)


# Get max channel number across all layers
max_channel_num = 1
for layer_name in layer_names:
    if activation_maps_dict[layer_name].shape[1] > max_channel_num:
        max_channel_num = activation_maps_dict[layer_name].shape[1]

with gr.Blocks() as demo:
    layers = None
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Input Image")
            submit_button = gr.Button(value="Submit")
            example = gr.Examples(examples=["test.jpg"], inputs=image)
        with gr.Column(scale=1):
            probs = gr.Label(label="Class Probs",num_top_classes=3)
    with gr.Row():
        with gr.Column(scale=1):
            chosen_layer = gr.Dropdown(choices=layer_names)
            chosen_filter = gr.Slider(0, max_channel_num, value=0, label="Count", info="Choose from the filters"),
        
        with gr.Column(scale=1):
            activation_map = gr.Image(type="numpy", label="Activation Map")

    submit_button.click(predict, inputs=image, outputs=probs)

demo.launch()
