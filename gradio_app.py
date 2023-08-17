import gradio as gr
import requests
from PIL import Image
from torchvision import transforms

import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from PIL import Image
import urllib.request
from matplotlib import cm

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

def range_norm(img):
    min = img.min()
    max = img.max()
    return (img - min)/(max-min+1e-6)


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
    layer_names.append(name+'.'+f'layer{relu_count}')
    layer.register_forward_hook(fwd_hook_fn)
    layer.register_backward_hook(back_hook_fn)
    relu_count += 1

vgg.features[0].register_backward_hook(first_layer_hook_fn)

out = vgg(input_tensor.requires_grad_())
activation_maps = activations.copy()
activation_maps_dict = dict([('conv_'+layer_name.split('.')[-1], activation_maps[i])for i, layer_name in enumerate(layer_names) if 'classifier' not in layer_name])

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
   return gr.update(minimum=1, maximum=activation_maps_dict[layer_name].shape[1], value=0)


def return_act_map(layer_name, chosen_filter, color):
    act_im = activation_maps_dict[layer_name][0][chosen_filter-1].detach().cpu().numpy()
    act_im = range_norm(act_im)
    if color != "gray":
        act_im = cm.jet(act_im)[:,:,:-1]
    return Image.fromarray(np.uint8(act_im*255))


def compute_input_grad():
    input_grad = image_reconstruction[0].permute(1,2,0).detach()
    input_grad = range_norm(input_grad)
    input_grad = Image.fromarray(np.uint8(input_grad*255))
    return input_grad


with gr.Blocks() as demo:
    # Inference
    with gr.Box():
        gr.Markdown("## Run inference")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Input Image")
                # TODO: Add a button to clear results and input image
                
                with gr.Row():
                    with gr.Column(scale=1):
                        reset_button = gr.ClearButton(value="Reset", elem_classes="feedback")
                    with gr.Column(scale=1):
                        submit_button = gr.Button(value="Compute")

                example = gr.Examples(examples=["test.jpg"], inputs=image)
            with gr.Column(scale=1):
                probs = gr.Label(label="Class Probs",num_top_classes=3)

    # Visualize activation maps
    with gr.Box():
        gr.Markdown("## Visualize activation maps")
        with gr.Row():
            with gr.Column(scale=1):
                chosen_layer = gr.Dropdown(choices=activation_maps_dict.keys(), label="Layer", info="Choose from the layers", interactive=True)
                chosen_filter = gr.Slider(label="Filter", info="Choose from the filters", interactive=True)
                color = gr.Radio(["heatmap", "gray"], value="heatmap", label="Color", info="Choose the color of the activation map")
            
            with gr.Column(scale=1):
                activation_map = gr.Image(type="pil", label="Activation Map", height=300)

    # Visualize input gradient
    with gr.Box():
        gr.Markdown("## Visualize input gradient")
        with gr.Row():
            with gr.Column(scale=1):
                image2 = gr.Image(type="pil", label="Input Image", height=300)
                compute_grad_btn = gr.Button(value="Compute")
                

            with gr.Column(scale=1):
                input_grad = gr.Image(type="pil", label="Input gradient", height=300)

    # Set up callbacks
    reset_button.add([image, image2, probs, activation_map, input_grad])
    submit_button.click(predict, inputs=image, outputs=probs)
    submit_button.click(predict2, inputs=image, outputs=[image, image2])
    chosen_layer.change(return_total_layer_num, inputs=chosen_layer, outputs=chosen_filter)
    chosen_filter.change(return_act_map, inputs=[chosen_layer, chosen_filter, color], outputs=activation_map)
    color.change(return_act_map, inputs=[chosen_layer, chosen_filter, color], outputs=activation_map)
    compute_grad_btn.click(compute_input_grad, outputs=input_grad)

demo.launch()
