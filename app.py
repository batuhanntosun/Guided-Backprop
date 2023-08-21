import gradio as gr
import numpy as np
from PIL import Image
from matplotlib import cm

from guided_backprop import GuidedBackprop
from utils import range_norm


class GradioApp:
    def __init__(self):
        self.gp = None
        self.probs = None
        self.input_grad = None
        self.activation_maps = None

        # Define GUI elements
        with gr.Blocks() as self.app:
            # Inference
            with gr.Box():
                gr.Markdown("## Run inference")
                with gr.Row():
                    with gr.Column(scale=1):
                        image = gr.Image(type="pil", label="Input Image", height=200)
                        with gr.Row():
                            with gr.Column(scale=1):
                                reset_button = gr.ClearButton(value="Reset", elem_classes="feedback")
                            with gr.Column(scale=1):
                                run_button = gr.Button(value="Run")

                        example = gr.Examples(
                            examples=[
                                "images/bird.jpg", 
                                "images/lion.jpg",
                                "images/tiger.jpg",
                                "images/pomegranate.jpg",
                                "images/strawberry.jpg"
                            ], 
                            inputs=image)
                    with gr.Column(scale=1):
                        model_name = gr.Dropdown(choices=["VGG19", "AlexNet"], label="Model", info="Choose from the models", interactive=True)
                        probs = gr.Label(label="Class Probs", num_top_classes=3)

            # Visualize activation maps
            with gr.Box():
                gr.Markdown("## Visualize activation maps")
                with gr.Row():
                    with gr.Column(scale=1):
                        chosen_layer = gr.Dropdown(label="Layer", info="Choose from the layers", interactive=True)
                        chosen_filter = gr.Slider(label="Filter", info="Choose from the filters", interactive=True)
                        color = gr.Radio(["heatmap", "gray"], value="heatmap", label="Color", info="Choose the color of the activation map")
                    
                    with gr.Column(scale=1):
                        activation_map = gr.Image(type="pil", label="Activation Map", height=300)

            # Visualize input gradient
            with gr.Box():
                gr.Markdown("## Visualize input gradient")
                with gr.Row():
                    with gr.Column(scale=1):
                        image_ = gr.Image(type="pil", label="Input Image", height=300)
                        compute_grad_btn = gr.Button(value="Compute")

                    with gr.Column(scale=1):
                        input_grad = gr.Image(type="pil", label="Input gradient", height=300)

            # Set up callbacks
            run_button.click(
                self.run, 
                inputs=[model_name, image], 
                outputs=[image, image_, probs, chosen_layer])
            
            reset_button.add([image, image_, probs, activation_map, input_grad])
            
            chosen_layer.change(
                self.update_filter_num, 
                inputs=chosen_layer, outputs=chosen_filter)
            
            chosen_filter.change(
                self.get_activation_map, 
                inputs=[chosen_layer, chosen_filter, color], 
                outputs=activation_map)
            
            color.change(
                self.get_activation_map, 
                inputs=[chosen_layer, chosen_filter, color], 
                outputs=activation_map)
            
            compute_grad_btn.click(
                self.get_input_grad, 
                outputs=input_grad)

    def run(self, model_name, input_image):
        self.gp = GuidedBackprop(model_name)
        self.probs = self.gp.predict(input_image)
        self.input_grad, self.activation_maps = self.gp.backprop()
        layers = gr.update(choices=list(self.activation_maps.keys()))
        return input_image, input_image, self.probs, layers

    def update_filter_num(self, layer_name):
        return gr.update(minimum=1, maximum=self.activation_maps[layer_name].shape[1], value=0)
    
    def get_activation_map(self, layer_name, chosen_filter, color):
        act_im = self.activation_maps[layer_name][0][chosen_filter-1].detach().cpu().numpy()
        act_im = range_norm(act_im)
        if color != "gray":
            act_im = cm.jet(act_im)[:,:,:-1]
        return Image.fromarray(np.uint8(act_im*255))
    
    def get_input_grad(self):
        input_grad = self.input_grad[0].permute(1,2,0).detach()
        input_grad = range_norm(input_grad)
        input_grad = Image.fromarray(np.uint8(input_grad*255))
        return input_grad

    def launch(self):
        self.app.launch()


if __name__ == "__main__":
    gradio_app = GradioApp()
    gradio_app.launch()