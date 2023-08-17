import torch
import torch.nn as nn
from torchvision import datasets, models


class GuidedBackprop:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.weights = self._get_model(model_name)
        self.transforms = self._get_transforms()
        self.input_grad = None
        self.activation_maps = []
        self.activation_maps_dict = None 
        self.layer_names = []
        self.out = None
        self.model_hooks()
        
    def _get_model(self, model_name):
        weight_name = model_name.upper() + "_Weights"
        weights = getattr(models, weight_name).DEFAULT
        model = getattr(models, model_name.lower())(weights=weights)
        return model, weights

    def _get_transforms(self):
        transforms = self.weights.transform()
        return transforms
 
    def model_hooks(self):
        # Forward pass, get activation maps
        def first_layer_hook_fn(m, g_in, g_out):
            self.input_grad = g_in[0]

        def fwd_hook_fn(m, i, o):
            self.activation_maps.append(o) 

        def back_hook_fn(m, g_in, g_out):
            activation = self.activation_maps.pop()
            act_mask = activation > 0
            grad_positive = g_out[0] > 0

            # Mask the gradients with the activation
            grad = g_out[0] * act_mask * grad_positive
            return (grad,)

        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.ReLU):
                self.layer_names.append(f'conv_layer{relu_count}')
                layer.register_forward_hook(fwd_hook_fn)
                layer.register_backward_hook(back_hook_fn)
                relu_count += 1
        
        self.model.features[0].register_backward_hook(first_layer_hook_fn)

    def predict(self, input_image):
        # Apply transforms
        input_image = self.transforms(input_image).unsqueeze(0)

        # Perform prediction
        self.model.eval()
        self.out = self.model(input_image.requires_grad_())
        prediction = torch.nn.functional.softmax(self.out[0], dim=0)
        confidences = {self.weights.meta["categories"][i]: float(prediction[i]) for i in range(1000)}
        
        # Create a dictionary of activation maps
        self.activation_maps_dict = dict(zip(self.layer_names, self.activation_maps))
    
        return confidences
    
    def backprop(self):
        # Backward pass
        grad_target_map = torch.zeros(self.out.shape, dtype=torch.float)
        grad_target_map[0][self.predictions.argmax().item()] = 1
        self.out.backward(grad_target_map)

        return self.input_grad, self.activation_maps_dict

