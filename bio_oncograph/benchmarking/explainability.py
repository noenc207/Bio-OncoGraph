import torch
import torch.nn.functional as F
import numpy as np

class GradCAMEvaluator:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float, device=input_tensor.device)
        one_hot_output[0][target_class] = 1
        
        output.backward(gradient=one_hot_output)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

class IntegratedGradientsEvaluator:
    def __init__(self, model):
        self.model = model
        
    def generate(self, input_tensor, target_class=None, steps=50):
        self.model.eval()
        baseline = torch.zeros_like(input_tensor)
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(0, steps + 1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad = True
        
        outputs = self.model(scaled_inputs)
        loss = outputs[:, target_class].sum()
        loss.backward()
        
        grads = scaled_inputs.grad
        avg_grads = torch.mean(grads[:-1], dim=0, keepdim=True)
        integrated_gradients = (input_tensor - baseline) * avg_grads
        
        return integrated_gradients.squeeze().cpu().detach().numpy()

class AttentionMapEvaluator:
    def __init__(self, model, attention_layer_name="attention"):
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.attention_weights = None
        
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name:
                module.register_forward_hook(self.save_attention)
                break
                
    def save_attention(self, module, input, output):
        if isinstance(output, tuple):
            self.attention_weights = output[1]
        else:
            self.attention_weights = output
            
    def generate(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if self.attention_weights is not None:
            return self.attention_weights.cpu().numpy()
        return None
