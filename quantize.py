import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://medium.com/@sayedebad.777/the-power-of-quantization-in-ml-a-pytorch-tutorial-part-4-ae521e8a00ae

# this is a quantized linear layer where the weights are set to int8 datatype
class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()
        
        # use register_buffer() to allow initialization with desired datatype
        # bypasses computing gradients Pytorch issues
        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))

        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        # scale based on the max value and force between -128 and 127
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        # quantize weights with known scales
        int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales            
        self.bias = None

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)

# forward pass from linear layer in quantized form
def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
      
    return output

# replace all of the linear layers in the model (nn.Linear) with our quantized linear layer
def replace_linear_with_target_and_quantize(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
          any([x == name for x in module_name_to_exclude]):
            print(name)
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)

