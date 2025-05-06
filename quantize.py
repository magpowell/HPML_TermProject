import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://medium.com/@sayedebad.777/the-power-of-quantization-in-ml-a-pytorch-tutorial-part-4-ae521e8a00ae
class W8A16LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantized_dtype=torch.int8, dtype=torch.float32):
        super().__init__()
        
        # use register_buffer() to allow initialization with desired datatype
        # bypasses computing gradients Pytorch issues
        self.register_buffer(
            "quantized_weights",
            torch.randint(-self._int_max(quantized_dtype)-1, self._int_max(quantized_dtype), (out_features, in_features), dtype=quantized_dtype))

        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))

        self.quantized_dtype = quantized_dtype

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        # scale based on the max value and force between 2^(n-1) - 1 and 2^(n-1)
        scales = w_fp32.abs().max(dim=-1).values / self._int_max(self.quantized_dtype)
        scales = scales.to(weights.dtype)

        # quantize weights with known scales
        quantized_weights = torch.round(weights/scales.unsqueeze(1)).to(self.quantized_dtype)

        self.quantized_weights = quantized_weights
        self.scales = scales            
        self.bias = None

    def _int_max(self, qdtype):
        if qdtype == torch.int16:
            return 32767
        elif qdtype == torch.int8:
            return 127
        else:
            raise NotImplementedError("Currently only supporting torch.int16, int8")

    def forward(self, input):
        return w8_a16_forward(self.quantized_weights, input, self.scales, self.bias)

# forward pass from linear layer in quantized form
def w8_a16_forward(weight, input, scales, bias=None):
    
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales
    
    if bias is not None:
        output = output + bias
      
    return output

# replace all of the linear layers in the model (nn.Linear) with our quantized linear layer
def replace_linear_with_target_and_quantize(module, target_class, target_qdtype):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear): 
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(child.in_features, child.out_features, old_bias is not None, target_qdtype, child.weight.dtype)
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, target_class, target_qdtype)

def model_size(model):
    # add up the total size of all the parameters and buffers in the model
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size, buffer_size
