import torch
import torch.nn as nn
from thop import profile, clever_format

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(input):
    if isinstance(input, torch.Tensor):
        return input.to(DEVICE)
    elif isinstance(input, list):
        return [to_device(v) for v in input]
    elif isinstance(input, tuple):
        return tuple(to_device(v) for v in input)
    elif isinstance(input, dict):
        return {k: to_device(v) for k, v in input.items()}
    elif isinstance(input, nn.Module):
        return input.to(DEVICE)
    else:
        return input


def process_data(example_input):
    if isinstance(example_input, torch.Tensor):
        example_input = [
            to_device(example_input),
        ]
    elif isinstance(example_input, dict):
        example_input = [to_device(v) for v in example_input.values()]
    elif isinstance(example_input, (list, tuple)):
        example_input = [to_device(v) for v in example_input]
    return example_input


def cal_flops(model, example_input):
    model = to_device(model)
    example_input = process_data(example_input)
    flops, params = profile(model, inputs=example_input, verbose=False)
    return flops, params, clever_format([flops, params], "%.3f")