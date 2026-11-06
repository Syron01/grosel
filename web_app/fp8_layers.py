"""FP8 Linear katmanlari.

FP8Linear       : agirlik FP8, hesaplama BF16 (her forward'da dequant). Guvenli, kanitlanmis.
FP8NativeLinear : agirlik FP8, matmul dogrudan FP8 tensor cekirdeginde (torch._scaled_mm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_fp8", torch.empty((out_features, in_features), dtype=FP8_DTYPE, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear):
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        layer.weight_fp8.copy_(linear.weight.detach().to(FP8_DTYPE))
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.detach().to(torch.bfloat16))
        return layer

    def forward(self, x):
        return F.linear(x, self.weight_fp8.to(x.dtype), self.bias)


class FP8NativeLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_fp8", torch.empty((out_features, in_features), dtype=FP8_DTYPE, device=device))
        self.register_buffer("scale_w", torch.ones((), dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)
        self._mm_ok = (in_features % 16 == 0) and (out_features % 16 == 0)

    @classmethod
    def from_linear(cls, linear):
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        w = linear.weight.detach().to(torch.float32)
        scale = (w.abs().amax() / FP8_MAX).clamp_min(1e-8)
        layer.weight_fp8.copy_((w / scale).to(FP8_DTYPE))
        layer.scale_w.copy_(scale)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.detach().to(torch.bfloat16))
        return layer

    def forward(self, x):
        if not self._mm_ok or x.dtype != torch.bfloat16:
            return F.linear(x, self.weight_fp8.to(x.dtype) * self.scale_w.to(x.dtype), self.bias)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        sa = (x2.abs().amax() / FP8_MAX).clamp_min(1e-8).to(torch.float32)
        xq = (x2 / sa).to(FP8_DTYPE)
        try:
            out = torch._scaled_mm(xq, self.weight_fp8.t(), scale_a=sa, scale_b=self.scale_w,
                                   bias=self.bias, out_dtype=torch.bfloat16)
        except Exception:
            self._mm_ok = False
            return F.linear(x, self.weight_fp8.to(x.dtype) * self.scale_w.to(x.dtype), self.bias)
        return out.view(*shape[:-1], self.out_features)


def _convert(module, factory):
    count = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            setattr(module, name, factory(child))
            count += 1
        elif isinstance(child, nn.Module):
            count += _convert(child, factory)
    return count


def convert_to_fp8(model):
    return _convert(model, FP8Linear.from_linear)


def convert_to_fp8_native(model):
    return _convert(model, FP8NativeLinear.from_linear)
