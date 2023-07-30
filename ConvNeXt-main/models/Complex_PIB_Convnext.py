import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import _calculate_fan_in_and_fan_out
from complexLayers import NaiveComplexBatchNorm2d
from timm.models.layers import trunc_normal_, DropPath

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, input):
        real = self.fc(input.real)
        imag = self.fc(input.imag)
        return torch.complex(real,imag)

# class NaiveComplexLayerNorm(nn.Module):
#     '''
#     Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
#     '''
#
#     def __init__(self, num_features, eps=1e-5, data_format="channels_last"):
#         super(NaiveComplexLayerNorm, self).__init__()
#         self.bn_r = LayerNorm(num_features, eps, data_format=data_format)
#         self.bn_i = LayerNorm(num_features, eps, data_format=data_format)
#
#     def forward(self, input):
#         return self.bn_r(input.real).type(torch.complex64) + 1j * self.bn_i(input.imag).type(torch.complex64)


# class ComplexLayerNorm(nn.Module):
#     '''
#     Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
#     '''
#
#     def __init__(self, num_features, eps=1e-5, data_format="channels_last"):
#         super(ComplexLayerNorm, self).__init__()
#         self.bn_r = nn.LayerNorm(num_features, eps)
#         self.bn_i = nn.LayerNorm(num_features, eps)
#
#     def forward(self, input):
#         return self.bn_r(input.real).type(torch.complex64) + 1j * self.bn_i(input.imag).type(torch.complex64)


# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     # work with diff dim tensors, not just 2D ConvNets
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor


class ComplexDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(ComplexDropPath, self).__init__()
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        return torch.complex(self.drop_path(x.real),
                             self.drop_path(x.imag))


class ComplexGeLU(nn.Module):
    def __init__(self, alpha=0.0, max_value=None, threshold=0, inplace=True):
        super(ComplexGeLU, self).__init__()
        self.alpha = alpha
        self.max_value = max_value
        self.threshold = threshold
        self.inplace = inplace

    def forward(self, z):
        real = torch.real(z)
        imag = torch.imag(z)

        real_gelu = F.gelu(real - self.threshold) + self.threshold
        imag_gelu = F.gelu(imag - self.threshold) + self.threshold

        if self.alpha != 0:
            real_gelu[real < self.threshold] = self.alpha * (real[real < self.threshold] - self.threshold)
            imag_gelu[imag < self.threshold] = self.alpha * (imag[imag < self.threshold] - self.threshold)

        if self.max_value is not None:
            real_gelu[real_gelu > self.max_value] = self.max_value
            imag_gelu[imag_gelu > self.max_value] = self.max_value

        z_relu = torch.complex(real_gelu, imag_gelu)
        return z_relu


class ComplexAdaptiveAvgPooling2D(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveAvgPooling2D, self).__init__()
        self.output_size = output_size

    def forward(self, inputs):
        inputs_r = torch.real(inputs)
        inputs_i = torch.imag(inputs)
        output_r = F.adaptive_avg_pool2d(input=inputs_r, output_size=self.output_size)
        output_i = F.adaptive_avg_pool2d(input=inputs_i, output_size=self.output_size)
        output = torch.complex(output_r, output_i)
        return output
class One_dimensional_complex_conv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1):

        super(One_dimensional_complex_conv, self).__init__()
        self.kernel = kernel_size
        if kernel_size != 1:
            self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=stride,
                                    padding=(0, padding),
                                    groups=groups)
            self.conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=stride,
                                    padding=(padding, 0),
                                    groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, groups=groups)

    def forward(self, input):
        if self.kernel != 1:
            A = self.conv_x(input.real)
            C = self.conv_x(input.imag)
            E = self.conv_y(input.real)
            B = self.conv_y(input.imag)
            out = torch.complex(A - B, C + E)
        else:
            out = torch.complex(self.conv(input.real), self.conv(input.imag))
        return out


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # Cut & paste from PyTorch official master until it's in a few official releases - RW
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.
#
#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#
#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#
#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()
#
#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)
#
#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor
#
#
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # type: (Tensor, float, float, float, float) -> Tensor
#     r"""Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         # >>> w = torch.empty(3, 5)
#         # >>> nn.init.trunc_normal_(w)
#     """
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
#     fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
#     if mode == 'fan_in':
#         denom = fan_in
#     elif mode == 'fan_out':
#         denom = fan_out
#     elif mode == 'fan_avg':
#         denom = (fan_in + fan_out) / 2
#
#     variance = scale / denom
#
#     if distribution == "truncated_normal":
#         # constant is stddev of standard normal truncated to (-2, 2)
#         trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
#     elif distribution == "normal":
#         tensor.normal_(std=math.sqrt(variance))
#     elif distribution == "uniform":
#         bound = math.sqrt(3 * variance)
#         tensor.uniform_(-bound, bound)
#     else:
#         raise ValueError(f"invalid distribution {distribution}")
#
#
# def lecun_normal_(tensor):
#     variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
# # from timm.models.registry import register_model


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        C_in = dim
        self.dwconv1 = One_dimensional_complex_conv(C_in, C_in, kernel_size=7, stride=1, padding=3, groups=C_in)
        self.norm = NaiveComplexBatchNorm2d(C_in)
        self.pwconv1 = One_dimensional_complex_conv(C_in, C_in * 2, kernel_size=1, padding=0)
        self.dwconv2 = One_dimensional_complex_conv(C_in * 2, C_in, kernel_size=7, stride=1, padding=3, groups=C_in)
        self.act = ComplexGeLU()
        self.pwconv2 = One_dimensional_complex_conv(C_in, C_in, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = ComplexDropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.dwconv2(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class ConvNeXt_PIB(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9 ,3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        # stem and 3 intermediate downsampling conv layers
        dims = [dim // 2 for dim in dims]
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            One_dimensional_complex_conv(in_chans, dims[0], kernel_size=4, stride=4),
            NaiveComplexBatchNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                NaiveComplexBatchNorm2d(dims[i], eps=1e-6),
                One_dimensional_complex_conv(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = NaiveComplexBatchNorm2d(dims[-1], eps=1e-6)  # final norm layer
        self.avg = ComplexAdaptiveAvgPooling2D(1)
        self.head = ComplexLinear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.fc.weight.data.mul_(head_init_scale)
        self.head.fc.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(self.avg(x))

    def forward(self, x):
        x = torch.complex(x,x)
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        x.abs()
        return x

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt_PIB(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt_PIB(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_PIB(depths=[3, 3, 27, 3], dims=[
                     128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_PIB(depths=[3, 3, 27, 3], dims=[
                     192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_PIB(depths=[3, 3, 27, 3], dims=[
                     256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
