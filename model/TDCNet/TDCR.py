import torch
import torch.nn as nn
import torch.nn.functional as F


class TDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), groups=1, bias=False, step=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.step = step
        self.groups = groups

    def get_time_gradient_weight(self):
        weight = self.conv.weight
        kT, kH, kW = weight.shape[2:]
        grad_weight = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
        if kT == 5:
            if self.step == -1:
                grad_weight[:, :, :, :, :] = -weight[:, :, :, :, :]
                grad_weight[:, :, 4, :, :] = weight[:, :, 0, :, :] + weight[:, :, 1, :, :] + weight[:, :, 2, :, :] + weight[:, :, 3, :, :] + weight[:, :, 4, :, :]
            elif self.step == 1:
                grad_weight[:, :, 4, :, :] = weight[:, :, 4, :, :]
                grad_weight[:, :, 3, :, :] = weight[:, :, 3, :, :] - weight[:, :, 4, :, :]
                grad_weight[:, :, 2, :, :] = weight[:, :, 2, :, :] - weight[:, :, 3, :, :]
                grad_weight[:, :, 1, :, :] = weight[:, :, 1, :, :] - weight[:, :, 2, :, :]
                grad_weight[:, :, 0, :, :] = -weight[:, :, 1, :, :]
            elif self.step == 2:
                grad_weight[:, :, 4, :, :] = weight[:, :, 4, :, :]
                grad_weight[:, :, 3, :, :] = weight[:, :, 3, :, :]
                grad_weight[:, :, 2, :, :] = weight[:, :, 2, :, :] - weight[:, :, 4, :, :]
                grad_weight[:, :, 1, :, :] = -weight[:, :, 3, :, :]
                grad_weight[:, :, 0, :, :] = -weight[:, :, 2, :, :]
        else:
            grad_weight = weight
        bias = self.conv.bias
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
        return grad_weight, bias

    def forward(self, x):
        weight, bias = self.get_time_gradient_weight()
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        return self.conv(x)


class RepConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1), groups=1, deploy=False):
        super(RepConv3D, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        if self.deploy:
            self.conv_reparam = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
        else:
            self.l_tdc = nn.Sequential(
                TDC(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, step=-1),
                nn.BatchNorm3d(out_channels)
            )
            self.s_tdc = nn.Sequential(
                TDC(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, step=1),
                nn.BatchNorm3d(out_channels)
            )
            self.m_tdc = nn.Sequential(
                TDC(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, step=2),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        if self.deploy:
            out = F.relu(self.conv_reparam(x))
        else:
            out = self.s_tdc(x) + self.m_tdc(x) + self.l_tdc(x)
            out = F.relu(out)
        return out

    def get_equivalent_kernel_bias(self):
        kernel_s_tdc, bias_s_tdc = self._fuse_conv_bn(self.s_tdc)
        kernel_m_tdc, bias_m_tdc = self._fuse_conv_bn(self.m_tdc)
        kernel_l_tdc, bias_l_tdc = self._fuse_conv_bn(self.l_tdc)
        kernel = kernel_s_tdc + kernel_m_tdc + kernel_l_tdc
        bias = bias_s_tdc + bias_m_tdc + bias_l_tdc
        return kernel, bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_reparam = nn.Conv3d(
            self.in_channels, self.out_channels, (5, 3, 3), self.stride,
            (2, 1, 1), groups=self.groups, bias=True
        )
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias
        self.deploy = True
        del self.s_tdc
        del self.m_tdc
        del self.l_tdc

    @staticmethod
    def _fuse_conv_bn(branch):
        if branch is None:
            return 0, 0

        def find_conv(module):
            if isinstance(module, nn.Conv3d):
                return module
            for child in module.children():
                conv = find_conv(child)
                if conv is not None:
                    return conv
            return None

        conv = find_conv(branch[0])
        bn = branch[1]
        if hasattr(branch[0], 'get_time_gradient_weight'):
            w, bias = branch[0].get_time_gradient_weight()
        else:
            w = conv.weight
            if conv.bias is not None:
                bias = conv.bias
            else:
                bias = torch.zeros_like(bn.running_mean)
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1, 1)
        bias = (bias - mean) / var_sqrt * gamma + beta
        return w, bias
