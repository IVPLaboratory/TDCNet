from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (T, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij'))  # 3, T, H, W
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, k=None, v=None, mask=None):
        B_, N, C = x.shape
        if k is None or v is None:
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # [B_, num_heads, N, head_dim]
            k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, T, H, W, C = x.shape
    window_size = list(window_size)
    if T < window_size[0]:
        window_size[0] = T
    if H < window_size[1]:
        window_size[1] = H
    if W < window_size[2]:
        window_size[2] = W
    x = x.view(B, T // window_size[0] if window_size[0] > 0 else 1, window_size[0],
               H // window_size[1] if window_size[1] > 0 else 1, window_size[1],
               W // window_size[2] if window_size[2] > 0 else 1, window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, T, H, W):
    x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class SelfAttention(nn.Module):
    def __init__(self, dim, window_size=(2, 8, 8), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., use_shift=False, shift_size=None, mlp_ratio=2.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_shift = use_shift
        self.shift_size = shift_size if shift_size is not None else tuple([w // 2 for w in window_size]) if use_shift else tuple([0] * len(window_size))
        self.attn1 = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn2 = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def create_mask(self, x_shape, device):
        B, T, H, W, C = x_shape
        img_mask = torch.zeros((1, T, H, W, 1), device=device)
        cnt = 0
        t_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        h_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        w_slices = (slice(0, -self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None))
        for t in t_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, t, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, T, H, W, C = x.shape
        window_size, shift_size = get_window_size((T, H, W), self.window_size, self.shift_size)
        shortcut = x
        x = self.norm1(x)
        pad_t = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        shortcut = F.pad(shortcut, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        _, Tp, Hp, Wp, _ = x.shape
        x_windows = window_partition(x, window_size)
        attn_windows = self.attn1(x_windows, mask=None)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        x = window_reverse(attn_windows, window_size, B, Tp, Hp, Wp)
        x = shortcut + x
        x = x + self.mlp1(self.norm2(x))
        shortcut = x
        x = self.norm3(x)
        if self.use_shift and any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = self.create_mask((B, Tp, Hp, Wp, C), x.device)
            x_windows = window_partition(shifted_x, window_size)
            attn_windows = self.attn2(x_windows, mask=attn_mask)
            attn_windows = attn_windows.view(-1, *(window_size + (C,)))
            shifted_x = window_reverse(attn_windows, window_size, B, Tp, Hp, Wp)
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        if pad_t > 0:
            x = x[:, :T, :, :, :]
            shortcut = shortcut[:, :T, :, :, :]
        if pad_h > 0:
            x = x[:, :, :H, :, :]
            shortcut = shortcut[:, :, :H, :, :]
        if pad_w > 0:
            x = x[:, :, :, :W, :]
            shortcut = shortcut[:, :, :, :W, :]

        x = shortcut + x
        x = x + self.mlp2(self.norm4(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, window_size=(2, 8, 8), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., mlp_ratio=2.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, q, k, v):
        B, T, H, W, C = q.shape
        window_size = get_window_size((T, H, W), self.window_size)
        shortcut = v
        q = self.norm1_q(q)
        k = self.norm1_k(k)
        v = self.norm1_v(v)
        pad_t = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
        q = F.pad(q, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        k = F.pad(k, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        v = F.pad(v, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        _, Tp, Hp, Wp, _ = q.shape

        q_windows = window_partition(q, window_size)
        k_windows = window_partition(k, window_size)
        v_windows = window_partition(v, window_size)
        attn_windows = self.attn(q_windows, k_windows, v_windows)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Tp, Hp, Wp)
        x = shifted_x
        if pad_t > 0:
            x = x[:, :T, :, :, :]
        if pad_h > 0:
            x = x[:, :, :H, :, :]
        if pad_w > 0:
            x = x[:, :, :, :W, :]
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
