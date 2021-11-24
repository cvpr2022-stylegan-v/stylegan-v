# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
import torch.nn.functional as F

from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import conv2d_resample, upfirdn2d, bias_act, fma

from training.motion import MotionEncoder
from training.layers import (
    FullyConnectedLayer,
    GenInput,
    CoordFuser,
    TimeFuser,
    TemporalDifferenceEncoder,
    MultiTimeEncoder,
    JointTimeEncoder,
    Conv2dLayer,
    MappingNetwork,
    remove_diag,
    get_max_dist,
)

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@misc.profiled_function
def fmm_modulate_linear(x: Tensor, weight: Tensor, styles: Tensor, noise=None, activation: str="demod") -> Tensor:
    """
    x: [batch_size, c_in, height, width]
    weight: [c_out, c_in, 1, 1]
    style: [batch_size, num_mod_params]
    noise: Optional[batch_size, 1, height, width]
    """
    batch_size, c_in, h, w = x.shape
    c_out, c_in, kh, kw = weight.shape
    rank = styles.shape[1] // (c_in + c_out)

    assert kh == 1 and kw == 1
    assert styles.shape[1] % (c_in + c_out) == 0

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[:, :c_out * rank] # [batch_size, left_matrix_size]
    right_matrix = styles[:, c_out * rank:] # [batch_size, right_matrix_size]

    left_matrix = left_matrix.view(batch_size, c_out, rank) # [batch_size, c_out, rank]
    right_matrix = right_matrix.view(batch_size, rank, c_in) # [batch_size, rank, c_in]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank) # [batch_size, c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight.squeeze(3).squeeze(2).unsqueeze(0) * (modulation + 1.0) # [batch_size, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, c_out, c_in]
    W = W.to(dtype=x.dtype)

    # out = torch.einsum('boi,bihw->bohw', W, x)
    x = x.view(batch_size, c_in, h * w) # [batch_size, c_in, h * w]
    out = torch.bmm(W, x) # [batch_size, c_out, h * w]
    out = out.view(batch_size, c_out, h, w) # [batch_size, c_out, h, w]

    if not noise is None:
        out = out.add_(noise)

    return out

#----------------------------------------------------------------------------

@misc.profiled_function
def maybe_upsample(x, upsampling_mode: str, up: int) -> Tensor:
    if up == 1:
        return x

    if upsampling_mode == 'bilinear':
        x = F.interpolate(x, mode='bilinear', align_corners=True, scale_factor=up)
    elif upsampling_mode == 'nearest':
        x = F.interpolate(x, mode='nearest', scale_factor=up)
    else:
        raise NotImplementedError(f"Unknown upsampling mode: {upsampling_mode}")

    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        instance_norm   = False,        # Use instance norm?
        cfg             = {},           # Additional config
    ):
        super().__init__()

        self.cfg = cfg
        self.resolution = resolution
        self.use_fmm = self.resolution in self.cfg.fmm.get('resolutions', [])
        self.up = up
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        if self.use_fmm:
            self.affine = FullyConnectedLayer(w_dim, (in_channels + out_channels) * self.cfg.fmm.rank, bias_init=0)
        else:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if self.cfg.use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.instance_norm = instance_norm

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.cfg.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.cfg.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster

        if self.instance_norm:
            x = x / (x.std(dim=[2,3], keepdim=True) + 1e-8) # [batch_size, c, h, w]

        if self.use_fmm:
            x = fmm_modulate_linear(x=x, weight=self.weight, styles=styles, noise=noise, activation=self.cfg.fmm.activation)
        else:
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
                padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
                fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        motion_w_dim,                       # Motion code size
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        cfg                 = {},           # Additional config
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.w_dim = w_dim

        if resolution <= self.cfg.input.resolution:
            self.resolution = self.cfg.input.resolution
            self.up = 1
            self.input_resolution = self.cfg.input.resolution
        else:
            self.resolution = resolution
            self.up = 2
            self.input_resolution = resolution // 2

        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.use_fmm = self.resolution in self.cfg.fmm.get('resolutions', [])
        self.kernel_size = 1 if self.use_fmm else 3
        self.use_instance_norm = self.use_fmm and in_channels > 0 and cfg.get('fmm', {}).get('instance_norm', False)

        if self.cfg.time_enc.per_resolution:
            assert self.architecture != 'resnet'
            self.time_fuser = TimeFuser(self.cfg, resolution=self.resolution, motion_w_dim=motion_w_dim)
            self.time_emb_dim = self.time_fuser.get_total_dim()
        else:
            self.time_fuser = None
            self.time_emb_dim = 0

        if in_channels == 0:
            self.input = GenInput(self.cfg, out_channels, w_dim, motion_w_dim=motion_w_dim)
            conv1_in_channels = self.input.total_dim + self.time_emb_dim
        else:
            up_for_conv0 = 1 if self.use_fmm else self.up # For FMM, we'll upsample manually

            if self.cfg.coords.enabled and (not self.cfg.coords.per_resolution or self.resolution > self.input_resolution):
                assert self.architecture != 'resnet'
                self.coord_fuser = CoordFuser(
                    cfg=self.cfg.coords,
                    w_dim=self.w_dim,
                    resolution=self.resolution // up_for_conv0,
                    t_resolution=self.cfg.max_num_frames)
                conv0_in_channels = in_channels + self.coord_fuser.total_dim + self.time_emb_dim
            else:
                self.coord_fuser = None
                conv0_in_channels = in_channels + self.time_emb_dim

            # We are not using instance norm in conv0, because we concatenate coords to it (sometimes)
            # and some coords can be all-zero
            self.conv0 = SynthesisLayer(conv0_in_channels, out_channels, w_dim=w_dim, resolution=self.resolution, up=up_for_conv0,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last,
                kernel_size=self.kernel_size, cfg=cfg, instance_norm=False, **layer_kwargs)
            self.num_conv += 1
            conv1_in_channels = out_channels

        self.conv1 = SynthesisLayer(conv1_in_channels, out_channels, w_dim=w_dim, resolution=self.resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, kernel_size=self.kernel_size, cfg=cfg,
            instance_norm=self.use_instance_norm, **layer_kwargs)
        self.num_conv += 1

        if self.cfg.get('num_extra_convs', {}).get(str(self.resolution), 0) > 0:
            assert self.architecture != 'resnet', "Not implemented for resnet"
            self.extra_convs = nn.ModuleList([
                SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=self.resolution,
                    conv_clamp=conv_clamp, channels_last=self.channels_last, kernel_size=self.kernel_size,
                    instance_norm=self.use_instance_norm, cfg=cfg, **layer_kwargs)
                    for _ in range(self.cfg.num_extra_convs[str(self.resolution)])])
            self.num_conv += len(self.extra_convs)
        else:
            self.extra_convs = None

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=self.up,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, t=None, motion_w=None, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or (isinstance(x, Tensor) and int(x.shape[0]) == 1))

        # Input.
        if self.in_channels == 0:
            conv1_w = next(w_iter)
            x = self.input(ws.shape[0], conv1_w, t=t, motion_w=motion_w, device=ws.device, dtype=dtype, memory_format=memory_format)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.input_resolution, self.input_resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        x = maybe_upsample(x, self.cfg.fmm_upsampling_mode, self.up) if self.use_fmm else x # [batch_size, c, h, w]

        # Main layers.
        if self.in_channels == 0:
            x = x if self.time_fuser is None else self.time_fuser(x, t=t, motion_w=motion_w) # [batch_size, c + time_emb_dim, h, w]
            x = self.conv1(x, conv1_w, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            conv0_w = next(w_iter)

            if self.coord_fuser is not None:
                x = self.coord_fuser(x, conv0_w, t=t, dtype=dtype, memory_format=memory_format)

            if self.time_fuser is not None:
                x = self.time_fuser(x, t=t, motion_w=motion_w) # [b, c + coord_dim + time_dim, h, w]

            x = self.conv0(x, conv0_w, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        if not self.extra_convs is None:
            for conv, w in zip(self.extra_convs, w_iter):
                x = conv(x, w, fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.input_resolution, self.input_resolution])

            if self.up == 2:
                if self.use_fmm:
                    img = maybe_upsample(img, self.cfg.fmm_upsampling_mode, 2)
                else:
                    img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        cfg             = {},       # Additional config
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()

        self.w_dim = w_dim
        self.cfg = cfg
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if self.cfg.motion.w_dim > 0:
            self.motion_encoder = MotionEncoder(self.cfg, resolutions=self.block_resolutions)
            self.motion_w_dim = self.motion_encoder.get_output_dim()
        else:
            self.motion_encoder = None
            self.motion_w_dim = 0

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=self.w_dim + (self.motion_w_dim if self.cfg.time_enc.cond_type == 'concat_w' else 0),
                motion_w_dim=self.motion_w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                cfg=cfg,
                **block_kwargs)
            self.num_ws += block.num_conv

            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, t=None, c=None, l=None, motion_noise=None, motion_w=None, **block_kwargs):
        assert len(ws) == len(c) == len(t), f"Wrong shape: {ws.shape}, {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        block_ws = []

        if self.motion_encoder is None:
            ws = ws.repeat_interleave(t.shape[1], dim=0) # [batch_size * num_frames, num_ws, w_dim]
            motion_w = None
        else:
            if motion_w is None:
                motion_info = self.motion_encoder(c, t, l=l, w=ws[:, 0], motion_noise=motion_noise) # [batch_size * num_frames, motion_w_dim]
                motion_w = motion_info['motion_w'] # [batch_size * num_frames, motion_w_dim]

            if not self.cfg.time_enc.per_resolution and self.cfg.time_enc.cond_type in ['concat_w', 'sum_w']:
                misc.assert_shape(motion_w, [t.shape[0] * t.shape[1], self.motion_w_dim])

                if self.cfg.time_enc.cond_type == 'concat_w':
                    motion_ws = motion_w.unsqueeze(1).repeat(1, self.num_ws, 1) # [batch_size * num_frames, num_ws, motion_w_dim]
                    ws = torch.cat([ws.repeat_interleave(t.shape[1], dim=0), motion_ws], dim=2) # [batch_size * num_frames, num_ws, w_dim + motion_w_dim]
                elif self.cfg.time_enc.cond_type == 'sum_w':
                    ws = ws.repeat_interleave(t.shape[1], dim=0) + motion_w.unsqueeze(1) # [batch_size * num_frames, num_ws, w_dim + motion_w_dim]
            else:
                ws = ws.repeat_interleave(t.shape[1], dim=0) # [batch_size * num_frames, num_ws, w_dim]

        with torch.autograd.profiler.record_function('split_ws'):
            ws = ws.to(torch.float32)
            w_idx = 0

            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')

            if self.cfg.time_enc.per_resolution:
                motion_w = motion_info['motion_w'][res] # [batch_size * num_frames, motion_w_dim]

                if self.cfg.time_enc.cond_type == "concat_w":
                    cur_ws = torch.cat([cur_ws, motion_w.unsqueeze(1).repeat(1, cur_ws.shape[1], 1)], dim=2) # [batch_size * num_frames, num_cur_ws, w_dim + motion_w_dim]
                elif self.cfg.time_enc.cond_type == "sum_w":
                    cur_ws = cur_ws + motion_w.unsqueeze(1) # [batch_size * num_frames, num_cur_ws, w_dim]
                elif self.cfg.time_enc.cond_type == "concat_act":
                    pass
                else:
                    raise NotImplementedError(f"Unkown agg op: {self.cfg.motion.agg}")

            if self.cfg.time_enc.cond_type != 'concat_act':
                motion_w = None # To make sure that we do not leak.

            x, img = block(x, img, cur_ws, t=t, motion_w=motion_w, **block_kwargs)

        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        cfg                 = {},   # Config
    ):
        super().__init__()

        self.cfg = cfg
        self.sampling_dict = OmegaConf.to_container(OmegaConf.create({**self.cfg.sampling}))
        self.z_dim = self.cfg.z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, cfg=cfg, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=self.z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, t, l, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        assert len(z) == len(c) == len(t), f"Wrong shape: {z.shape}, {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        ws = self.mapping(z, c, l=l, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff) # [batch_size, num_ws, w_dim]
        img = self.synthesis(ws, t=t, c=c, l=l, **synthesis_kwargs) # [batch_size * num_frames, c, h, w]

        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        c_dim               = 0,            # Embedding size for t.
        cfg                 = {},           # Main config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()
        conv0_in_channels = in_channels if in_channels > 0 else tmp_channels

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        if self.cfg.hyper_type in ['hyper', 'dummy_hyper']:
            assert next(trainable_iter)
            self.conv0 = SynthesisLayer(
                conv0_in_channels,
                tmp_channels,
                w_dim=c_dim,
                resolution=self.resolution,
                kernel_size=3,
                activation=activation,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                cfg=self.cfg.dummy_synth_cfg)
        elif self.cfg.hyper_type == 'no_hyper':
            self.conv0 = Conv2dLayer(conv0_in_channels, tmp_channels, kernel_size=3, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)
        else:
            raise NotImplementedError("Unknown hyper type:", self.cfg.hyper_type)

        if int(self.resolution) in [int(r) for r in self.cfg.contr.get('resolutions', [])] and self.cfg.num_frames_per_sample > 1:
            assert self.cfg.agg.type != "concat" or self.cfg.agg.concat_res < self.resolution, \
                f"Cant compute similarities after concatenation: {self.cfg.agg.concat_res} > {self.resolution}"
            self.contr = GroupwiseContrastiveLayer(
                cfg=self.cfg, in_channels=conv0_in_channels, c_dim=c_dim, resolution=self.resolution,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            conv1_in_channels = self.contr.get_output_dim()
        else:
            self.contr = None
            conv1_in_channels = tmp_channels

        self.conv1 = Conv2dLayer(conv1_in_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(conv0_in_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, c, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        c_weight = 1.0 if self.cfg.get('is_hyper', True) else 0.0

        if self.cfg.hyper_type == 'hyper':
            cond_kwargs = {'w': c}
        elif self.cfg.hyper_type == 'dummy_hyper':
            cond_kwargs = {'w': c * 0.0}
        elif self.cfg.hyper_type == 'no_hyper':
            cond_kwargs = {}
        else:
            raise NotImplementedError("Unknwon hyper type", self.cfg.hyper_type)

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, **cond_kwargs)
            x = x if self.contr is None else self.contr(x, c)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x, **cond_kwargs)
            x = x if self.contr is None else self.contr(x, c)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [N(C+1)HW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class GroupwiseContrastiveLayer(torch.nn.Module):
    """
    This layer compares images of the same video with one another
    and concatenates the similarity scores back to their original activations
    """
    def __init__(self, cfg, in_channels: int, resolution: int, c_dim: int, conv_clamp: int=None, channels_last: bool=False):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.group_size = self.cfg.num_frames_per_sample
        self.dim = self.cfg.contr.dim
        self.scale = 1 if self.dim <= 3 else ((self.dim - 2) ** 2 / self.dim) ** 0.5
        self.diff_based = self.cfg.contr.get('diff_based', False)

        self.transform = SynthesisLayer(
            in_channels=in_channels,
            out_channels=self.cfg.contr.dim,
            w_dim=c_dim,
            resolution=resolution,
            kernel_size=self.cfg.contr.kernel_size,
            activation='lrelu',
            conv_clamp=conv_clamp,
            channels_last=channels_last,
            cfg=self.cfg.dummy_synth_cfg,
        )
        self.agg = self.cfg.contr.agg

    def get_output_dim(self) -> int:
        if self.agg in ["mean", "min", "max", "fmin_rmax"]:
            return self.in_channels + 1
        elif self.agg == "none":
            if self.diff_based:
                return self.in_channels + (self.group_size - 1) * (self.group_size - 2)
            else:
                return self.in_channels + self.group_size - 1
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        bn, c_in, h, w = x.shape
        num_groups = bn // self.group_size
        y = self.transform(x, c) # [bn, dim, h, w]
        y = y.reshape(num_groups, self.group_size, self.dim, h, w) # [num_groups, group_size, dim, h, w]

        if self.diff_based:
            # We first compute differences between frames features
            # and then we compute similarities between those vectors
            # This should show D how the pixels moved
            d = y[:, 1:] - y[:, :-1] # [num_groups, group_size - 1, dim, h, w]
            # Unfortunately, we cannot normalize the diffs because R1 penalty falls with NaNs...
            # So, normalize the scale at least somehow...
            d_sims = (1.0 / np.sqrt(d.shape[2])) * d.permute(0, 3, 4, 1, 2) @ d.permute(0, 3, 4, 2, 1) # [num_groups, h, w, group_size - 1, group_size - 1]
            assert not torch.isnan(d_sims).any(), "There are NaNs in the diffs tensor"
            d_sims = remove_diag(d_sims) # [num_groups, h, w, group_size - 1, group_size - 2]
            y = d_sims.unsqueeze(1).repeat(1, self.group_size, 1, 1, 1, 1) # [num_groups, group_size, h, w, group_size - 1, group_size - 2]
            y = y.view(bn, h, w, (self.group_size - 1) * (self.group_size - 2)).permute(0, 3, 1, 2) # [bn, (group_size - 1) * (group_size - 2), h, w]
        else:
            y = F.normalize(y, dim=2) # [num_groups, group_size, dim, h, w]
            y = y.permute(0, 3, 4, 1, 2) @ y.permute(0, 3, 4, 2, 1) # [num_groups, h, w, group_size, group_size]
            y = y * self.scale # [num_groups, h, w, group_size, group_size]
            y = remove_diag(y) # [num_groups, h, w, group_size, group_size - 1]
            y = y.permute(0, 3, 4, 1, 2) # [num_groups, group_size, group_size - 1, h, w]
            y = y.reshape(bn, self.group_size - 1, h, w) # [bn, group_size - 1, h, w]

        if self.agg == "mean":
            y = y.mean(dim=1, keepdim=True) # [bn, 1, h, w]
        elif self.agg == "max":
            y = y.max(dim=1, keepdim=True)[0] # [bn, 1, h, w]
        elif self.agg == "min":
            y = y.min(dim=1, keepdim=True)[0] # [bn, 1, h, w]
        elif self.agg == "none":
            y = y # [bn, group_size - 1, h, w]
        else:
            raise NotImplementedError

        y = torch.cat([x, y], dim=1) # [bn, in_channel + d, h, w]

        return y

#----------------------------------------------------------------------------

@persistence.persistent_class
class FeatDiffLayer(torch.nn.Module):
    """
    Computes differences between consecutive frames features
    """
    def __init__(self, cfg, in_channels: int, dim: int, resolution: int, c_dim: int, conv_clamp: int=None, channels_last: bool=False):
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.group_size = self.cfg.num_frames_per_sample
        self.dim = dim

        self.transform = SynthesisLayer(
            in_channels=in_channels,
            out_channels=self.dim,
            w_dim=c_dim,
            resolution=resolution,
            kernel_size=3,
            activation='lrelu',
            conv_clamp=conv_clamp,
            channels_last=channels_last,
            cfg=self.cfg.dummy_synth_cfg,
        )

    def get_output_dim(self) -> int:
        return (self.group_size - 1) * self.dim

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        bn, c_in, h, w = x.shape
        num_groups = bn // self.group_size
        y = self.transform(x, c) # [bn, dim, h, w]
        y = y.reshape(num_groups, self.group_size, self.dim, h, w) # [num_groups, group_size, dim, h, w]
        d = y[:, 1:] - y[:, :-1] # [num_groups, group_size - 1, dim, h, w]
        d = d.view(num_groups, (self.group_size - 1) * self.dim, h, w) # [num_groups, (group_size - 1) * c, h, w]

        return d

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cfg                 = {},       # Architecture config.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()

        self.cfg = cfg
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

        if self.cfg.predict_dists_weight > 0.0:
            self.dist_predictor = nn.Sequential(
                FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation),
                torch.nn.Flatten(),
                FullyConnectedLayer(in_channels, get_max_dist(self.cfg.sampling), activation='linear'),
            )
        else:
            self.dist_predictor = None

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)

        x = self.conv(x)
        dist_preds = None if self.dist_predictor is None else self.dist_predictor(x.flatten(1)) # [batch_size]
        x = self.fc(x.flatten(1))
        x = self.out(x) # [batch_size, out_dim]

        if not self.dist_predictor is None:
            # If one uncomments this, then we'll encounter a DDP consistency error for some reason
            x = x + dist_preds.sum() * 0.0

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim)) # [batch_size, 1]

        assert x.dtype == dtype
        return x, dist_preds

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        cfg                 = {},       # Additional config.
    ):
        super().__init__()

        self.cfg = cfg
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        if self.cfg.num_frames_per_sample > 1:
            if self.cfg.time_enc_type == 'diff':
                self.time_encoder = TemporalDifferenceEncoder(self.cfg)
            elif self.cfg.time_enc_type == 'multi':
                self.time_encoder = MultiTimeEncoder(self.cfg)
            elif self.cfg.time_enc_type == 'joint':
                self.time_encoder = JointTimeEncoder(self.cfg)
            else:
                raise NotImplementedError(f"Unknown time encoder in D: {self.cfg.time_enc_type}")

            assert self.time_encoder.get_total_dim() > 0
        else:
            self.time_encoder = None

        if self.c_dim == 0 and self.time_encoder is None:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        conditioning_dim = c_dim + (0 if self.time_encoder is None else self.time_encoder.get_total_dim())
        cur_layer_idx = 0

        if self.cfg.agg.type == "concat" and self.cfg.agg.get('concat_diff_dim', 0) > 0:
            self.diff_transform = FeatDiffLayer(
                cfg=self.cfg,
                in_channels=channels_dict[self.cfg.agg.concat_res] // self.cfg.num_frames_div_factor,
                dim=self.cfg.agg.concat_diff_dim,
                resolution=self.cfg.agg.concat_res,
                c_dim=conditioning_dim,
                conv_clamp=conv_clamp)
        else:
            self.diff_transform = None

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]

            if self.cfg.agg.type == "concat":
                # Adjust numbers of channels
                if res // 2 == self.cfg.agg.concat_res:
                    out_channels = out_channels // self.cfg.num_frames_div_factor

                if res == self.cfg.agg.concat_res:
                    in_channels = (in_channels // self.cfg.num_frames_div_factor) * self.cfg.num_frames_per_sample
                    in_channels += (0 if self.diff_transform is None else self.diff_transform.get_output_dim())

            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, cfg=self.cfg, c_dim=conditioning_dim, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(z_dim=0, c_dim=conditioning_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, cfg=self.cfg, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, t, **block_kwargs):
        # TODO: pass img in [b, c, t, h, w] shape instead of [b * t, c, h, w]

        assert len(img) == t.shape[0] * t.shape[1], f"Wrong shape: {img.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        if not self.time_encoder is None:
            t_embs = self.time_encoder(t.view(-1, self.cfg.num_frames_per_sample)) # [batch_size, t_dim]
            c_orig = torch.cat([c, t_embs], dim=1) # [batch_size, c_dim + t_dim]
            c = c_orig.repeat_interleave(t.shape[1], dim=0) # [batch_size * num_frames, c_dim + t_dim]

            if self.cfg.dummy_c:
                c = c * 0.0
                c_orig = c_orig * 0.0

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if self.cfg.agg.type == "concat" and res == self.cfg.agg.concat_res:
                d = None if self.diff_transform is None else self.diff_transform(x, c) # [batch_size, num_frames - 1, diff_c, h, w]
                x = x.view(-1, self.cfg.num_frames_per_sample, *x.shape[1:]) # [batch_size, num_frames, c, h, w]
                x = x.view(x.shape[0], -1, *x.shape[3:]) # [batch_size, num_frames * c, h, w]
                x = x if self.diff_transform is None else torch.cat([x, d], dim=1) # [batch_size, num_frames * c + (num_frames - 1) * d_dim, h, w]
                c = c_orig
            x, img = block(x, img, c, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        if c.shape[1] > 0:
            cmap = self.mapping(None, c)
        x, dist_preds = self.b4(x, img, cmap)
        x = x.squeeze(1) # [batch_size]

        return {'image_logits': x, 'dist_preds': dist_preds}

#----------------------------------------------------------------------------
