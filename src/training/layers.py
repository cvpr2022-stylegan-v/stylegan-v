import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from scipy import stats

from src.torch_utils import persistence
from src.torch_utils.ops import bias_act, upfirdn2d, conv2d_resample
from src.torch_utils import misc

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        cfg             = {},       # Additional config
    ):
        super().__init__()

        self.cfg = cfg
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        if cfg.get('use_video_len_cond', False):
            self.time_encoder = FixedTimeEncoder(self.cfg.max_num_frames, transformer_pe=self.cfg.transformer_pe, skip_small_t_freqs=self.cfg.get('skip_small_t_freqs', 0))
            time_dim = self.time_encoder.get_dim()
        else:
            self.time_encoder = None
            time_dim = 0

        features_list = [z_dim + embed_features + time_dim] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, l=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

            if not self.time_encoder is None:
                video_len_embs = self.time_encoder(l.unsqueeze(1)) # [batch_size, num_fourier_feats]
                x = torch.cat([x, video_len_embs], dim=1) if x is not None else video_len_embs # [batch_size, d]

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        instance_norm   = False,        # Should we apply instance normalization to y?
        lr_multiplier   = 1.0,          # Learning rate multiplier.
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.instance_norm = instance_norm
        self.lr_multiplier = lr_multiplier

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * (self.weight_gain * self.lr_multiplier)
        b = (self.bias.to(x.dtype) * self.lr_multiplier) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.instance_norm:
            x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-8) # [batch_size, c, h, w]

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class GenInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, w_dim: int, motion_w_dim: int=None):
        super().__init__()

        self.cfg = cfg

        if self.cfg.input.type == 'const':
            self.input = torch.nn.Parameter(torch.randn([channel_dim, self.cfg.input.resolution, self.cfg.input.resolution]))
            self.total_dim = channel_dim
        elif self.cfg.input.type == 'coords':
            self.input = CoordsInput(self.cfg, w_dim, channel_dim)
            self.total_dim = self.input.get_total_dim()
        elif self.cfg.input.type == 'temporal':
            self.input = TemporalInput(self.cfg, channel_dim, w_dim=w_dim, motion_w_dim=motion_w_dim)
            self.total_dim = self.input.get_total_dim()
        elif self.cfg.input.type == 'variable':
            self.input = VariableInput(self.cfg, channel_dim, w_dim)
            self.total_dim = self.input.get_total_dim()
        else:
            raise NotImplementedError

    def forward(self, batch_size: int, w: Tensor=None, t: Optional[Tensor]=None, motion_w: Optional[Tensor]=None, device=None, dtype=None, memory_format=None) -> Tensor:
        if self.cfg.input.type == 'const':
            x = self.input.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        elif self.cfg.input.type == 'coords':
            x = self.input(batch_size, w, device=device, dtype=dtype, memory_format=memory_format)
        elif self.cfg.input.type == 'temporal':
            x = self.input(t, w=w, motion_w=motion_w) # [batch_size, d, h, w]
        elif self.cfg.input.type == 'variable':
            x = self.input(w) # [batch_size, d, h, w]
        else:
            raise NotImplementedError

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class CoordsInput(nn.Module):
    def __init__(self, cfg: DictConfig, w_dim: int, channel_dim):
        super().__init__()

        self.cfg = cfg
        self.channel_dim = channel_dim
        self.coord_fuser = CoordFuser(self.cfg.input.coord_fuser_cfg, w_dim, self.cfg.input.resolution)

        if self.cfg.input.has_const:
            self.const = torch.nn.Parameter(torch.randn([channel_dim, self.cfg.input.resolution, self.cfg.input.resolution]))
        else:
            self.const = None

    def get_total_dim(self) -> int:
        return self.coord_fuser.total_dim + (0 if self.const is None else self.const.shape[0])

    def forward(self, batch_size: int, w: Optional[Tensor]=None, device='cpu', dtype=None, memory_format=None) -> Tensor:
        if self.const is None:
            x = torch.empty(batch_size, 0, self.cfg.resolution, self.cfg.resolution) # [batch_size, 0, h, w]
        else:
            x = self.const.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [batch_size, c_dim, h, w]

        x = x.to(device, dtype=dtype, memory_format=memory_format) # [batch_size, c_dim, h, w]
        out = self.coord_fuser(x, w, dtype=dtype, memory_format=memory_format) # [batch_size, c_dim + coord_dim, h, w]

        return out

#----------------------------------------------------------------------------

def construct_time_encoder(G_cfg: DictConfig, resolution: int):
    if G_cfg.time_enc.type == 'predictable':
        return PredictableTimeEncoder(
            max_period_len=G_cfg.motion.max_period_len,
            latent_dim=G_cfg.motion.w_dim,
            num_feats=G_cfg.time_enc.num_feats,
            fourier_scale=G_cfg.time_enc.fourier_scale,
        )
    elif G_cfg.time_enc.type == 'periodic_feats':
        if G_cfg.time_enc.per_resolution:
            num_feats_per_freq = min(G_cfg.time_enc.channel_base // resolution, G_cfg.time_enc.channel_max)
        else:
            num_feats_per_freq = G_cfg.time_enc.num_feats_per_freq

        return PeriodicFeatsTimeEncoder(
            latent_dim=G_cfg.motion.w_dim,
            cfg=G_cfg,
        )
    else:
        return FixedTimeEncoder(
            max_num_frames=G_cfg.motion.max_period_len,
            transformer_pe=G_cfg.transformer_pe,
            skip_small_t_freqs=G_cfg.get('skip_small_t_freqs', 0),
        )

#----------------------------------------------------------------------------

@persistence.persistent_class
class FixedTimeEncoder(nn.Module):
    def __init__(self,
            max_num_frames: int,            # Maximum T size
            transformer_pe: bool=False,     # Whether we should use positional embeddings from Transformer
            d_model: int=512,               # d_model for Transformer PE's
            skip_small_t_freqs: int=0,      # How many high frequencies we should skip
        ):
        super().__init__()

        assert max_num_frames >= 1, f"Wrong max_num_frames: {max_num_frames}"

        if transformer_pe:
            assert skip_small_t_freqs == 0, "Cant use `skip_small_t_freqs` with `transformer_pe`"
            fourier_coefs = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0) # [1, d_model]
        else:
            fourier_coefs = construct_log_spaced_freqs(max_num_frames, skip_small_t_freqs=skip_small_t_freqs)

        self.register_buffer('fourier_coefs', fourier_coefs) # [1, num_fourier_feats]

    def get_dim(self) -> int:
        return self.fourier_coefs.shape[1] * 2

    def progressive_update(self, curr_kimg):
        pass

    def forward(self, t: Tensor) -> Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        t = t.view(-1).float() # [batch_size * num_frames]
        fourier_raw_embs = self.fourier_coefs * t.unsqueeze(1) # [bf, num_fourier_feats]

        fourier_embs = torch.cat([
            fourier_raw_embs.sin(),
            fourier_raw_embs.cos(),
        ], dim=1) # [bf, num_fourier_feats * 2]

        return fourier_embs

#----------------------------------------------------------------------------

@persistence.persistent_class
class PredictableTimeEncoder(nn.Module):
    def __init__(self, max_period_len: int, latent_dim: int, num_feats: int, fourier_scale: float=1.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_feats = num_feats
        self.max_period_len = max_period_len

        # Creating the affine without bias so not to sync motion
        self.affine = FullyConnectedLayer(latent_dim, num_feats * 2, activation='linear', bias=False)
        self.register_buffer('fourier_scale', torch.tensor(fourier_scale).float())

    def get_dim(self) -> int:
        return self.num_feats * 2

    def progressive_update(self, curr_kimg):
        pass

    def forward(self, t: Tensor, latent: Tensor) -> Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"
        assert t.numel() == len(latent), f"Wrong shape: {t.shape}, {latent.shape}"
        misc.assert_shape(latent, [t.numel(), self.latent_dim])

        t = t.view(-1).float() # [bf]
        wave_params = self.affine(latent) # [bf, num_feats * 2]
        periods, phases = wave_params[:, :self.num_feats], wave_params[:, self.num_feats:] # [bf, num_feats], [bf, num_feats]
        raw_fourier_embs = periods * t.unsqueeze(1) / self.max_period_len + phases # [bf, num_feats]
        fourier_embs = torch.cat([raw_fourier_embs.sin(), raw_fourier_embs.cos()], dim=1) # [bf, num_feats * 2]

        return fourier_embs

#----------------------------------------------------------------------------

@persistence.persistent_class
class PeriodicFeatsTimeEncoder(nn.Module):
    def __init__(self,
        latent_dim: int=512,
        cfg: DictConfig = {},
    ):
        super().__init__()

        self.cfg = cfg
        self.latent_dim = latent_dim
        self.num_feats_per_freq = self.cfg.time_enc.get('num_feats_per_freq', 1)
        self.use_cosine = self.cfg.time_enc.get('use_cosine', True)
        if self.cfg.time_enc.freqs_dist == 'linspace':
            freqs = construct_frequencies(self.cfg.time_enc.num_freqs, self.cfg.time_enc.min_period_len, self.cfg.time_enc.max_period_len)
        else:
            raise NotImplementedError(f"Unknown freqs dist: {self.cfg.time_enc.freqs_dist}")
        self.register_buffer('freqs', freqs) # [1, num_fourier_feats]

        if self.cfg.time_enc.get('growth_kimg', 0) > 0:
            self.register_buffer('weights', torch.zeros_like(freqs))
        else:
            self.weights = None

        # Creating the affine without bias so not to sync motion
        self.amplitudes_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1] * self.num_feats_per_freq * 2, activation='linear', bias=False)

        if self.cfg.time_enc.get('predict_periods'):
            self.periods_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1] * self.num_feats_per_freq, activation='linear', bias=False)
        else:
            self.periods_predictor = None

        if self.cfg.time_enc.get('predict_phases'):
            self.phase_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1] * self.num_feats_per_freq, activation='linear', bias=False)
        else:
            self.phase_predictor = None

        self.progressive_update(0)

    def get_dim(self) -> int:
        return self.num_feats_per_freq * self.freqs.shape[1] * (2 if self.use_cosine else 1)

    def progressive_update(self, curr_kimg: int):
        if self.cfg.time_enc.get('growth_kimg', 0) > 0:
            new_weights = compute_progressive_weights(self.freqs.shape[1], curr_kimg, self.cfg.time_enc.growth_kimg, self.cfg.time_enc.num_opened_dims) # [num_freqs]
            new_weights = torch.from_numpy(new_weights).to(self.freqs.device) # [num_freqs]
            new_weights = new_weights.repeat_interleave(self.num_feats_per_freq, dim=0) # [num_freqs * num_feats_per_freq]
            self.weights.data.copy_(new_weights) # [1, num_freqs * num_feats_per_freq]

    def forward(self, t: Tensor, latent: Tensor, return_coefs: bool=False) -> Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"
        assert t.numel() == len(latent), f"Wrong shape: {t.shape}, {latent.shape}"
        misc.assert_shape(latent, [t.numel(), self.latent_dim])

        batch_size, num_frames = t.shape # [1], [1]
        t = t.view(-1).float().unsqueeze(1) # [bf, 1]

        if not self.periods_predictor is None:
            periods = self.periods_predictor(latent) # [bf, num_fourier_feats * num_feats_per_freq]
            raw_pos_embs = t * (periods.tanh() + 1) # [bf, num_fourier_feats * num_feats_per_freq]
        else:
            raw_pos_embs = t # [bf, 1]
            periods = None

        freqs = self.freqs.repeat_interleave(self.num_feats_per_freq, dim=1) # [1, num_fourier_feats * num_feats_per_freq]
        raw_pos_embs = freqs * raw_pos_embs # [bf, num_fourier_feats * num_feats_per_freq]

        if not self.phase_predictor is None or (self.training and self.cfg.time_enc.phase_dropout_std > 0.0):
            period_lens = 2 * np.pi / self.freqs.squeeze(0) # [num_fourier_feats]
            phase_scales = self.cfg.time_enc.max_period_len / period_lens # [num_fourier_feats]
            phase_scales = phase_scales.unsqueeze(0).repeat_interleave(self.num_feats_per_freq, dim=1) # [bf, num_fourier_feats * num_feats_per_freq]

        if not self.phase_predictor is None:
            phases = self.phase_predictor(latent) # [bf, num_fourier_feats * num_feats_per_freq]
            raw_pos_embs = raw_pos_embs + phases * phase_scales # [bf, num_fourier_feats * num_feats_per_freq]
        else:
            phases = None

        if self.training and self.cfg.time_enc.phase_dropout_std > 0.0:
            phase_noise = torch.randn(batch_size, self.freqs.shape[1], device=t.device).repeat_interleave(num_frames, dim=0) # [bf, num_fourier_feats]
            phase_noise = phase_noise.repeat_interleave(self.num_feats_per_freq, dim=1) # [bf, num_forier_feats * num_feats_per_freq]
            raw_pos_embs = raw_pos_embs + self.cfg.time_enc.phase_dropout_std * phase_noise * phase_scales # [bf, num_fourier_feats * num_feats_per_freq]

        if self.use_cosine:
            pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1) # [bf, num_fourier_feats * num_feats_per_freq * 2]
        else:
            pos_embs = raw_pos_embs.sin() # [bf, num_fourier_feats * num_feats_per_freq]

        amplitudes = self.amplitudes_predictor(latent) # [bf, output_dim]
        time_embs = amplitudes * pos_embs # [bf, output_dim]

        if not self.weights is None:
            time_embs = time_embs * self.weights.repeat(1, 2) # [1, output_dim]

        if return_coefs:
            return {
                'periods': periods,
                'phases': phases,
                'amplitudes': amplitudes,
                'time_embs': time_embs,
            }
        else:
            return time_embs

#----------------------------------------------------------------------------

@persistence.persistent_class
class VariableInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, w_dim: int):
        super().__init__()

        self.cfg = cfg
        self.repeat = self.cfg.input['repeat']
        self.channel_dim = channel_dim

        # Const input
        self.const = nn.Parameter(torch.randn(1, channel_dim, self.cfg.input.resolution, self.cfg.input.resolution))
        fc_output_dim = channel_dim if self.repeat else self.const.numel() # [1]
        self.fc = FullyConnectedLayer(w_dim, fc_output_dim, activation='lrelu')

    def get_total_dim(self):
        return self.channel_dim * 2

    def forward(self, w: Tensor) -> Tensor:
        const_part = self.const.repeat(len(w), 1, 1, 1) # [batch_size, channel_dim, h, w]

        if self.repeat:
            res = self.cfg.input.resolution
            var_part = self.fc(w).unsqueeze(2).unsqueeze(3).repeat(1, 1, res, res) # [batch_size, channel_dim, h, w]
        else:
            var_part = self.fc(w).view(len(w), *self.const.shape[1:]) # [batch_size, channel_dim, h, w]

        return torch.cat([const_part, var_part], dim=1) # [batch_size, channel_dim * 2, h, w]

#----------------------------------------------------------------------------

@persistence.persistent_class
class TemporalInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, w_dim: int, motion_w_dim: int):
        super().__init__()

        self.cfg = cfg
        self.motion_w_dim = motion_w_dim

        # Const input
        if self.cfg.input.get('has_const', True):
            self.const = nn.Parameter(torch.randn(1, channel_dim, self.cfg.input.resolution, self.cfg.input.resolution))
        else:
            self.const = None

        # Variable input
        if self.cfg.input.get('has_variable_input', False):
            self.repeat = self.cfg.input['var_repeat']
            fc_output_dim = channel_dim if self.repeat else channel_dim * self.cfg.input.resolution ** 2 # [1]
            self.fc = FullyConnectedLayer(w_dim, fc_output_dim, activation='lrelu')
        else:
            self.fc = None

    def get_total_dim(self):
        total_dim = self.motion_w_dim
        total_dim += 0 if self.const is None else self.const.shape[1]
        total_dim += 0 if self.fc is None else self.const.shape[1]

        return total_dim

    def forward(self, t: Tensor, motion_w: Tensor, w: Tensor=None) -> Tensor:
        """
        motion_w: [batch_size, motion_w_dim]
        """
        out = torch.cat([
            self.const.repeat(len(motion_w), 1, 1, 1),
            motion_w.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.cfg.input.resolution, self.cfg.input.resolution),
        ], dim=1) # [batch_size, channel_dim + num_fourier_feats * 2]

        if not self.fc is None:
            if self.repeat:
                res = self.cfg.input.resolution
                var_part = self.fc(w).unsqueeze(2).unsqueeze(3).repeat(1, 1, res, res) # [batch_size, channel_dim, h, w]
            else:
                var_part = self.fc(w).view(len(w), -1, *out.shape[2:]) # [batch_size, channel_dim, h, w]

            out = torch.cat([out, var_part], dim=1) # [batch_size, channel_dim + num_fourier_feats * 2]

        return out

#----------------------------------------------------------------------------

class TemporalDifferenceEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        if self.cfg.num_frames_per_sample > 1:
            self.d = 256
            self.const_embed = nn.Embedding(self.cfg.max_num_frames, self.d)
            self.time_encoder = FixedTimeEncoder(
                self.cfg.max_num_frames,
                transformer_pe=self.cfg.transformer_pe,
                skip_small_t_freqs=self.cfg.get('skip_small_t_freqs', 0))

    def get_total_dim(self) -> int:
        if self.cfg.num_frames_per_sample == 1:
            return 1
        else:
            if self.cfg.sampling_type == 'uniform':
                return self.d + self.time_encoder.get_dim()
            else:
                return (self.d + self.time_encoder.get_dim()) * (self.cfg.num_frames_per_sample - 1)

    def forward(self, t: Tensor) -> Tensor:
        misc.assert_shape(t, [None, self.cfg.num_frames_per_sample])

        batch_size = t.shape[0]

        if self.cfg.num_frames_per_sample == 1:
            out = torch.zeros(len(t), 1, device=t.device)
        else:
            if self.cfg.sampling_type == 'uniform':
                num_diffs_to_use = 1
                t_diffs = t[:, 1] - t[:, 0] # [batch_size]
            else:
                num_diffs_to_use = self.cfg.num_frames_per_sample - 1
                t_diffs = (t[:, 1:] - t[:, :-1]).view(-1) # [batch_size * (num_frames - 1)]
            # Note: float => round => long is necessary when it's originally long
            const_embs = self.const_embed(t_diffs.float().round().long()) # [batch_size * num_diffs_to_use, d]
            fourier_embs = self.time_encoder(t_diffs.unsqueeze(1)) # [batch_size * num_diffs_to_use, num_fourier_feats]
            out = torch.cat([const_embs, fourier_embs], dim=1) # [batch_size * num_diffs_to_use, d + num_fourier_feats]
            out = out.view(batch_size, num_diffs_to_use, -1).view(batch_size, -1) # [batch_size, num_diffs_to_use * (d + num_fourier_feats)]

        return out

#----------------------------------------------------------------------------

class MultiTimeEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        if self.cfg.num_frames_per_sample > 1:
            self.d = 256
            self.const_embed = nn.Embedding(self.cfg.max_num_frames, self.d)
            self.time_encoder = FixedTimeEncoder(self.cfg.max_num_frames, transformer_pe=self.cfg.transformer_pe, skip_small_t_freqs=self.cfg.get('skip_small_t_freqs', 0))
            self.fc = FullyConnectedLayer(
                in_features=self.cfg.num_frames_per_sample * (self.d + self.time_encoder.get_dim()),
                out_features=self.d,
                activation='lrelu')

    def get_total_dim(self) -> int:
        if self.cfg.num_frames_per_sample == 1:
            return 1
        else:
            return self.d

    def forward(self, t: Tensor) -> Tensor:
        misc.assert_shape(t, [None, self.cfg.num_frames_per_sample])

        batch_size, num_frames = t.shape

        if self.cfg.num_frames_per_sample == 1:
            out = torch.zeros(len(t), 1, device=t.device)
        else:
            t = t.view(-1).long() # [batch_size * num_frames]
            const_embs = self.const_embed(t) # [batch_size * num_frames, d]
            fourier_embs = self.time_encoder(t.float()) # [batch_size * num_frames, num_fourier_feats]
            embs = torch.cat([const_embs, fourier_embs], dim=1) # [batch_size * num_frames, d + num_fourier_feats]
            embs = embs.view(batch_size, num_frames, -1).view(batch_size, -1) # [batch_size, num_frames * (d + num_fourier_feats)]
            out = self.fc(embs).view(batch_size, self.d) # [batch_size, d]

        return out

#----------------------------------------------------------------------------

class JointTimeEncoder(nn.Module):
    """
    Combines both diff-based and multi-time embedders
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.diff_enc = TemporalDifferenceEncoder(cfg)
        self.mult_enc = MultiTimeEncoder(cfg)

    def get_total_dim(self) -> int:
        return self.diff_enc.get_total_dim() + self.mult_enc.get_total_dim()

    def forward(self, t: Tensor) -> Tensor:
        return torch.cat([self.diff_enc(t), self.mult_enc(t)], dim=1) # [batch_size, diff_enc_dim + mult_enc_dim]

#----------------------------------------------------------------------------

@persistence.persistent_class
class CoordFuser(nn.Module):
    """
    CoordFuser which concatenates coordinates across dim=1 (we assume channel_first format)
    """
    def __init__(self, cfg: DictConfig, w_dim: int, resolution: int, t_resolution: int):
        super().__init__()

        self.cfg = cfg
        self.resolution = resolution
        self.res_cfg = self.cfg.res_configs[str(resolution)]
        self.log_emb_size = self.res_cfg.get('log_emb_size', 0)
        self.random_emb_size = self.res_cfg.get('random_emb_size', 0)
        self.shared_emb_size = self.res_cfg.get('shared_emb_size', 0)
        self.predictable_emb_size = self.res_cfg.get('predictable_emb_size', 0)
        self.const_emb_size = self.res_cfg.get('const_emb_size', 0)
        self.t_log_emb_size = self.res_cfg.get('t_log_emb_size', 0)
        self.t_const_emb_size = self.res_cfg.get('t_const_emb_size', 0)
        self.fourier_scale = self.res_cfg.get('fourier_scale', np.sqrt(10))
        self.use_cosine = self.res_cfg.get('use_cosine', False)
        self.use_raw_coords = self.res_cfg.get('use_raw_coords', False)
        self.init_dist = self.res_cfg.get('init_dist', 'randn')
        self._fourier_embs_cache = None
        self._full_cache = None
        self.use_full_cache = cfg.get('use_full_cache', False)

        if self.log_emb_size > 0:
            self.register_buffer('log_basis', generate_logarithmic_basis(
                resolution, self.log_emb_size, use_diagonal=self.res_cfg.get('use_diagonal', False))) # [log_emb_size, 2]

        if self.random_emb_size > 0:
            self.register_buffer('random_basis', self.sample_w_matrix((self.random_emb_size, 2), self.fourier_scale))

        if self.shared_emb_size > 0:
            self.shared_basis = nn.Parameter(self.sample_w_matrix((self.shared_emb_size, 2), self.fourier_scale))

        if self.predictable_emb_size > 0:
            self.W_size = self.predictable_emb_size * self.cfg.coord_dim
            self.b_size = self.predictable_emb_size
            self.affine = FullyConnectedLayer(w_dim, self.W_size + self.b_size, bias_init=0)

        if self.const_emb_size > 0:
            self.const_embs = nn.Parameter(torch.randn(1, self.const_emb_size, resolution, resolution).contiguous())

        if self.t_log_emb_size > 0:
            self.time_encoder = FixedTimeEncoder(t_resolution, transformer_pe=self.cfg.transformer_pe, skip_small_t_freqs=self.cfg.get('skip_small_t_freqs', 0))

        if self.t_const_emb_size > 0:
            raise NotImplementedError("We are going to feed fractional t at test time")
            self.t_embed = nn.Embedding(t_resolution, self.t_const_emb_size)

        self.total_dim = self.get_total_dim()
        self.is_modulated = (self.predictable_emb_size > 0)

    def sample_w_matrix(self, shape: Tuple[int], scale: float):
        if self.init_dist == 'randn':
            return torch.randn(shape) * scale
        elif self.init_dist == 'rand':
            return (torch.rand(shape) * 2 - 1) * scale
        else:
            raise NotImplementedError(f"Unknown init dist: {self.init_dist}")

    def get_total_dim(self) -> int:
        if self.cfg.fallback:
            return 0

        total_dim = 0
        total_dim += (self.cfg.coord_dim if self.use_raw_coords else 0)
        if self.log_emb_size > 0:
            total_dim += self.log_basis.shape[0] * (2 if self.use_cosine else 1)
        total_dim += self.random_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.shared_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.predictable_emb_size * (2 if self.use_cosine else 1)
        total_dim += self.const_emb_size
        if self.t_log_emb_size > 0:
            total_dim += self.time_encoder.get_dim()
        total_dim += self.t_const_emb_size

        return total_dim

    def forward(self, x: Tensor, w: Tensor=None, t: Tensor=None, dtype=None, memory_format=torch.contiguous_format) -> Tensor:
        """
        Dims:
            @arg x is [batch_size, in_channels, img_size, img_size]
            @arg w is [batch_size, w_dim]
            @return out is [batch_size, in_channels + fourier_dim + cips_dim, img_size, img_size]
        """
        assert memory_format is torch.contiguous_format

        if self.cfg.fallback:
            return x

        to_kwargs = dict(dtype=dtype, memory_format=memory_format)
        batch_size, in_channels, img_size = x.shape[:3]
        out = x

        if self.use_full_cache and (not self._full_cache is None) and (self._full_cache.device == x.device) and \
           (self._full_cache.shape == (batch_size, self.get_total_dim(), img_size, img_size)):
           return torch.cat([x, self._full_cache], dim=1)

        if (not self._fourier_embs_cache is None) and (self._fourier_embs_cache.device == x.device) and \
           (self._fourier_embs_cache.shape == (batch_size, self.get_total_dim() - self.const_emb_size, img_size, img_size)):
            out = torch.cat([out, self._fourier_embs_cache], dim=1)
        else:
            raw_embs = []
            raw_coords = generate_coords(batch_size, img_size, x.device, align_corners=True) # [batch_size, coord_dim, img_size, img_size]

            if self.use_raw_coords:
                out = torch.cat([out, raw_coords.to(**to_kwargs)], dim=1)

            if self.log_emb_size > 0:
                log_bases = self.log_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, log_emb_size, 2]
                raw_log_embs = torch.einsum('bdc,bcxy->bdxy', log_bases, raw_coords) # [batch_size, log_emb_size, img_size, img_size]
                raw_embs.append(raw_log_embs)

            if self.random_emb_size > 0:
                random_bases = self.random_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, random_emb_size, 2]
                raw_random_embs = torch.einsum('bdc,bcxy->bdxy', random_bases, raw_coords) # [batch_size, random_emb_size, img_size, img_size]
                raw_embs.append(raw_random_embs)

            if self.shared_emb_size > 0:
                shared_bases = self.shared_basis.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, shared_emb_size, 2]
                raw_shared_embs = torch.einsum('bdc,bcxy->bdxy', shared_bases, raw_coords) # [batch_size, shared_emb_size, img_size, img_size]
                raw_embs.append(raw_shared_embs)

            if self.predictable_emb_size > 0:
                misc.assert_shape(w, [batch_size, None])
                mod = self.affine(w) # [batch_size, W_size + b_size]
                W = self.fourier_scale * mod[:, :self.W_size] # [batch_size, W_size]
                W = W.view(batch_size, self.predictable_emb_size, self.cfg.coord_dim) # [batch_size, predictable_emb_size, coord_dim]
                bias = mod[:, self.W_size:].view(batch_size, self.predictable_emb_size, 1, 1) # [batch_size, predictable_emb_size, 1]
                raw_predictable_embs = (torch.einsum('bdc,bcxy->bdxy', W, raw_coords) + bias) # [batch_size, predictable_emb_size, img_size, img_size]
                raw_embs.append(raw_predictable_embs)

            if len(raw_embs) > 0:
                raw_embs = torch.cat(raw_embs, dim=1) # [batch_suze, log_emb_size + random_emb_size + predictable_emb_size, img_size, img_size]
                raw_embs = raw_embs.contiguous() # [batch_suze, -1, img_size, img_size]
                out = torch.cat([out, raw_embs.sin().to(**to_kwargs)], dim=1) # [batch_size, -1, img_size, img_size]

                if self.use_cosine:
                    out = torch.cat([out, raw_embs.cos().to(**to_kwargs)], dim=1) # [batch_size, -1, img_size, img_size]

        # if self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
        #     self._fourier_embs_cache = out[:, x.shape[1]:].detach()

        if self.t_log_emb_size > 0:
            misc.assert_shape(t, [batch_size])
            t_fourier_embs = self.time_encoder(t).to(**to_kwargs).unsqueeze(2).unsqueeze(3) # [batch_size, num_fourier_feats * 2, 1, 1]
            out = torch.cat([out, t_fourier_embs.repeat(1, 1, img_size, img_size)], dim=1) # [batch_size, -1, img_size, img_size]

        if self.t_const_emb_size > 0:
            misc.assert_shape(t, [batch_size])
            t_const_embs = self.t_embed(t).to(**to_kwargs).unsqueeze(2).unsqueeze(3) # [batch_size, d, 1, 1]
            out = torch.cat([out, t_const_embs.repeat(1, 1, img_size, img_size)], dim=1) # [batch_size, -1, img_size, img_size]

        if self.const_emb_size > 0:
            misc.assert_shape(x, [batch_size, None, self.resolution, self.resolution])
            const_embs = self.const_embs.repeat([batch_size, 1, 1, 1])
            const_embs = const_embs.to(**to_kwargs)
            out = torch.cat([out, const_embs], dim=1) # [batch_size, total_dim, img_size, img_size]

        if self.use_full_cache and self.predictable_emb_size == 0 and self.shared_emb_size == 0 and out.shape[1] > x.shape[1]:
            self._full_cache = out[:, x.shape[1]:].detach()

        return out

#----------------------------------------------------------------------------

class TimeFuser(nn.Module):
    """
    Concatenates time embeddings to a hidden representation
    We use embeddings of different frequency depending on the block resolution
    """
    def __init__(self, cfg: DictConfig, resolution: int):
        super().__init__()

        self.cfg = cfg
        assert self.cfg.time_enc.input_slowest_period_len <= self.cfg.motion.max_period_len, \
            f"Wrong input slowest period len: {self.cfg.time_enc.input_slowest_period_len} vs {self.cfg.motion.max_period_len}"
        # We are altering only those frequencies, which period is <= self.cfg.time_enc.input_slowest_period_len
        if self.cfg.time_enc.input_slowest_period_len > 0:
            num_freqs_to_alter = construct_log_spaced_freqs(self.cfg.time_enc.input_slowest_period_len).shape[1] # [1]
            num_freqs_to_preserve = int(np.log2(resolution).item()) - 1
            num_small_freqs_to_skip = max(num_freqs_to_alter - num_freqs_to_preserve, 0)
        else:
            num_small_freqs_to_skip = 0

        # Computing fourier coefs
        self.time_encoder = construct_time_encoder(self.cfg, resolution=resolution)

    def get_total_dim(self) -> int:
        return self.time_encoder.get_dim()

    def forward(self, x: Tensor, t: Tensor, motion_w: Tensor):
        time_embs = self.time_encoder(t, latent=motion_w) # [batch_size * num_frames, time_emb_dim * 2]
        time_embs = time_embs.unsqueeze(2).unsqueeze(3).repeat(1, 1, *x.shape[2:]) # [batch_size * num_frames, time_emb_dim * 2, h, w]
        out = torch.cat([x, time_embs.to(x.dtype)], dim=1) # [batch_size, channel_dim + num_fourier_feats * 2]

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class EqLRConv1d(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: int=0,
        stride: int=1,
        activation: str='linear',
        lr_multiplier: float=1.0,
        bias=True,
        bias_init=0.0,
    ):
        super().__init__()

        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features, kernel_size]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features * kernel_size)
        self.bias_gain = lr_multiplier
        self.padding = padding
        self.stride = stride

        assert self.activation in ['lrelu', 'linear']

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, f"Wrong shape: {x.shape}"

        w = self.weight.to(x.dtype) * self.weight_gain # [out_features, in_features, kernel_size]
        b = self.bias # [out_features]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        y = F.conv1d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding) # [batch_size, out_features, out_len]
        if self.activation == 'linear':
            pass
        elif self.activation == 'lrelu':
            y = F.leaky_relu(y, negative_slope=0.2) # [batch_size, out_features, out_len]
        else:
            raise NotImplementedError
        return y

#----------------------------------------------------------------------------

# class TemporalConvolutionalAttentionBlock(nn.Module):
#     def __init__(self, cfg, in_channels: int):
#         super().__init__()

#         self.cfg = cfg

#     def forward(self, x: Tensor, t: Tensor) -> Tensor:
#         """
#         Params:
#             - x of shape [batch_size, num_frames, c, h, w]
#             - t of shape [batch_size, 1]
#         Returns:
#             - y of shape [batch_size, num_frames, c, h, w]
#         """

#         attn_weights = 123

#----------------------------------------------------------------------------

def generate_coords(batch_size: int, img_size: int, device='cpu', align_corners: bool=False) -> Tensor:
    """
    Generates the coordinates in [-1, 1] range for a square image
    if size (img_size x img_size) in such a way that
    - upper left corner: coords[0, 0] = (-1, -1)
    - upper right corner: coords[img_size - 1, img_size - 1] = (1, 1)
    """
    if align_corners:
        row = torch.linspace(-1, 1, img_size, device=device).float() # [img_size]
    else:
        row = (torch.arange(0, img_size, device=device).float() / img_size) * 2 - 1 # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size, img_size).repeat(batch_size, 1, 1, 1) # [batch_size, 2, img_size, img_size]

    return coords

#----------------------------------------------------------------------------

def generate_logarithmic_basis(
    resolution: int,
    max_num_feats: int=float('inf'),
    remove_lowest_freq: bool=False,
    use_diagonal: bool=True) -> Tensor:
    """
    Generates a directional logarithmic basis with the following directions:
        - horizontal
        - vertical
        - main diagonal
        - anti-diagonal
    """
    max_num_feats_per_direction = np.ceil(np.log2(resolution)).astype(int)
    bases = [
        generate_horizontal_basis(max_num_feats_per_direction),
        generate_vertical_basis(max_num_feats_per_direction),
    ]

    if use_diagonal:
        bases.extend([
            generate_diag_main_basis(max_num_feats_per_direction),
            generate_anti_diag_basis(max_num_feats_per_direction),
        ])

    if remove_lowest_freq:
        bases = [b[1:] for b in bases]

    # If we do not fit into `max_num_feats`, then trying to remove the features in the order:
    # 1) anti-diagonal 2) main-diagonal
    # while (max_num_feats_per_direction * len(bases) > max_num_feats) and (len(bases) > 2):
    #     bases = bases[:-1]

    basis = torch.cat(bases, dim=0)

    # If we still do not fit, then let's remove each second feature,
    # then each third, each forth and so on
    # We cannot drop the whole horizontal or vertical direction since otherwise
    # model won't be able to locate the position
    # (unless the previously computed embeddings encode the position)
    # while basis.shape[0] > max_num_feats:
    #     num_exceeding_feats = basis.shape[0] - max_num_feats
    #     basis = basis[::2]

    assert basis.shape[0] <= max_num_feats, \
        f"num_coord_feats > max_num_fixed_coord_feats: {basis.shape, max_num_feats}."

    return basis

#----------------------------------------------------------------------------

def generate_horizontal_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [0.0, 1.0], 4.0)


def generate_vertical_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0, 0.0], 4.0)


def generate_diag_main_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))


def generate_anti_diag_basis(num_feats: int) -> Tensor:
    return generate_wavefront_basis(num_feats, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], 4.0 * np.sqrt(2))

#----------------------------------------------------------------------------

def generate_wavefront_basis(num_feats: int, basis_block: List[float], period_length: float) -> Tensor:
    period_coef = 2.0 * np.pi / period_length
    basis = torch.tensor([basis_block]).repeat(num_feats, 1) # [num_feats, 2]
    powers = torch.tensor([2]).repeat(num_feats).pow(torch.arange(num_feats)).unsqueeze(1) # [num_feats, 1]
    result = basis * powers * period_coef # [num_feats, 2]

    return result.float()


#----------------------------------------------------------------------------

def sample_frames(cfg: Dict, total_video_len: int, **kwargs) -> np.ndarray:
    if cfg['type'] == 'random':
        return random_frame_sampling(cfg, total_video_len, **kwargs)
    elif cfg['type'] == 'uniform':
        return uniform_frame_sampling(cfg, total_video_len, **kwargs)
    else:
        raise NotImplementedError

#----------------------------------------------------------------------------

def random_frame_sampling(cfg: Dict, total_video_len: int, max_time_diff: int=float('inf'), use_fractional_t: bool=False) -> np.ndarray:
    min_time_diff = cfg["num_frames_per_sample"] - 1
    max_time_diff = min(total_video_len - 1, cfg.get('max_dist', float('inf')), max_time_diff)

    if type(cfg.get('total_dists')) in (list, tuple):
        time_diff_range = [d for d in cfg['total_dists'] if min_time_diff <= d <= max_time_diff]
    else:
        time_diff_range = range(min_time_diff, max_time_diff)

    time_diff: int = random.choice(time_diff_range)
    if use_fractional_t:
        offset = random.random() * (total_video_len - time_diff - 1)
    else:
        offset = random.randint(0, total_video_len - time_diff - 1)
    frames_idx = [offset]

    if cfg["num_frames_per_sample"] > 1:
        frames_idx.append(offset + time_diff)

    if cfg["num_frames_per_sample"] > 2:
        frames_idx.extend([(offset + t) for t in random.sample(range(1, time_diff), k=cfg["num_frames_per_sample"] - 2)])

    frames_idx = sorted(frames_idx)

    return np.array(frames_idx)

#----------------------------------------------------------------------------

def uniform_frame_sampling(cfg: Dict, total_video_len: int, max_time_diff: int=float('inf'), use_fractional_t: bool=False) -> np.ndarray:
    # Step 1: Select the distance between frames
    if type(cfg.get('dists')) in (list, tuple):
        valid_dists = [d for d in cfg['dists'] if (d * cfg['num_frames_per_sample'] - d + 1) <= min(total_video_len, max_time_diff)]
        d = random.choice(valid_dists)
    else:
        max_dist = min(cfg.get('max_dist', float('inf')), total_video_len // cfg['num_frames_per_sample'], max_time_diff // cfg['num_frames_per_sample'])
        d = random.randint(1, max_dist)

    d_total = d * cfg['num_frames_per_sample'] - d + 1

    # Step 2: Sample.
    if use_fractional_t:
        offset = random.random() * (total_video_len - d_total)
    else:
        offset = random.randint(0, total_video_len - d_total)

    frames_idx = offset + np.arange(cfg['num_frames_per_sample']) * d

    return frames_idx

#----------------------------------------------------------------------------

def construct_log_spaced_freqs(max_num_frames: int, skip_small_t_freqs: int=0) -> Tuple[int, Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_num_frames))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.unsqueeze(0).float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution

#----------------------------------------------------------------------------

def remove_diag(x: Tensor) -> Tensor:
    """
    Takes an input of size [..., n, n] and returns a tensor of size [..., n, n-1],
    where we removed diagonal element from the square matrix at the end

    Based on https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
    """
    assert x.shape[-1] == x.shape[-2], f"The end dims should be a matrix: {x.shape}"

    core_shape = x.shape[:-2] # The shape which is not fixed during the manipulations
    n = x.shape[-1] # Matrix size
    y = x.reshape(*core_shape, -1)[..., 1:] # Removed the very first element
    y = y.reshape(*core_shape, n-1, n+1) # Reshaped in such a way that diagonal elements are at the end
    y = y[..., :-1] # Removed the diagonal elements
    y = y.reshape(*core_shape, n, n-1) # Reshaped to the proper size

    return y

#----------------------------------------------------------------------------

def get_max_dist(sampling_cfg: DictConfig) -> int:
    """
    Computes the maximum distance between frames
    """
    if sampling_cfg.type == 'random':
        pass
    elif sampling_cfg.type == 'uniform':
        if type(sampling_cfg.get('dists')) in (list, tuple):
            max_dist = max([d for d in sampling_cfg.dists])
        else:
            max_dist = min(sampling_cfg.get('max_dist', float('inf')), sampling_cfg.max_num_frames)
    else:
        raise NotImplementedError(f"Unknown sampling type: {sampling_cfg.type}")

    return max_dist

#----------------------------------------------------------------------------

def ema(x: Tensor, alpha, dim: int=-1):
    """
    Adapted / copy-pasted from https://stackoverflow.com/a/42926270
    """
    #alpha = 2 / (window + 1.0)
    assert 0.0 <= alpha < 1.0, f"Bad alpha value: {alpha}. It should be in [0, 1)"
    assert dim == -1, f"Not implemented for dim: {dim}"
    assert x.size(dim) <= 1024, f"Too much points for a vectorized implementation: {x.shape}"

    alpha_rev = 1.0 - alpha # [1]
    num_points = x.size(dim) # [1]
    pows = alpha_rev ** (torch.arange(num_points + 1, device=x.device)) # [num_points + 1]
    scale_arr = 1 / pows[:-1].double() # [num_points]

    # Note: broadcast from the last dimension
    offset = x[..., [0]] * pows[1:] # [..., num_points]
    pw0 = alpha * (alpha_rev ** (num_points - 1)) # [1]
    cumsums = (x * pw0 * scale_arr).to(x.dtype).cumsum(dim=dim) # [..., num_points]
    x_ema = offset + (cumsums * scale_arr.flip(0)).to(x.dtype) # [..., num_points]

    return x_ema

#----------------------------------------------------------------------------

def construct_frequencies(num_freqs: int, min_period_len: int, max_period_len: int) -> Tensor:
    freqs = 2 * np.pi / (2 ** np.linspace(np.log2(min_period_len), np.log2(max_period_len), num_freqs)) # [num_freqs]
    freqs = torch.from_numpy(freqs[::-1].copy().astype(np.float32)).unsqueeze(0) # [1, num_freqs]

    return freqs

#----------------------------------------------------------------------------

def compute_progressive_weights(num_freqs: int, curr_iter: int, max_num_iters: int, num_opened_dims: int, smoothness: float=0.025):
    """
    Adapted from https://nerfies.github.io/, but we use different smoothing
    """
    alpha = (num_freqs * curr_iter) / max_num_iters # [1]
    dim_idx = np.arange(num_freqs) # [num_freqs]
    weights = (1 - np.cos(np.clip(smoothness * np.pi * (alpha - dim_idx + num_opened_dims), 0, np.pi))) / 2.0 # [num_freqs]
    weights[:num_opened_dims] = 1.0

    return weights.astype(np.float32)
