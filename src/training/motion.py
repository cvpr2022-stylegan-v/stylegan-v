from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
import torch.nn.functional as F

from src.torch_utils import misc
from src.torch_utils import persistence
from src.training.layers import (
    CoordFuser,
    Conv2dLayer,
    generate_coords,
    MappingNetwork,
    FixedTimeEncoder,
    construct_time_encoder,
    FullyConnectedLayer,
    ema,
    EqLRConv1d,
    PredictableTimeEncoder,
)

#----------------------------------------------------------------------------

@persistence.persistent_class
class MotionEncoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, resolutions: List[int]=None):
        super().__init__()
        self.cfg = cfg

        assert self.cfg.motion.gen_strategy in ["shuffled_embs", "static", "autoregressive", "randn", "conv"], f"Unknown generation strategy: {self.cfg.motion.gen_strategy}"
        assert self.cfg.motion.num_levels == 1, "We dropped the possibility of using hierarchical motion codes to simplify the code."

        if self.cfg.use_video_len_cond:
            self.l_time_encoder = FixedTimeEncoder(self.cfg.motion.max_period_len, transformer_pe=self.cfg.transformer_pe)
        else:
            self.l_time_encoder = None

        if self.cfg.motion.fourier:
            assert not self.cfg.time_enc.per_resolution, "Not implemented"
            self.time_encoder = construct_time_encoder(G_cfg=self.cfg, resolution=None)
            self.mapping = None
        else:
            self.time_encoder = None
            total_z_dim = self.cfg.motion.z_dim * self.cfg.motion.num_levels + self.cfg.motion.get('static_emb_dim', 0)
            total_z_dim += (self.cfg.w_dim if self.cfg.motion.w_conditioning else 0)
            self.mapping = MappingNetwork(
                z_dim=total_z_dim,
                c_dim=self.cfg.c_dim,
                w_dim=self.cfg.motion.w_dim,
                num_ws=None,
                num_layers=2,
                activation='lrelu',
                w_avg_beta=None,
                cfg=self.cfg,
            )

        self.num_frames_per_motion = [self.cfg.motion.start_fpm * self.cfg.motion.fpm_base ** lvl for lvl in range(self.cfg.motion.num_levels)] # [num_levels]

        if self.cfg.motion.gen_strategy == 'shuffled_embs':
            self.embeds = nn.ModuleDict({str(nfpm): nn.Embedding(self.cfg.motion.num_codes, self.cfg.motion.z_dim, max_norm=np.sqrt(self.cfg.motion.z_dim)) for nfpm in self.num_frames_per_motion})
        elif self.cfg.motion.gen_strategy == 'static':
            max_lens = [((self.cfg.max_num_frames + nfpm - 1) // nfpm) + 2 for nfpm in self.num_frames_per_motion]
            self.embeds = nn.ModuleDict({str(nfpm): nn.Embedding(l, self.cfg.motion.z_dim, max_norm=np.sqrt(self.cfg.motion.z_dim)) for l, nfpm in zip(max_lens, self.num_frames_per_motion)})
        elif self.cfg.motion.gen_strategy == 'autoregressive':
            video_len_dim = 0 if self.l_time_encoder is None else self.l_time_encoder.get_dim()
            self.rnn = nn.LSTM(
                input_size=self.cfg.motion.z_dim + video_len_dim + self.cfg.c_dim,
                hidden_size=self.cfg.motion.z_dim,
                bidirectional=False,
                batch_first=True)
            self._parameters_flattened = False
        elif self.cfg.motion.gen_strategy == 'randn':
            pass
        elif self.cfg.motion.gen_strategy == 'conv':
            video_len_dim = 0 if self.l_time_encoder is None else self.l_time_encoder.get_dim()
            input_dim = self.cfg.motion.z_dim + self.cfg.c_dim + video_len_dim + (self.cfg.w_dim if self.cfg.motion.w_conditioning else 0)

            self.conv = nn.Sequential(
                EqLRConv1d(input_dim, self.cfg.motion.z_dim, self.cfg.motion.kernel_size, padding=0, activation='lrelu', lr_multiplier=0.01),
                EqLRConv1d(self.cfg.motion.z_dim, self.cfg.motion.w_dim, self.cfg.motion.kernel_size, padding=0, activation='lrelu', lr_multiplier=0.01),
            )
            self.num_additional_codes = (self.cfg.motion.kernel_size - 1) * 2

        if self.cfg.motion.w_conditioning and self.cfg.motion.gen_strategy == "autoregressive":
            self.w_to_h = FullyConnectedLayer(self.cfg.w_dim, self.cfg.motion.z_dim, activation='lrelu')
            self.w_to_c = FullyConnectedLayer(self.cfg.w_dim, self.cfg.motion.z_dim, activation='lrelu')

    def progressive_update(self, curr_kimg: int):
        if not self.time_encoder is None:
            self.time_encoder.progressive_update(curr_kimg)

    def generate_motion(self, c: Tensor, t: Tensor, l: Tensor, num_frames_per_motion: int, w: Tensor=None, motion_noise: Tensor=None) -> Dict:
        """
        Arguments:
            - c of shape [batch_size, c_dim]
            - t of shape [batch_size, num_frames]
            - num_frames_per_motion: int
            - w of shape [batch_size, w_dim]
        """
        out = {}
        batch_size, num_frames = t.shape

        # Consutruct trajectories (from code idx for now)
        # TODO: construct batch-wise to save computation
        # max_traj_len = np.ceil((traj_t_max - traj_t_min + 1e-8) / num_frames_per_motion).astype(int) + 1 # [1]
        max_t = max(self.cfg.max_num_frames - 1, t.max().item()) # [1]
        max_traj_len = np.ceil(max_t / num_frames_per_motion).astype(int).item() + 2 # [1]

        if self.cfg.motion.gen_strategy == 'conv':
            max_traj_len += self.num_additional_codes # [1]

        if self.cfg.motion.gen_strategy == 'shuffled_embs':
            if motion_noise is None:
                traj_codes_idx = torch.randint(
                    low=0, high=self.cfg.motion.num_codes,
                    size=(batch_size, max_traj_len),
                    device=c.device) # [batch_size, max_traj_len]
                out['motion_noise'] = traj_codes_idx
            else:
                out['motion_noise'] = motion_noise
                traj_codes_idx = motion_noise[:batch_size, :max_traj_len] # [batch_size, max_traj_len]

        if self.cfg.motion.gen_strategy == 'static':
            traj_codes_idx = torch.arange(max_traj_len, device=c.device).unsqueeze(0).repeat(batch_size, 1) # [batch_size, max_traj_len]
            assert motion_noise is None
            out['motion_noise'] = None

        # Now, we should select neighbouring codes for each frame
        left_idx = (t / num_frames_per_motion).floor().long() # [batch_size, num_frames]
        batch_idx = torch.arange(batch_size, device=c.device).unsqueeze(1).repeat(1, num_frames) # [batch_size, num_frames]

        if self.cfg.motion.gen_strategy in ['shuffled_embs', 'static']:
            left_codes = self.embeds[str(num_frames_per_motion)](traj_codes_idx[batch_idx, left_idx]) # [batch_size, num_frames, motion.z_dim]
            right_codes = self.embeds[str(num_frames_per_motion)](traj_codes_idx[batch_idx, left_idx + 1]) # [batch_size, num_frames, motion.z_dim]
            out['full_trajs'] = None
        elif self.cfg.motion.gen_strategy in ('randn', 'autoregressive', 'conv'):
            if motion_noise is None:
                full_trajs = torch.randn(batch_size, max_traj_len, self.cfg.motion.z_dim, device=c.device) # [batch_size, max_traj_len, motion.z_dim]
                out['motion_noise'] = full_trajs
            else:
                out['motion_noise'] = motion_noise
                full_trajs = motion_noise[:batch_size, :max_traj_len, :self.cfg.motion.z_dim].to(c.device) # [batch_size, max_traj_len, motion.z_dim]

            if self.cfg.motion.gen_strategy in ('conv', 'autoregressive'):
                # Construct the conditioning for LSTM
                # We would like to condition it on video lens and c
                cond = torch.zeros(batch_size * num_frames, 0, device=c.device) # [bf, 0]
                if not self.l_time_encoder is None:
                    cond = self.l_time_encoder(l.unsqueeze(1)).repeat_interleave(t.shape[1], dim=0) # [bf, num_fourier_feats]

                if self.cfg.c_dim > 0:
                    # Different classes have different motions, so it should be useful to condition on c
                    cond = torch.cat([cond, c.repeat_interleave(t.shape[1], dim=0)], dim=1) # [bf, num_fourier_feats + c_dim]

                if self.cfg.motion.w_conditioning and self.cfg.motion_gen_strategy != 'autoregressive':
                    cond = torch.cat([cond, w.repeat_interleave(t.shape[1], dim=0)], dim=1) # [bf, num_fourier_feats + c_dim + w_dim]

                cond = cond.view(t.shape[0], t.shape[1], -1)[:, 0, :] # [batch_size, cond_dim]
                cond = cond.unsqueeze(1).repeat(1, max_traj_len, 1) # [batch_size, max_traj_len, cond_dim]
                full_trajs = torch.cat([full_trajs, cond], dim=2) # [batch_size, max_traj_len, motion.z_dim + cond_dim]

            if self.cfg.motion.gen_strategy == 'autoregressive':
                if not self._parameters_flattened:
                    self.rnn.flatten_parameters()
                    self._parameters_flattened = True

                if self.cfg.motion.w_conditioning:
                    h_0 = self.w_to_h(w) # [batch_size, rnn_dim]
                    c_0 = self.w_to_c(w) # [batch_size, rnn_dim]
                    # LSTM expects initial state input in the [num_layers, batch_size, rnn_dim] format
                    initial_state = (h_0.unsqueeze(0), c_0.unsqueeze(0))
                else:
                    initial_state = None
                full_trajs, _ = self.rnn(full_trajs, initial_state) # [batch_size, max_traj_len, motion.z_dim]
            elif self.cfg.motion.gen_strategy == 'conv':
                full_trajs = self.conv(full_trajs.permute(0, 2, 1)).permute(0, 2, 1) # [batch_size, max_traj_len, motion.w_dim]
            else:
                pass

            out['full_trajs'] = full_trajs # [batch_size, max_traj_len, motion.z_dim]
            left_codes = full_trajs[batch_idx, left_idx] # [batch_size, num_frames, motion.z_dim]
            right_codes = full_trajs[batch_idx, left_idx + 1] # [batch_size, num_frames, motion.z_dim]
        else:
            raise NotImplemented

        interp_weights = ((t % num_frames_per_motion) / num_frames_per_motion).unsqueeze(2).to(torch.float32) # [batch_size, num_frames, 1]
        motion_z = left_codes * (1 - interp_weights) + right_codes * interp_weights # [batch_size, num_frames, motion.z_dim]
        out['motion_z'] = motion_z.view(batch_size * num_frames, motion_z.shape[2]).to(torch.float32) # [batch_size * num_frames, motion.z_dim]

        return out

    def get_output_dim(self) -> int:
        if self.time_encoder is None:
            return self.cfg.motion.w_dim
        else:
            return self.time_encoder.get_dim()

    def forward(self, c: Tensor, t: Tensor, l: Tensor, w: Tensor=None, motion_noise: Dict=None, return_time_embs_coefs: bool=None) -> Dict:
        assert len(c) == len(t) == len(l) == len(w), f"Wrong shape: {c.shape}, {t.shape}, {l.shape}, {w.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        batch_size, num_frames = t.shape
        out = {}
        motion_info = self.generate_motion(c, t, l, self.num_frames_per_motion[0], w=w, motion_noise=motion_noise)
        motion_z = motion_info['motion_z'].view(t.shape[0] * t.shape[1], -1) # [batch_size * num_frames, motion.z_dim]

        if self.cfg.motion.w_conditioning and not self.cfg.motion.gen_strategy in ('autoregressive', 'conv'):
            motion_z = torch.cat([motion_z, w.repeat_interleave(t.shape[1], dim=0)], dim=1) # [batch_size * num_frames, motion.z_dim + time_dim + static_emb_dim + w_dim]

        # Aggregate the motion
        if not self.mapping is None:
            mapping_kwargs = dict(c=c.repeat_interleave(t.shape[1], dim=0), l=l.repeat_interleave(t.shape[1], dim=0))
            motion_z = self.mapping(motion_z, **mapping_kwargs) # [batch_size * num_frames, motion.w_dim]

        if not self.time_encoder is None:
            time_enc_out = self.time_encoder(t=t, latent=motion_z, return_coefs=return_time_embs_coefs)

            if return_time_embs_coefs:
                out = {**time_enc_out, **out}
                motion_w = time_enc_out['time_embs']
            else:
                motion_w = time_enc_out
        else:
            motion_w = motion_z

        out['motion_w'] = motion_w # [batch_size * num_frames, motion.w_dim]
        out['motion_noise'] = motion_info['motion_noise'] # (Any shape)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class MotionCodesMixer(nn.Module):
    """
    Linear interpolation might be suboptimal. This module is designed to mix left/right codes through an MLP
    """
    def __init__(self, code_dim: int, interp_weights_resolution: int, c_dim: int, G_cfg: DictConfig={}):
        super().__init__()

        self.c_dim = c_dim
        self.interp_encoder = FixedTimeEncoder(interp_weights_resolution, transformer_pe=G_cfg.transformer_pe)
        self.mapping = MappingNetwork(
            z_dim=code_dim * 2 + self.interp_encoder.get_dim(),
            c_dim=c_dim,
            w_dim=code_dim,
            num_ws=None,
            num_layers=2,
            activation='lrelu',
            w_avg_beta=None,
            cfg=G_cfg,
        )

    def forward(self, left_codes: Tensor, right_codes: Tensor, interp_weights: Tensor, c: Tensor, l: Tensor=None) -> Tensor:
        """
        Shapes:
            - left_codes, right_codes: [batch_size, num_frames, code_dim]
            - interp_weights: [batch_size, num_frames, 1]
            - c: [batch_size * num_frames, c_dim]
        """
        batch_size, num_frames, code_dim = left_codes.shape

        misc.assert_shape(right_codes, [batch_size, num_frames, code_dim])
        misc.assert_shape(interp_weights, [batch_size, num_frames, 1])
        misc.assert_shape(c, [batch_size * num_frames, self.c_dim])

        left_codes = left_codes.view(batch_size * num_frames, code_dim) # [batch_size * num_frames, code_dim]
        right_codes = right_codes.view(batch_size * num_frames, code_dim) # [batch_size * num_frames, code_dim]
        interp_weights = interp_weights.view(batch_size * num_frames) # [batch_size * num_frames]
        interp_weights = self.interp_encoder(interp_weights * self.interp_encoder.t_resolution) # [batch_size * num_frames, t_dim]
        mixed_codes = self.mapping(z=torch.cat([left_codes, right_codes, interp_weights], dim=1), c=c, l=l) # [batch_size * num_frames, code_dim]

        return mixed_codes

#----------------------------------------------------------------------------

@persistence.persistent_class
class Decoder(nn.Module):
    """
    Simple convolutional decoder
    """
    def __init__(self,
        in_resolution: int,
        out_resolution: int,
        in_channels: int,
        out_channels: int,
        out_activation: str='linear',
        channel_base: int= 32768,
        channel_max: int=512,
        **conv_kwargs):

        super().__init__()

        assert in_resolution % 2 == 0 or in_resolution == 1
        assert out_resolution % 2 == 0
        assert in_resolution <= out_resolution

        self.in_channels = in_channels
        self.in_resolution = in_resolution

        in_res_log2 = int(np.log2(in_resolution))
        out_res_log2 = int(np.log2(out_resolution))
        block_resolutions = [2 ** i for i in range(in_res_log2, out_res_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in block_resolutions}
        channels_dict = {**channels_dict, **{in_resolution: in_channels}, **{out_resolution: out_channels}}
        layers = []

        for res in block_resolutions[:-1]:
            layers.append(Conv2dLayer(
                in_channels=channels_dict[res],
                out_channels=channels_dict[res * 2],
                kernel_size=3,
                activation=(out_activation if res == out_resolution else 'lrelu'),
                up=2,
                **conv_kwargs
            ))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        misc.assert_shape(x, [None, self.in_channels, self.in_resolution, self.in_resolution])

        return self.model(x) # [bn, out_channels, out_resolution, out_resolution]

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowSynthesisNetwork(nn.Module):
    """
    Predicts optical flow that is happening between motion codes.

    TODO: occlusion
    TODO: backward motion
    """
    def __init__(self, cfg: DictConfig, content_feat_dim: int):
        super().__init__()

        self.cfg = cfg
        self.motion_feats_generator = Decoder(
            in_resolution=1,
            out_resolution=self.cfg.flow_synthesis.motion_gen_res,
            in_channels=self.cfg.w_dim + self.cfg.motion.z_dim,
            out_channels=256,
            out_activation='lrelu',
            instance_norm=True,
        )
        self.coord_fuser = CoordFuser(
            cfg=self.cfg.flow_synthesis.coords,
            w_dim=self.cfg.w_dim,
            resolution=self.cfg.flow_synthesis.motion_gen_res,
            t_resolution=self.cfg.max_num_frames)
        self.motion_feats_to_shifts = nn.Sequential(
            Conv2dLayer(in_channels=512 + self.coord_fuser.get_total_dim(), out_channels=256, kernel_size=3, activation='lrelu', instance_norm=True),
            Conv2dLayer(in_channels=256, out_channels=2, kernel_size=1, activation='linear', bias=False, lr_multiplier=1.0 / np.sqrt(content_feat_dim)),
        )

    def forward(self, w: Tensor, t: Tensor, motion_info: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Arguments:
        - w: [batch_size * num_frames, num_ws, w_dim]
        - t: [batch_size * num_frames, motion.z_dim]
        - motion_codes: [batch_size * num_frames]
        """
        left_input = torch.cat([w, motion_info['motion_codes_left']], dim=1).unsqueeze(2).unsqueeze(3) # [batch_size * num_frames, w_dim + motion.z_dim, 1, 1]
        right_input = torch.cat([w, motion_info['motion_codes_right']], dim=1).unsqueeze(2).unsqueeze(3) # [batch_size * num_frames, w_dim + motion.z_dim, 1, 1]
        motion_feats_left = self.motion_feats_generator(left_input) # [batch_size * num_frames, c, h, w]
        motion_feats_right = self.motion_feats_generator(right_input) # [batch_size * num_frames, c, h, w]
        motion_feats = torch.cat([motion_feats_left, motion_feats_right], dim=1) # [batch_size * num_frames, c * 2, h, w]
        motion_feats = self.coord_fuser(motion_feats, w=w, t=t) # [batch_size * num_frames, c * 2 + coord_dim, h, w]
        shifts = self.motion_feats_to_shifts(motion_feats) # [bn, 2, h, w]
        interp_weights = motion_info['interp_weights'].unsqueeze(2).unsqueeze(3) # [batch_size * num_frames, 1, 1, 1]
        optical_flow_unnorm = shifts.permute(0, 2, 3, 1) # [bn, h, w, 2]
        norms = optical_flow_unnorm.norm(float('inf'), dim=(1, 2), keepdim=True) # [bn, 1, 1, 2]
        dcoefs = norms.clamp(1.0, float('inf')) # [bn, 1, 1, 2]
        optical_flow = optical_flow_unnorm / (dcoefs + 1e-8) # [bn, h, w, 2]
        optical_flow = interp_weights * optical_flow # [bn, h, w, 2]

        return optical_flow

#----------------------------------------------------------------------------

def apply_optical_flow(content_feats: Tensor, optical_flow: Tensor) -> Tensor:
    misc.assert_shape(content_feats, [optical_flow.shape[0], None, optical_flow.shape[1], optical_flow.shape[2]])

    bn, img_size = optical_flow.shape[0], optical_flow.shape[1]
    grid = generate_coords(bn, img_size, device=content_feats.device, align_corners=True) # [bn, 2, h, w]
    grid = grid.permute(0, 2, 3, 1) # [bn, h, w, 2]
    out = F.grid_sample(content_feats.to(optical_flow.dtype), grid + optical_flow, align_corners=True) # [bn, c, h, w]

    return out.to(content_feats.dtype)

#----------------------------------------------------------------------------
