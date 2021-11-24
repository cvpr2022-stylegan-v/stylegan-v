"""Temporal Frechet Inception Distance (TFID)"""

import copy
import pickle
import torch
import numpy as np
import scipy.linalg

from src import dnnlib
from . import metric_utils

# We use a different batch size depending on the resolution
NUM_FRAMES_PER_INFERENCE_STEP = {
    128: 256,
    256: 128,
    512: 64,
    1024: 32,
}

#----------------------------------------------------------------------------

def compute_tfid(opts, max_real: int, num_gen: int, num_frames: int):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    opts = copy.deepcopy(opts)
    opts.dataset_kwargs.load_n_consecutive = num_frames
    opts.dataset_kwargs.discard_short_videos = True
    batch_size = NUM_FRAMES_PER_INFERENCE_STEP[opts.dataset_kwargs.resolution] // num_frames

    stats_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=0,
        capture_mean_cov=True, max_items=max_real, batch_size=batch_size, feature_stats_cls=DiffBasedFeatureStats, video_len=num_frames).get_mean_cov()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_opts.dataset_kwargs.load_n_consecutive = num_frames
        gen_opts.dataset_kwargs.discard_short_videos = True
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames)

    stats_gen = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs=detector_kwargs, rel_lo=0, rel_hi=1, capture_mean_cov=True,
        max_items=num_gen, batch_size=batch_size, feature_stats_cls=DiffBasedFeatureStats, video_len=num_frames, **gen_kwargs).get_mean_cov()

    if opts.rank != 0:
        return [float('nan') for i in range(num_frames - 1)]

    fids = []
    for (mu_real, sigma_real), (mu_gen, sigma_gen) in zip(stats_real, stats_gen):
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2)).item()
        fids.append(fid)

    return fids

#----------------------------------------------------------------------------

class DiffBasedFeatureStats:
    def __init__(self, *args, video_len: int=0, max_items: int=None, **kwargs):
        assert video_len > 0, "Please, specify `video_len` explicitly."

        self.video_len = video_len
        self.max_items = max_items
        self.stats_list = [metric_utils.FeatureStats(*args, max_items=max_items, **kwargs) for _ in range(video_len - 1)]

    def is_full(self):
        return all(s.is_full() for s in self.stats_list)

    def append_torch(self, x, *args, **kwargs):
        """
        We assume that all x are of the same video
        """
        assert x.ndim == 2, f"Bad shape: {x.shape}"
        assert x.shape[0] % self.video_len == 0, f"Bad shape: {x.shape}"

        batch_size, feat_dim = x.shape[0] // self.video_len, x.shape[1]
        x = x.view(batch_size, self.video_len, feat_dim) # [batch_size, video_len, feat_dim]
        diffs = x.unsqueeze(2) - x.unsqueeze(1) # [batch_size * video_len, video_len, feat_dim]

        for frame_dist_idx in range(self.video_len - 1):
            x_idx = torch.arange(frame_dist_idx + 1, self.video_len) # [video_len - frame_dist_idx]
            y_idx = torch.arange(self.video_len - frame_dist_idx - 1) # [video_len - frame_dist_idx]
            shifts = diffs[:, x_idx, y_idx, :] # [batch_size, frame_dist_idx, feat_dim]
            shifts = shifts.view(batch_size * len(x_idx), feat_dim) # [batch_size * frame_dist_idx, feat_dim]
            self.stats_list[frame_dist_idx].append_torch(shifts, *args, **kwargs)

    def get_mean_cov(self):
        return [s.get_mean_cov() for s in self.stats_list]

    def save(self, *args, **kwargs):
        metric_utils.FeatureStats.save(self, *args, **kwargs)

    @property
    def num_items(self) -> int:
        return min(s.num_items for s in self.stats_list)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = DiffBasedFeatureStats(capture_all=s.stats_list[0].capture_all, max_items=s.max_items, video_len=s.video_len)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------
