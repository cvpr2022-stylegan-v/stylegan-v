import os
from typing import List, Callable, Optional, Dict

from PIL import Image
import torch
from torch import Tensor
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import utils
import torchvision.transforms.functional as TVF

from src.training.layers import generate_coords


@torch.no_grad()
def generate_videos(
    G: Callable, z: Tensor, c: Tensor, ts: Tensor, l: Tensor, motion_noise: Optional[Tensor]=None,
    noise_mode='const', truncation_psi=1.0, verbose: bool=False, as_grids: bool=False, batch_size_num_frames: int=100) -> Tensor:

    assert len(ts) == len(z) == len(c) == len(l), f"Wrong shape: {ts.shape}, {z.shape}, {c.shape}, {l.shape}"
    assert ts.ndim == 2, f"Wrong shape: {ts.shape}"

    G.eval()
    videos = []

    if c.shape[1] > 0 and truncation_psi < 1:
        num_ws_to_average = 1000
        c_for_avg = c.repeat_interleave(num_ws_to_average, dim=0) # [num_classes * num_ws_to_average, num_classes]
        z_for_avg = torch.randn(c_for_avg.shape[0], G.z_dim, device=z.device) # [num_classes * num_ws_to_average, z_dim]
        w = G.mapping(z_for_avg, c=c_for_avg)[:, 0] # [num_classes * num_ws_to_average, w_dim]
        w_avg = w.view(-1, num_ws_to_average, G.w_dim).mean(dim=1) # [num_classes, w_dim]

    iters = range(len(z))
    iters = tqdm(iters, desc='Generating videos') if verbose else iters

    if motion_noise is None and not G.synthesis.motion_encoder is None:
        motion_noise = G.synthesis.motion_encoder(c=c, t=ts, l=l, w=torch.zeros(len(z), G.w_dim, device=z.device))['motion_noise'] # [...any...]

    for video_idx in iters:
        curr_video = []

        for curr_ts in ts[[video_idx]].split(batch_size_num_frames, dim=1):
            curr_z = z[[video_idx]] # [1, z_dim]
            curr_c = c[[video_idx]] # [1, c_dim]
            curr_l = l[[video_idx]] # [1]
            curr_motion_noise = motion_noise[[video_idx]]

            if curr_c.shape[1] > 0 and truncation_psi < 1:
                curr_w = G.mapping(curr_z, c=curr_c, l=curr_l, truncation_psi=1) # [1, num_ws, w_dim]
                curr_w = truncation_psi * curr_w + (1 - truncation_psi) * w_avg.unsqueeze(1) # [1, num_ws, w_dim]
                out = G.synthesis(
                    ws=curr_w,
                    c=curr_c,
                    t=curr_ts,
                    motion_noise=curr_motion_noise,
                    noise_mode=noise_mode) # [1 * video_len, 3, h, w]
            else:
                out = G(
                    z=curr_z,
                    c=curr_c,
                    t=curr_ts,
                    l=curr_l,
                    motion_noise=curr_motion_noise,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode) # [1 * video_len, 3, h, w]

            out = (out * 0.5 + 0.5).clamp(0, 1).cpu() # [1 * video_len, 3, h, w]
            curr_video.extend(out)

        videos.append(torch.stack(curr_video))

    if as_grids:
        frame_grids = torch.stack(videos).permute(1, 0, 2, 3, 4) # [video_len, len(z), c, h, w]
        frame_grids = [utils.make_grid(fs, nrow=int(np.sqrt(len(z)))) for fs in frame_grids] # [video_len, 3, grid_h, grid_w]

        return torch.stack(frame_grids)
    else:
        return torch.stack(videos)


def run_batchwise(fn: Callable, data_kwargs: Dict[str, Tensor], batch_size: int, **kwargs) -> Tensor:
    data_kwargs = {k: v for k, v in data_kwargs.items() if not v is None}
    seq_len = len(data_kwargs[list(data_kwargs.keys())[0]])
    result = []

    for i in range((seq_len + batch_size - 1) // batch_size):
        curr_data_kwargs = {k: d[i * batch_size: (i+1) * batch_size] for k, d in data_kwargs.items()}
        result.append(fn(**curr_data_kwargs, **kwargs))

    return torch.cat(result, dim=0)


def save_video_frames_as_mp4(frames: List[Tensor], fps: int, save_path: os.PathLike, verbose: bool=False):
    # Load data
    frame_h, frame_w = frames[0].shape[1:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = tqdm(frames, desc='Saving videos') if verbose else frames
    for frame in frames:
        assert frame.shape[0] == 3, "RGBA/grayscale images are not supported"
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Uncomment this line to release the memory.
    # It didn't work for me on centos and complained about installing additional libraries (which requires root access)
    # cv2.destroyAllWindows()
    video.release()


def save_video_frames_as_frames(frames: List[Tensor], save_dir: os.PathLike, time_offset: int=0):
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        save_path = os.path.join(save_dir, f'{i + time_offset:06d}.jpg')
        TVF.to_pil_image(frame).save(save_path, q=95)


@torch.no_grad()
def convert_optical_flow_to_imgs(optical_flow: Tensor, tmp_dir, verbose: bool=False) -> List[Image.Image]:
    assert optical_flow.ndim == 4, f"Wrong shape: {optical_flow.shape}"
    assert optical_flow.shape[3] == 2, f"Wrong shape: {optical_flow.shape}"
    os.makedirs(tmp_dir, exist_ok=True) # To dump dummy images to

    img_size = optical_flow.shape[1]
    X, Y = generate_coords(1, img_size, align_corners=True)[0] # [2, h, w]

    imgs = []
    iters = tqdm(optical_flow, desc='Converting optical flow to imgs') if verbose else optical_flow

    for flow in iters:
        U, V = flow.permute(2, 0, 1).cpu()
        fig = plt.figure(figsize=(5, 5))
        plt.quiver(X.numpy(), Y.numpy(), U.numpy(), V.numpy(), angles='xy', scale_units='xy', scale=1)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(tmp_dir, 'remove_me.jpg')) # Does not render without this...
        plt.close() # Render and close
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
        imgs.append(rgba_to_rgb(img))

    return imgs


def rgba_to_rgb(image, color=(255, 255, 255)):
    """
    Copy-pasted from https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
    """
    image.load()

    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel

    return background
