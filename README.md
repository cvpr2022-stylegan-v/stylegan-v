# A Continuous Video Generator with the Price, Quality and Perks of StyleGAN2
Samples: [https://cvpr2022-stylegan-v.github.io](https://cvpr2022-stylegan-v.github.io/)

### Installation
To install the dependencies, run the following command:
```
conda env create -f environment.yaml -p env
conda activate ./env
```
You will also need to install `av` via pip.
After than, make a search for `/PATH/TO` over this repo and replace with the necessary paths.
For clip editing, you will need to install [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) and `clip`.
This repo is built on top of [StyleGAN2-ADA](https://github.com/nvlabs/stylegan2-ada).
__Make sure that [StyleGAN2-ADA](https://github.com/nvlabs/stylegan2-ada) is runnable on your system.__

If you have Ampere GPUs, then use `environment-ampere.yaml` instead.

### Training
To launch the training, use the following command:
```
python src/infra/launch.py hydra.run.dir=. +exp_suffix=my_experiment env=local dataset=sky_timelapse_256 num_gpus=4
```

Available dataset configs are located in `configs/dataset`.

If you do not want `hydra` to create some log directories (typically, you don't), run the above command this way:
```
python src/infra/launch.py hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled +exp_suffix=my_experiment dataset=sky_timelapse_256 num_gpus=4
```

In case [slurm](https://slurm.schedmd.com/documentation.html) is installed on your system, you can submit the slurm job with the above training by adding `+slurm=true` parameter.
Sbatch arguments are specified in `configs/infra.yaml`, you can update them with your required ones.
Also note that you can create your own environment in `configs/env`.

On older GPUs (non V100 and newer), custom CUDA kernels (bias_act and upfirdn2n) might fail to compile. The following two lines can help:
```
export TORCH_CUDA_ARCH_LIST="7.0"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

To train the `MoCoGAN+SG2` model, just use the `mocogan.yaml` model config with `traditional_16.yaml` sampling:
```
python src/infra/launch.py hydra.run.dir=. +exp_suffix=my_experiment env=local dataset=sky_timelapse_256 num_gpus=4 model=mocogan sampling=traditional_16
```

### Dataset structure
Dataset should be structured as:
```
dataset/
    video1/
        - frame1.jpg
        - frame2.jpg
        - ...
    video2/
        - frame1.jpg
        - frame2.jpg
        - ...
    ...
```
So, we use a frame-wise structure, because it makes loading faster for sparse sampling.
It can also be a zip archive (to avoid additional overhead when copying it to a cluster).

Datasets can be downloaded here:
- SkyTimelapse: https://github.com/weixiong-ur/mdgan
- UCF: https://www.crcv.ucf.edu/data/UCF101.php
- FaceForensics: https://github.com/ondyari/FaceForensics
- RainbowJelly: https://www.youtube.com/watch?v=P8Bit37hlsQ

After downloading `RainbowJelly`, save it as `rainbow_jelly.mp4` and convert it into the dataset by running:
```
python src/scripts/convert_video_to_dataset.py -s rainbow_jelly.mp4 -t /tmp/data/rainbow_jelly_256 --target_size 256 -sf 150 -cs 512
```

### Projection and editing
Those two files provide projection and editing:
- `src/scripts/project.py`
- `src/scripts/clip_edit.py`

### Infrastructure and visualization
You will find a lot of useful scripts in `src/scripts`

### FVD computation
To compute FVD between two datasets, run the following command:
```
python src/scripts/calc_metrics_for_dataset.py --real_data_path /path/to/dataset_a.zip --fake_data_path /path/to/dataset_b.zip --mirror 1 --gpus 4 --resolution 256 --metrics fvd2048_16f,fvd2048_128f,fvd2048_128f_subsample,fid50k_full --verbose 0 --use_cache 0
```
Both datasets should be in the format specified above.
They can be either zip archives or normal directories.
This will compute several metrics:
- FID
- FVD 16
- FVD 128
- FVD 128 subsampled

### License
This repo is built on top of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), so it is restricted by the [NVidia license](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).
