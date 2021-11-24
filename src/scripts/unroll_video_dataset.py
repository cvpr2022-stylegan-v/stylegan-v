import os
import json
import shutil
import argparse

from tqdm import tqdm


def unroll_video_dataset(dataset_path: str):
    # os.chdir(os.path.dirname(dataset_path))

    target_data_path = f'{dataset_path}_unrolled'
    video_names = sorted(os.listdir(dataset_path))
    # target_dataset_name = os.path.basename(target_data_path)
    has_labels = 'dataset.json' in video_names

    if has_labels:
        with open(os.path.join(dataset_path, 'dataset.json'), 'r') as f:
            old_labels = json.load(f)['labels']
            new_labels = {}

    for video_name in tqdm(video_names):
        video_path = os.path.join(dataset_path, video_name)

        if os.path.isfile(video_path):
            assert video_name == 'dataset.json'
            continue

        for i, frame_file_name in enumerate(os.listdir(video_path)):
            frame_path = os.path.join(video_path, frame_file_name)
            curr_video_name = f'{video_name}_{i:06d}'
            curr_target_dir = os.path.join(target_data_path, curr_video_name)
            curr_target_path = os.path.join(target_data_path, curr_video_name, frame_file_name)
            os.makedirs(curr_target_dir, exist_ok=True)
            shutil.copy(frame_path, curr_target_path)

            if has_labels:
                new_labels[curr_video_name] = old_labels[video_name]

    if has_labels:
        with open(os.path.join(target_data_path, 'dataset.json'), 'w') as f:
            json.dump({'labels': new_labels}, f)

    # shutil.make_archive(target_data_path, 'zip', target_data_path, base_dir=target_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unrolls a video dataset into a dir of one-frame videos')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()

    unroll_video_dataset(args.dataset_path)
