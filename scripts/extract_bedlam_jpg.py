"""Extract JPG frames from the official BEDLAM 30 fps MP4 layout."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import ROOT


def mp4_to_jpg(video, frame_folder):
    import cv2

    cap = cv2.VideoCapture(str(video))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not cv2.imwrite(str(Path(frame_folder) / f'{count:04d}.jpg'), frame):
            raise IOError(f'Could not write frame {count} from {video}')
        count += 1
    cap.release()


def mp4_to_jpg_ffmpeg(video, frame_folder):
    import ffmpeg

    (
        ffmpeg.input(str(video))
        .output(str(Path(frame_folder) / '%04d.jpg'), vf='fps=30', start_number=0, qscale=1)
        .run(quiet=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bedlam-root', default=f'{ROOT}/bedlam_30fps',
        help='BEDLAM root containing mp4/<scene>/mp4/*.mp4 (default: TRAM_DATA_ROOT/bedlam_30fps)',
    )
    parser.add_argument(
        '--output-root',
        help='Image destination (default: <bedlam-root>/bedlam_data/images)',
    )
    args = parser.parse_args()
    from tqdm import tqdm

    root = Path(args.bedlam_root)
    output_root = Path(args.output_root) if args.output_root else root / 'bedlam_data' / 'images'
    videos = sorted(root.glob('mp4/*/mp4/*.mp4'))
    if not videos:
        raise FileNotFoundError(f'No BEDLAM MP4 files found under {root / "mp4"}')
    for video in tqdm(videos):
        scene = video.parts[-3]
        sequence = video.stem
        image_dir = output_root / scene / 'jpg' / sequence
        image_dir.mkdir(parents=True, exist_ok=True)
        mp4_to_jpg_ffmpeg(video, image_dir)
    print(f'Extracted {len(videos)} videos into {output_root}')


if __name__ == '__main__':
    main()
