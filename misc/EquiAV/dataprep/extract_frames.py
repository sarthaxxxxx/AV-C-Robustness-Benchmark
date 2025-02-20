import os
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import pandas as pd
from tqdm import tqdm

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])

def extract_frame(video_id, input_video_path, save_dir, extract_frame_num=10):
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # this is to avoid vggsound video's bug on not accurate frame count
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))
    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num/extract_frame_num))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        _, frame = vidcap.read()
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        # save in 'target_path/frame_{i}/video_id.jpg'
        if os.path.exists(save_dir + '/frame_{:d}/'.format(i)) == False:
            os.makedirs(save_dir + '/frame_{:d}/'.format(i))
        save_image(image_tensor, save_dir + '/frame_{:d}/'.format(i) + video_id + '.jpg')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Python script to extract frames from a video, save as jpgs.")
    parser.add_argument("-meta_file", type=str, default='/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/EquiAV/dataprep/meta/vgg_comb.csv', help="Should be a csv file of a single columns, each row is the input video path.")
    parser.add_argument("-vid_dir", type=str, default='/mnt/data2/wpian/VGGSound/VGGSound', help="The place to store the video frames.")
    parser.add_argument("-save_dir", type=str, default='/mnt/data2/saksham/AV_robust/equiAV/', help="The place to store the video frames.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # note the first row (header) is skipped
    df = pd.read_csv(args.meta_file)
    if args.dry_run:
        df = df.head(5)
    num_file = len(df)
    print('Total {:d} videos are input'.format(num_file))
    for i, row in tqdm(df.iterrows(), total=num_file):
        file_path = os.path.join(args.vid_dir, row['vid'] + '.mp4')
        save_dir = os.path.join(args.save_dir, row['vid'])
        os.makedirs(save_dir, exist_ok=True)
        try :
            extract_frame(row['vid'], file_path, save_dir)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")