import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Easy video feature extractor')
parser.add_argument("-meta_file", type=str, default='/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/EquiAV/dataprep/meta/vgg_comb.csv', help="Should be a csv file of a single columns, each row is the input video path.")
parser.add_argument("-vid_dir", type=str, default='/mnt/data2/wpian/VGGSound/VGGSound', help="The place to store the video frames.")
parser.add_argument("-save_dir", type=str, default='/mnt/data2/saksham/AV_robust/equiAV_audio/', help="The place to store the video frames.")
parser.add_argument("--dry_run", action="store_true")

args = parser.parse_args()

df = pd.read_csv(args.meta_file)
if args.dry_run:
    df = df.head(5)

num_file = len(df)
print('Total {:d} videos are input'.format(num_file))

# first resample audio
for i, row in tqdm(df.iterrows(), total=num_file):
    input_f = os.path.join(args.vid_dir,row['vid'] + '.mp4')
    video_id = row['vid']
    output_f_1 = args.save_dir + '/' + video_id + '_intermediate.wav'
    os.system('ffmpeg -i {:s} -vn -ar 16000 {:s}'.format(input_f, output_f_1)) # save an intermediate file

for i, row in tqdm(df.iterrows(), total=num_file):
    input_f = os.path.join(args.vid_dir, row['vid'] + '.mp4')
    video_id = row['vid']
    output_f_1 = args.save_dir + '/' + video_id + '_intermediate.wav'
    output_f_2 = args.save_dir + '/' + video_id + '.wav'
    os.system('sox {:s} {:s} remix 1'.format(output_f_1, output_f_2))
    os.remove(output_f_1)