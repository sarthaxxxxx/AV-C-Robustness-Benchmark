import os
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm

meta_data = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/final_list.csv'
df = pd.read_csv(meta_data)
df.head()

def mix_audio(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    # audio = audio.set_duration(video.duration)
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")

base_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/subset'
save_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/data'
for i, row in tqdm(df.iterrows(), total=len(df)):
    video_path = os.path.join(base_dir, row['noise'],row['video_id']+ '.mp4')
    audio_path = os.path.join(base_dir, row['noise'],row['video_id']+ '.wav')
    output_path = os.path.join(save_dir, row['label'] + '_' + row['video_id'] + '.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mix_audio(video_path, audio_path, output_path)