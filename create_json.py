import json
import csv
import os

path_to_videos = './assets/eval_segments'
path_to_frames = './assets/eval_frames'
path_to_audio = './assets/eval_audio'
path_to_labels = './assets/eval_segments.csv'

save_path = './assets/eval.json'

labels_dict = {} 
with open(path_to_labels, 'r', newline='') as f:
    reader = csv.reader(f, skipinitialspace=True)
    for row in reader:
        if row[0].startswith('#'):
            continue
        video_id = row[0]
        labels = row[3]
        labels_dict[video_id] = labels

data = []

for video in os.listdir(path_to_videos):
    video_id = video.split('.')[0]
    wav = os.path.join(os.path.abspath(path_to_audio), video_id + ".wav")
    video_path = os.path.abspath(path_to_frames)
    labels = labels_dict[video_id]

    data.append({
        'wav': wav,
        'labels': labels,
        'video_id': video_id,
        'video_path': video_path,
    })

with open(save_path, 'w') as f:
    json.dump({'data': data}, f, indent=4)
