import json
import csv
import os

# Path to the directory containing the videos
path_to_videos = './AudioSet/eval_segments'
# Path to the directory containing the frames
path_to_frames = './AudioSet/eval_frames'
# Path to directory containing the audio
path_to_audio = './AudioSet/eval_audio'
# Path to any metadata your dataset requires
path_to_labels = './AudioSet/eval_segments.csv'

data = []

for image in os.listdir(path_to_videos):
    video_id = image.split('.')[0]
    wav = os.path.join(os.path.abspath(path=path_to_audio), video_id) + ".wav"
    video_path = os.path.abspath(path=path_to_frames)
    labels = ''

    with open(path_to_labels, 'r', newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)

        for row in reader:
            if row[0].startswith('#'):
                continue # Skip headers

            if row[0] == video_id:
                labels = labels + row[3]
    
    data_dict = {
                 'wav': wav,
                 'labels': labels,
                 'video_id': video_id,
                 'video_path': video_path,}
    
    data.append(data_dict)

json_data = {'data': data}

with open('./AudioSet/eval.json', 'w') as f:
    json_file = json.dump(json_data, f, indent=4)