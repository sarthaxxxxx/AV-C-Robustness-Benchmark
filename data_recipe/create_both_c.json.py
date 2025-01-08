import os
import json
import copy
import random
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--clean-path', type=str, default='/home/jovyan/workspace/AV-C-Robustness-Benchmark/utils/epic-kitchen/clean/severity_0.json') # path to clean json (Severity 0)
parser.add_argument('--video-c-path', type=str, default="/home/jovyan/workspace/AV-C-Robustness-Benchmark/data_recipe/corrupt_frames")
parser.add_argument('--audio-c-path', type=str, default="/home/jovyan/workspace/AV-C-Robustness-Benchmark/data_recipe/corrupt_audio")
parser.add_argument('--corruption', nargs='*', default=['all'])
args = parser.parse_args()


severity_list = range(1, 6)

corruption_list = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'speckle_noise',
    'snow',
    'frost',
    'spatter',
    ]

for corruption in corruption_list:
    for severity in [1, 2, 3, 4, 5]:
        save_path = os.path.join(os.path.dirname(args.clean_path)[:-5], 'both')
        if not os.path.exists(os.path.join(save_path, corruption)):
            os.makedirs(os.path.join(save_path, corruption))

        with open(args.clean_path, 'r') as f:
            video_data = json.load(f)

        dic_list = []
        for dic in video_data['data']:
            new_dic = {
                "video_id": dic.get("video_id"),
                "wav": os.path.join(args.audio_c_path, corruption, 'severity_{}'.format(severity), '{}.wav'.format(dic.get("video_id"))),
                "video_path": os.path.join(args.video_c_path, f'{corruption}/severity_{severity}/'),
                "labels": dic.get("labels")
            }
            dic_list.append(new_dic)

        print(len(dic_list))
        random.shuffle(dic_list)
        new_json = {"data": dic_list}
        final_save_path = os.path.join(save_path, corruption)
        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)
        with open(os.path.join(final_save_path, 'severity_{}.json'.format(severity)), "w") as file1:
            json.dump(new_json, file1, indent=1)
