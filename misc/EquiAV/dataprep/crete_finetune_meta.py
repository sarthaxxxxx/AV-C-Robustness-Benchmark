import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

frame_base_dir = '/mnt/data1/saksham/AV_robust/equiAV/'
aud_base_dir = '/mnt/data1/saksham/AV_robust/equiAV_audio'
save_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/EquiAV/dataprep/VGGSound'

path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/EquiAV/dataprep/meta/vgg_comb.csv'
df = pd.read_csv(path)

# Filter the dataframe efficiently
SPLIT = 'train'
save_path = os.path.join(save_path, SPLIT+'.json')
df = df[df['split'] == SPLIT].reset_index(drop=True)

data = {"data": []}

def is_valid(vid):
    """Check if a video has all required frames and an audio file."""
    video_path = os.path.join(frame_base_dir, vid)
    audio_path = os.path.join(aud_base_dir, vid + ".wav")

    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return False
    
    # Check for all required frames
    for i in range(10):
        if not os.path.exists(os.path.join(video_path, f'frame_{i}', vid + '.jpg')):
            return False
    
    return True

def process_row(row):
    """Process a single row and return a valid data entry if conditions are met."""
    vid = row["vid"]
    if is_valid(vid):
        return {
            "video_id": vid,
            "video_path": os.path.join(frame_base_dir, vid),
            "wav": os.path.join(aud_base_dir, vid + ".wav"),
            "labels": row["label_equiav"],
        }
    return None

# Use multiprocessing for speedup
with Pool(8) as pool:
    results = list(tqdm(pool.imap(process_row, df.to_dict("records")), total=len(df)))

# Filter out None values (invalid entries)
data["data"] = [entry for entry in results if entry is not None]

# Save JSON file
with open(save_path, "w") as f:
    json.dump(data, f)

print(f"Saved {save_path}, {len(data['data'])}/{len(df)}")

###################### Iterative process to create finetune meta ######################

# import os
# import json
# import pandas as pd
# from tqdm import tqdm

# frame_base_dir = '/mnt/data1/saksham/AV_robust/equiAV/'
# aud_base_dir = '/mnt/data1/saksham/AV_robust/equiAV_audio'
# save_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/EquiAV/dataprep/VGGSound'

# path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/EquiAV/dataprep/meta/vgg_comb.csv'
# df = pd.read_csv(path)

# SPLIT = 'test' 
# df = df[df['split'] == SPLIT].reset_index(drop=True)

# data = {}
# data['data'] = []

# for i, row in tqdm(df.iterrows(), total=len(df)):
#     vid = row['vid']
#     video_path = os.path.join(frame_base_dir, vid)
#     audio_path = os.path.join(aud_base_dir, vid+'.wav')
#     if not os.path.exists(video_path) or not os.path.exists(audio_path):
#         continue
#     curr_dict = {}
#     curr_dict['video_id'] = vid
#     curr_dict['video_path'] = video_path
#     curr_dict['wav'] = audio_path
#     curr_dict['label'] = row['label_equiav']
#     data['data'].append(curr_dict)

# save_path = os.path.join(save_path, SPLIT+'.json')
# with open(save_path, 'w') as f:
#     json.dump(data, f)
# print(f'Saved {save_path}, {len(data["data"])} samples')