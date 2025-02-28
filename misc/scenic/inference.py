import tensorflow as tf
import soundfile as sf
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# def extract_frames(video_path, num_frames=32, img_size=224):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     frame_indices = np.linspace(
#         0, total_frames - 1, num_frames, dtype=int
#     )

#     frames = []
#     for idx in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         success, frame = cap.read()
#         if not success:
#             break
#         # Resize to 224 x 224
#         frame = cv2.resize(frame, (img_size, img_size))
#         # Convert BGR to RGB
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
    
#     cap.release()

#     frames = np.array(frames, dtype=np.float32)
#     return frames


def extract_frames(video_path, num_frames=32, img_size=224):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:  # If the video can't be read
        print(f"Warning: No frames found in {video_path}")
        return np.zeros((num_frames, img_size, img_size, 3), dtype=np.float32)  # Return black frames
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        while not success and idx < total_frames - 1:
            idx += 1  # Move to the next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
        
        if not success and idx > 0:
            idx -= 1  # Move back
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
        
        if success:
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            print(f"Warning: Could not retrieve frame {idx} in {video_path}")
    
    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((img_size, img_size, 3), dtype=np.float32))  # Pad with black frames
    
    return np.array(frames, dtype=np.float32)

def load_audio_waveform(audio_path, desired_samples=128000):
    waveform, sr = sf.read(audio_path)

    if len(waveform) < desired_samples:
        pad_length = desired_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_length), mode='constant')
    else:
        waveform = waveform[:desired_samples]
    waveform = waveform[:, np.newaxis]
    return waveform.astype(np.float32)

meta_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/scenic/data/vgg_train_subset.csv'
df = pd.read_csv(meta_path)
# df = df.head(5)
base_vid_dir = '/mnt/data1/wpian/VGGSound/VGGSound'
base_aud_dir = '/mnt/data1/saksham/AV_robust/equiAV_audio/'

model_dir = "/mnt/data1/saksham/AV_robust/scenic/tf_saved_model"
model = tf.saved_model.load(model_dir)

def get_pred(vid):
    # print(vid)
    vid_path = os.path.join(base_vid_dir, vid + '.mp4')
    aud_path = os.path.join(base_aud_dir, vid + '.wav')
    
    frames_np = extract_frames(vid_path, num_frames=32, img_size=224)
    frames_tf = tf.convert_to_tensor(frames_np[None, ...], dtype=tf.float32)

    waveform_np = load_audio_waveform(aud_path, desired_samples=128000)
    waveform_tf = tf.convert_to_tensor(waveform_np[None, ...], dtype=tf.float32)

    rgb_input = frames_tf  # shape: (1, 32, 224, 224, 3)
    waveform_input = waveform_tf  # shape: (1, 128000, 1)

    inputs = {
        'rgb': rgb_input,
        'waveform': waveform_input
    }

    outputs = model(inputs)
    max_index = tf.argmax(outputs, axis=1).numpy()[0]  # Convert to NumPy and extract scalar
    return max_index
    # print(max_index)

tqdm.pandas()
df['pred'] = df['vid'].progress_apply(get_pred)
save_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/scenic/data/vgg_train_subset_pred.csv'
df.to_csv(save_path, index=False)    