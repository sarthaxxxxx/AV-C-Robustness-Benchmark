import tensorflow as tf
import soundfile as sf
import numpy as np
import cv2
import os

def extract_frames(video_path, num_frames=32, img_size=224):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = np.linspace(
        0, total_frames - 1, num_frames, dtype=int
    )

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            break
        # Resize to 224 x 224
        frame = cv2.resize(frame, (img_size, img_size))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    frames = np.array(frames, dtype=np.float32)
    return frames

def load_audio_waveform(audio_path, desired_samples=128000):
    waveform, sr = sf.read(audio_path)

    if len(waveform) < desired_samples:
        pad_length = desired_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_length), mode='constant')
    else:
        waveform = waveform[:desired_samples]
    waveform = waveform[:, np.newaxis]
    return waveform.astype(np.float32)

base_vid_dir = '/mnt/data1/wpian/VGGSound/VGGSound'
base_aud_dir = '/mnt/data1/saksham/AV_robust/equiAV_audio/'

vid = 'snh7E7llb48_000070'
vid_path = os.path.join(base_vid_dir, vid + '.mp4')
aud_path = os.path.join(base_aud_dir, vid + '.wav')

frames_np = extract_frames(vid_path, num_frames=32, img_size=224)
frames_tf = tf.convert_to_tensor(frames_np[None, ...], dtype=tf.float32)

waveform_np = load_audio_waveform(aud_path, desired_samples=128000)
waveform_tf = tf.convert_to_tensor(waveform_np[None, ...], dtype=tf.float32)

# Create dummy inputs with the correct shapes
# dummy_rgb = tf.random.normal([1, 32, 224, 224, 3])
# dummy_waveform = tf.random.normal([1, 128000, 1])  # note 128000 instead of 160000

# dummy_input = {
#     'rgb': dummy_rgb,
#     'waveform': dummy_waveform
# }

rgb_input = frames_tf  # shape: (1, 32, 224, 224, 3)
waveform_input = waveform_tf  # shape: (1, 128000, 1)

inputs = {
    'rgb': rgb_input,
    'waveform': waveform_input
}

model_dir = "/mnt/data1/saksham/AV_robust/scenic/tf_saved_model"
model = tf.saved_model.load(model_dir)

outputs = model(inputs)
max_index = tf.argmax(outputs, axis=1).numpy()[0]  # Convert to NumPy and extract scalar
print(max_index)
print(outputs.shape)