#!/bin/bash

# generate corrupted datasets

# AUDIO
# user input
severity=1 # severity of corruption (1-5)
audio_data_path=$"../sample_audio" # path to the original dataset (audio)
audio_save_path=$"../corrupt_audio" # path to save the corrupted dataset (audio)
noise_path=$"../../data/VGGSound/NoisyAudios" # path to the noisy audios to overlay
visual_data_path=$"../sample_frames" # path to the original dataset (visual)
visual_save_path=$"../corrupt_frames" # path to save the corrupted dataset (visual)


echo "Corrupting audio dataset with severity $severity"
echo "Original dataset path: $data_path"
echo "Saving corrupted dataset to: $save_path"
echo "Noise path: $noise_path"

python3 generate_audio_corruptions.py --data_path $audio_data_path --save_path $audio_save_path --noise_path $noise_path --severity $severity --corruption "all"

echo "Done with audio corruption"


sleep 10

# IMAGE

echo "Corrupting image dataset with severity $severity"
echo "Original dataset path: $data_path"
echo "Saving corrupted dataset to: $save_path"

python3 generate_visual_corruptions.py --data_path $visual_data_path --save_path $visual_save_path --severity $severity --corruption "all"

echo "Done with image corruption"