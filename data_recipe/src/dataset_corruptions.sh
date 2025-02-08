#!/bin/bash

# generate corrupted datasets

# AUDIO
# user input
severity=1 # severity of corruption (1-5)
corruption="gaussian_noise"
audio_data_path="/home/jovyan/EPIC-KITCHENS-SOUND" # path to the original dataset (audio)
audio_save_path="/home/jovyan/SOUND-C" # path to save the corrupted dataset (audio)
noise_path="/home/jovyan/workspace/AC-C-Robustness-Benchmark/data/VGGSound/NoisyAudios" # path to the noisy audios to overlay
visual_data_path="/home/jovyan/EPIC-KITCHENS" # path to the original dataset (visual)
visual_save_path="/home/jovyan/FRAMES-C/$corruption/severity_$severity" # path to save the corrupted dataset (visual)


echo "Corrupting audio dataset with severity $severity"
echo "Original dataset path: $audio_data_path"
echo "Saving corrupted dataset to: $audio_save_path"
echo "Noise path: $noise_path"

python3 generate_audio_corruptions.py --data_path $audio_data_path --save_path $audio_save_path --noise_path $noise_path --severity $severity --corruption $corruption

echo "Done with audio corruption"

# sleep 5

# IMAGE

echo "Corrupting image dataset with severity $severity"
echo "Original dataset path: $visual_data_path"
echo "Saving corrupted dataset to: $visual_save_path"

python3 generate_visual_corruptions.py --data_path $visual_data_path --save_path $visual_save_path --severity $severity --corruption $corruption

echo "Done with image corruption"