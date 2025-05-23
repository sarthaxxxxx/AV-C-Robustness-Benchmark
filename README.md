# AV-C-Robustness-Benchmark
This repo contains the corruptions for 


## AV2C Dataset
We release the AV2C (Audio-Visual 2 Corruptions) dataset to supplement our work. `avc2_dataset.py` contains the dataset which takes any audio-visual datasets and adds any of the corruptions at any severity level that we released. The file is modular, allowing anyone to modify it for their datasets and their models. `/corruptions` contains the code for each of the corruptions, their severity levels, the images to create corrupted images, and the audio to create corrupted audios. 

### Setting up the current dataset