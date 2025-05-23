# AV-C-Robustness-Benchmark
## Introduction
[[Demo](https://www.youtube.com/watch?v=hYdcRO3BuIY&ab_channel=SarthakMaharana)]


This repo contains the official PyTorch implementation of AVRobustBench, a benchmark consisting of four audio-visual datasets, AudioSet-2C, VGGSound-2C, KINETICS-2C, and EpicKitchens-2C. "2C" stands for adding corruptions to both modalities. These datasets span diverse domains, environments, and action categories, offering a broad and realistic evaluation suite for audio-visual recognition models. Our benchmark takes these datasets and adds up to 15 different co-occuring and correlated corruptions at 5 different severity levels to benchmark state-of-the-art audio-visual models to test robustness to corruptions. Our datasets are [here](https://huggingface.co/datasets/sakshamsingh1/av_robust_data/tree/main).


While recent audio-visual models have demonstrated impressive performance, their robustness to distributional shifts at test-time remains not fully understood. Existing robustness benchmarks mainly focus on single modalities, making them insufficient for thoroughly assessing the robustness of audio-visual models. Motivated by real-world scenarios where shifts can occur simultaneously in both audio and visual modalities, we introduce AVRobustBench, a comprehensive benchmark designed to evaluate the test-time robustness of audio-visual recognition models.


## AVRobustBench Dataset
We release the AVRobustBench (Audio-Visual 2 Corruptions) dataset to supplement our work. `dataset.py` contains the code which takes any audio-visual datasets and adds any of the corruptions at any severity level that we released. The file is modular, allowing anyone to modify it for their datasets and their models. `/corruptions` contains the code for each of the corruptions, their severity levels, the images to create corrupted images, and the audio to create corrupted audios. 

### Setting up the current dataset
To begin, you will be required to create a json file for your dataset containing its wav path, labels, video ID, and video path. For example, `BkjpjAohg-0` is the video ID for a file `BkjpjAohg-0.mp4`. The video path is the directory containing the frames for the video ID. As an example using AudioSet:

```json
{
    "data": [
        {
            "wav": "/home/adrian/Data/AudioSet/eval_audio/BkjpjAohg-0.wav",
            "labels": "/m/04rlf,/m/07pjwq1,/m/07s72n,/m/08cyft",
            "video_id": "BkjpjAohg-0",
            "video_path": "/home/adrian/Data/AudioSet/eval_frames"
        },
        {
            "wav": "/home/adrian/Data/AudioSet/eval_audio/4ufZrEAJnJI.wav",
            "labels": "/m/0242l",
            "video_id": "4ufZrEAJnJI",
            "video_path": "/home/adrian/Data/AudioSet/eval_frames"
        }]
}
```

The dataset json file need not follow this structure. Modify it for your needs. If you modify the json file, you will need to modify `dataset.py`. Note that the dataset file does not contain any logic for labels, as different datasets have different label structures. A user can easily modify our code to add their relevant metadata, labels, and any other corruptions. Our dataset makes it easy to get an image/audio pair from their dataset and add a corruption to both modalities. 


Create a json path and pass this into the dataset class.
```python
from dataset import AVRobustBench
file_path = 'path/to/your/json'
dataset = AVRobustBench(file_path, corruption='gaussian', severity=5)
```

Each entry in `AVRobustBench` contains an `(image, audio)` tuple with the option of adding corruptions on them. `image` is a PIL image, while `audio` is a BytesIO .wav file-like. PIL images are the standard, but there is no standard for audio files. Some codebases use `torchaudio`, `librosa`, `soundfile`, or something else. Due to `audio` being a .wav file-like stored in memory, it can be passed as a .wav to any audio library.


`demo.ipynb` showcases a few examples of using the dataset.
