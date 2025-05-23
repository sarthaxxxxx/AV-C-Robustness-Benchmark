# AV-C-Robustness-Benchmark
This repo contains the corruptions for 


## AV2C Dataset
We release the AV2C (Audio-Visual 2 Corruptions) dataset to supplement our work. `avc2_dataset.py` contains the dataset which takes any audio-visual datasets and adds any of the corruptions at any severity level that we released. The file is modular, allowing anyone to modify it for their datasets and their models. `/corruptions` contains the code for each of the corruptions, their severity levels, the images to create corrupted images, and the audio to create corrupted audios. 

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
        }, ...
```

The dataset json file need not follow this structure. Modify it for your needs. If you modify the json file, you will need to modify `avc2_dataset.py`. Note that the dataset file does not contain any logic for labels, as different datasets have different labels. A user can easily modify our code to add their relevant metadata, labels, and any other corruptions. Our dataset makes it easy to get an image/audio from their dataset and add a corruption to both modalities. 


Create a json path and pass this into the dataset class.
```python
from avc2_dataset import AV2C
file_path = 'path/to/your/json'
dataset = AV2C(file_path, corruption='gaussian', severity=5)
```

Each entry in `AV2C` contains an `(image, audio)` with the option of adding corruptions on them. `image` is a PIL image, while `audio` is a BytesIO .wav file-like. PIL images are the standard, but there is no standard for audio files. Some libraries use `torchaudio`, `librosa`, `soundfile`, or something else. Due to `audio` being a .wav file-like stored in memory, it can be passed as a .wav to any audio library.