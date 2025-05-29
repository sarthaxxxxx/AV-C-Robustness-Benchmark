import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import io
import soundfile
from typing import Optional, Union, IO, Tuple, List, Dict
from pathlib import Path
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip
import tempfile

from corruptions.corruptions import corruption_dict

# Modified from https://github.com/YuanGongND/cav-mae/blob/master/src/dataloader.py
class AVRobustBench(Dataset):
    '''
    Create an instance of the AVRobustBench dataset.

    By default, no corruption is applied and frame 4 of all frames is chosen.

    Args:
        json_file (path-like json object or file-like json object): The json file containing the metadata.
        frame_num (int): The specific frame to use and to corrupt, default is 4.
        corruption (str, optional): The corruption to apply to the image and audio, default is none.
        severity (int): The severity level of the corruption, default is 5.
        all_frames (bool, optional): Returns a list of all frames if true, default is False.
    '''

    def __init__(self, 
                 json_file: Union[str, Path, IO], 
                 frame_num: int = 4, 
                 corruption: Optional[str] = None, 
                 severity: int = 5,
                 all_frames: Optional[bool] = False) -> None:
        
        self.datapath = json_file
        self.frame_num = frame_num
        self.corruption = corruption
        self.severity = severity
        self.all_frames = all_frames

        if self.corruption is not None:
            assert self.corruption in corruption_dict, f"{self.corruption!r} is not a valid corruption"
            self.visual_corruption, self.audio_corruption = corruption_dict[self.corruption]


        with open(self.datapath, 'r') as f:
            data_json = json.load(f)

        self.data = data_json['data']
        self.data = self.process_data(self.data)

    def process_data(self, data_json: List[Dict[str, str]]) -> np.ndarray:
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], 
                            data_json[i]['labels'], 
                            data_json[i]['video_id'], 
                            data_json[i]['video_path']]
            
        data_np = np.array(data_json, dtype=str)

        return data_np

    def decode_data(self, np_data: np.ndarray) -> dict[str, str]:
        datum = {}

        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]

        return datum

    def get_image(self, filename: str) -> Image.Image:
        image = Image.open(filename)

        if self.corruption is not None:
            image = self.visual_corruption(image, severity=self.severity)

        return image

    def get_wav(self, filename: str) -> IO[bytes]:
        waveform, sr = torchaudio.load(filename)

        if self.corruption is not None:
            waveform = self.audio_corruption(waveform=waveform, intensity=self.severity)

        waveform = waveform.numpy().T

        buffer = io.BytesIO()
        soundfile.write(buffer, waveform, sr, format='WAV')
        buffer.seek(0)

        return buffer

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[List[Image.Image], IO[bytes]]:
        datum = self.data[index]
        datum = self.decode_data(datum)

        audio_path = str(datum['wav'])
        audio = self.get_wav(audio_path)

        video_path = str(datum['video_path'])
        video_id = str(datum['video_id'])

        frames = []

        if not self.all_frames:
            image_path = video_path + f'/frame_{self.frame_num}/' + video_id + '.jpg'
            image = self.get_image(image_path)
            frames.append(image)
        else:
            for i, frame in enumerate(os.listdir(video_path)):    
                image_path = video_path + f'/{frame}/' + video_id + '.jpg'
                image = self.get_image(image_path)
                frames.append(image)

        return frames, audio
    
    @staticmethod
    def create_video(video_path: str, 
                     corruption: str = 'gaussian', 
                     severity: int = 5, 
                     duration: Optional[float] = None, 
                     save_path: Optional[str] = None) -> io.BytesIO:
        """
        Apply visual/audio corruption to an MP4 and return it as a BytesIO with the option to save. 
        
        Default corruption is gaussian.

        Args:
            video_path (str): Path to the source .mp4 file.
            corruption (str): Name of the corruption.
            severity (int): Corruption severity.
            duration (int, optional): Seconds from start to process (None for full length).
            save_path (str, optional): If provided, also write the output to this path.

        Returns:
            A BytesIO containing the corrupted MP4 data.
        """

        assert corruption in corruption_dict, f"{corruption} is not a corruption"
        visual_corruption, audio_corruption = corruption_dict[corruption]


        clip = VideoFileClip(video_path)
        if duration is not None:
            clip = clip.subclip(0, duration)
        fps = clip.fps

        frames = []
        for frame in clip.iter_frames(fps=fps, dtype="uint8"):
            image = Image.fromarray(frame)
            image = visual_corruption(x=image, severity=severity)
            frames.append(np.array(image))

        corrupted_video = ImageSequenceClip(frames, fps=fps)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            clip.audio.write_audiofile(tmp_audio.name, logger=None, verbose=False)
            waveform, sr = torchaudio.load(tmp_audio.name)
            waveform = audio_corruption(waveform=waveform, intensity=severity)
            torchaudio.save(tmp_audio.name, waveform, sr)

        corrupted_audio = AudioFileClip(tmp_audio.name)
        corrupted_video = corrupted_video.set_audio(corrupted_audio)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            corrupted_video.write_videofile(tmp_vid.name, codec="libx264", audio_codec="aac", logger=None, verbose=False)
            tmp_vid_path = tmp_vid.name

        buf = io.BytesIO()
        with open(tmp_vid_path, 'rb') as f:
            buf.write(f.read())
        buf.seek(0)

        if save_path:
            with open(save_path, 'wb') as f:
                f.write(buf.getvalue())

        os.remove(tmp_audio.name)
        os.remove(tmp_vid_path)

        return buf
    