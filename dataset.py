import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import io
import soundfile

from corruptions.corruptions import corruption_dict

# Modified from https://github.com/YuanGongND/cav-mae/blob/master/src/dataloader.py
class AVRobustBench(Dataset):
    def __init__(self, json_file, frame_num=4, corruption=None|str, severity=5):
        self.datapath = json_file
        self.frame_num = frame_num
        self.corruption = corruption
        self.severity = severity

        if self.corruption in corruption_dict.keys():
            self.visual_corruption, self.audio_corruption = corruption_dict[self.corruption]
        else: "Corruption does not exist."

        with open(json_file, 'r') as f:
            data_json = json.load(f)

        self.data = data_json['data']
        self.data = self.process_data(self.data)

    def process_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename):
        image = Image.open(filename)
        if self.corruption is not None:
            image = self.visual_corruption(image, self.severity)

        return image

    def get_wav(self, filename):
        waveform, sr = torchaudio.load(filename)

        if self.corruption is not None:
            waveform = self.audio_corruption(waveform, self.severity)

        waveform = waveform.numpy().T

        buffer = io.BytesIO()
        soundfile.write(buffer, waveform, sr, format='WAV')
        buffer.seek(0)

        return buffer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        datum = self.data[index]
        datum = self.decode_data(datum)

        image_path = datum['video_path'] + f'/frame_{self.frame_num}/' + datum['video_id'] + '.jpg'
        image = self.get_image(image_path)

        audio_path = datum['wav']
        audio = self.get_wav(audio_path)

        return image, audio
