import os
import random
import numpy as np
import torch
from .base import BaseDataset

class MUSICDataset(BaseDataset):
    def __init__(self, meta_path, opt, **kwargs):
        super(MUSICDataset, self).__init__(
            meta_path, opt, **kwargs)
        self.fps = opt.frameRate
        self.audio_path = opt.audio_path
        self.frame_path = opt.frame_path
        self.categories = opt.categories

    def encode(self, id):
        """ label encoding
            Returns:
              1d array, multimonial representation, e.g. [1,0,1,0,0,...]
            """
        categories = self.categories
        id_to_idx = {id: index for index, id in enumerate(categories)}
        index = id_to_idx[id]
        label = torch.from_numpy(np.array(index)).long()
        return label

    def __getitem__(self, index):

        # the first video
        info = self.list_sample[index]
        cls = info[1]
        count_frames = info[2]
        label = self.encode(cls)

        if not self.split == 'train':
            random.seed(index)

        if self.split == 'train':
            center_frame = random.randint(1, int(count_frames))
        else:
            center_frame = int(count_frames) // 2

        path_frame = os.path.join(self.frame_path, info[0]+'.mp4', '{:06d}.jpg'.format(center_frame))
        path_audio = os.path.join(self.audio_path, info[0] + '.wav')

        # import pdb; pdb.set_trace()
        frame = self._load_frames(path_frame)
        center_time = (center_frame - 0.5) / self.fps
        audio = self._load_audio(path_audio, center_time)

        # # load frames and audios, STFT
        # try:
        #     frame = self._load_frames(path_frame)
        #     center_time = (center_frame - 0.5) / self.fps
        #     audio = self._load_audio(path_audio, center_time)

        # except Exception as e:
        #     print('Failed loading frame/audio: {}'.format(e))

        #     raise e

        ret_dict = {'frames': frame, 'audios': audio, 'labels':label}
        if self.split != 'train':
            ret_dict['info'] = info

        return ret_dict