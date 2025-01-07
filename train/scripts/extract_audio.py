import os
import moviepy
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

video_pth = '/mnt/user/saksham/AV_robust/data/AVSync15/videos'
sound_list = os.listdir(video_pth)
save_pth = '/mnt/data2/saksham/AV_robust/avsync15/audio'
print(sound_list)

for sound in sound_list:
    audio_pth = os.path.join(video_pth, sound)
    lis = os.listdir(audio_pth)
    if not os.path.exists(os.path.join(save_pth, sound)):
        os.makedirs(os.path.join(save_pth, sound))
    exist_lis = os.listdir(os.path.join(save_pth, sound))
    for audio_id in lis:
        if audio_id[-4:] != ".mp4":
            continue
        name = os.path.join(video_pth, sound, audio_id)
        try:
            video = VideoFileClip(name)
            audio = video.audio
            audio_name = audio_id[:-4] + '.wav'
            if audio_name in exist_lis:
                continue
            audio.write_audiofile(os.path.join(save_pth, sound, audio_name), fps=16000)
            print("finish video id: " + audio_name)
        except:
            print(name)


