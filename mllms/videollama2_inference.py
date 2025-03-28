'''

how to download ckpt:

huggingface-cli download DAMO-NLP-SG/VideoLLaMA2.1-7B-16F --local-dir <YOUR PATH>

How to run :

CUDA_VISIBLE_DEVICES=1 python inference.py --model-path /mnt/ssd0/saksham/av_robust/VideoLLaMA2.1-7B-AV --modal-type av

'''


import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import argparse

def inference(args):

    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)

    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    # Audio-visual Inference

    audio_video_path = "assets/8wsYT1PRjIg_000074.mp4"
    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    if args.modal_type == "a":
        audio_video_tensor = preprocess(audio_video_path)
    else:
        audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
    # question = f"Who plays the instrument louder?"
    question = f"Which class does the video belong to in VGGSound?"

    output = mm_infer(
        audio_video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='audio' if args.modal_type == "a" else "video",
        do_sample=False,
    )

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', required=True)
    args = parser.parse_args()

    inference(args)
