import gradio as gr
import pandas as pd
import os
import random
import json

base_vid_dir = '/mnt/data2/wpian/VGGSound/VGGSound'
meta_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/qualitative_analysis/data/vgg_test_vqgan.csv'
save_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/qualitative_analysis/data'

df = pd.read_csv(meta_path)
vid_label_map = {df['vid'][i]: df['label'][i] for i in range(len(df))}

DRY_RUN = False
if DRY_RUN:
    sampled_keys = random.sample(list(vid_label_map.keys()), 10)
    vid_label_map = {k: vid_label_map[k] for k in sampled_keys}
    # df = df.sample(10, random_state=42).reset_index(drop=True)
    save_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/qualitative_analysis/data/dry_run'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_videos(name, aq, vq, comments, vid):
    name = name.strip().replace(' ', '_').lower()
    user_dir = os.path.join(save_dir, name)
    ensure_dir(user_dir)

    print(name, vid, aq, vq, comments)

    seen_indices_file = os.path.join(user_dir, 'seen_indices.json')
    if os.path.exists(seen_indices_file):
        seen_indices = json.load(open(seen_indices_file))
    else:
        seen_indices = []

    if vid is not None:
        save_data = {}
        save_data['vid'] = vid
        save_data['aq'] = aq
        save_data['vq'] = vq
        save_data['comments'] = comments
        save_path = os.path.join(user_dir, vid+'.json')
        json.dump(save_data, open(save_path, 'w'))

        seen_indices.append(vid)
        json.dump(seen_indices, open(seen_indices_file, 'w'))

    unseen_indices = list(set(vid_label_map.keys()) - set(seen_indices))

    if unseen_indices:
        curr_vid = random.choice(unseen_indices)
    else:
        return "","","","","",""

    curr_vid_path = os.path.join(base_vid_dir, curr_vid+'.mp4')
    curr_label = vid_label_map[curr_vid]

    updated_vq = gr.update(value=3, label="Video Quality")
    updated_aq = gr.update(value=3, label="Audio Quality")

    return curr_vid_path, gr.update(value=curr_vid), gr.update(value=curr_label), updated_vq, updated_aq, gr.update(value=None)
 
with gr.Blocks() as demo:

    with gr.Row():
        gr.Markdown("## 🎬 Welcome to the Video Analysis Interface")
        gr.Markdown("""
        ### 📝 Guidelines:
        Type your name and click submit
        P.S. You can come back to this interface anytime, just use the exact same name you used before.
        """)

    with gr.Row():        
        name = gr.Textbox(label="Type in your full name with _ between (Eg. John_Doe)")
 
    with gr.Row():
        video = gr.Video(height=400, width=500)
        with gr.Column():
            vid = gr.Label(label="Video ID: ")
            label = gr.Label(label="Class Label: ")

    with gr.Row():
        with gr.Column():
            vq = gr.Slider(minimum=1, maximum=5, step=1, label="Video Quality")
            aq = gr.Slider(minimum=1, maximum=5, step=1, label="Audio Quality")
        
        with gr.Column():
            comments = gr.Textbox(label="Comments regarding noise?", placeholder="What noises are present in the video/audio?")

    submit_bttn = gr.Button("Submit")
 
    submit_bttn.click(
        get_videos,
        inputs=[name, vq, aq, comments, vid],
        outputs=[video, vid, label, vq, aq, comments]
    )
 
demo.launch(share=True)
# demo.launch()