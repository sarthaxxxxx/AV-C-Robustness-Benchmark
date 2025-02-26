import gradio as gr
import pandas as pd
import os
import random
import json

base_vid_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/data/subset'
meta_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/final_list.csv'
save_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/data/annotations'

df = pd.read_csv(meta_path)
df = df.sample(frac=1).reset_index(drop=True)
label_vid_list = list(df['label_vid'])

labels = ['baby crying', 'dog barking', 'driving motorcycle', \
            'lawn mowing', 'people clapping', 'people whistling', \
            'playing badminton', 'playing bass guitar', 'playing table tennis', \
            'tractor digging']

DRY_RUN = False
if DRY_RUN:
    label_vid_list = label_vid_list[:5]
    save_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/data/annotations/dry_run'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_videos(name, vid_path, selected_label):
    name = name.strip().replace(' ', '_').lower()
    user_dir = os.path.join(save_dir, name)
    ensure_dir(user_dir)

    seen_indices_file = os.path.join(user_dir, 'seen_indices.json')
    if os.path.exists(seen_indices_file):
        seen_indices = json.load(open(seen_indices_file))
    else:
        seen_indices = []

    if vid_path is not None:
        label_vid = os.path.basename(vid_path).rsplit('.', 1)[0]
        print(name, label_vid, selected_label)

        save_data = {}
        save_data['label_vid'] = label_vid
        save_data['selected_label'] = selected_label
        save_path = os.path.join(user_dir, label_vid+'.json')
        json.dump(save_data, open(save_path, 'w'))

        seen_indices.append(label_vid)
        json.dump(seen_indices, open(seen_indices_file, 'w'))

    unseen_indices = list(set(label_vid_list) - set(seen_indices))

    if unseen_indices:
        curr_vid = random.choice(unseen_indices)
    else:
        return "",""

    curr_vid_path = os.path.join(base_vid_dir, curr_vid+'.mp4')
    updated_label = gr.update(value=labels)

    return curr_vid_path, updated_label
 
with gr.Blocks() as demo:

    with gr.Row():
        gr.Markdown("## 🎬 Welcome to the Analysis Interface")
        gr.Markdown("""
        ### 📝 Guidelines:
        - Type your name and click submit
        - You can come back to this interface anytime, just use the exact same name you used before.
        - There are 30 video in total and <strong>you would see an ERROR message at the end</strong>.
        """)

    with gr.Row():        
        name = gr.Textbox(label="Type in your full name with _ between (Eg. John_Doe)")
 
    with gr.Row():
        vid_path = gr.Video(height=448, width=224)
        selected_label = gr.Radio(labels)

    submit_bttn = gr.Button("Submit")
 
    submit_bttn.click(
        get_videos,
        inputs=[name, vid_path, selected_label],
        outputs=[vid_path, selected_label]
    )
 
demo.launch(share=True)
# demo.launch()