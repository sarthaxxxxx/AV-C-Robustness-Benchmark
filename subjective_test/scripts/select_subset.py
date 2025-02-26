import os
import pandas as pd
import random

path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/selected_labels.csv'
df = pd.read_csv(path)

df = df.groupby("label").sample(n=3).reset_index(drop=True)

path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/subset'

noises = ["gaussian_noise", "interference", "rain"]

df["noise"] = None
for label in df["label"].unique():
    label_indices = df[df["label"] == label].index.tolist()
    random.shuffle(noises)  # Shuffle noises for randomness
    for idx, noise in zip(label_indices, noises):
        df.at[idx, "noise"] = noise

df['label'] = df['label'].apply(lambda x: x.replace(' ', '_')).reset_index(drop=True)

save_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/subjective_test/final_list.csv'
df.to_csv(save_path, index=False)