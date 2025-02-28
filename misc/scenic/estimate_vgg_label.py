import pandas as pd

path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/misc/scenic/data/vgg_train_subset_pred.csv'
df = pd.read_csv(path)

label_pred_dict = (
    df.groupby("label_equiav")["pred"]
    .apply(lambda x: x.value_counts().index.tolist())  # Sort `pred` by frequency
    .to_dict()
)

sorted_label_pred_dict = dict(sorted(label_pred_dict.items(), key=lambda item: len(item[1])))

final_map = {}
used_preds = set()

#initialize with -1 pred
for label, _ in sorted_label_pred_dict.items():
    final_map[label] = -1

for label, preds in sorted_label_pred_dict.items():
    for pred in preds:
        if pred not in used_preds:
            final_map[label] = pred
            used_preds.add(pred)
            break

#randomly assign the remaining preds
all_preds = set(range(309))
unused_preds = all_preds - used_preds

for k,v in final_map.items():
    if v == -1:
        final_map[k] = unused_preds.pop()