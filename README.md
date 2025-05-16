<div align="center">

## Audio-Visual Robustness Benchmark to Common Corruptions


</div>


### Step 1 : Create AV-C benchmark dataset and JSON files.

- Run dataset_corruptions.sh in data_recipe/src to create the corruption dataset (would have to tweak the paths, etc). 
- Next, for your dataset, you WOULD need a reference_json file (as in data/vgg_test_refer.json). Same style, pls!
- Third, run create_clean_json.py (MANDATORY) and link this reference_json file. After that, run create_both_c_json.py. This would create json files for your new dataset.

