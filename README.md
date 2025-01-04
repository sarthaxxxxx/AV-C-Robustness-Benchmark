<div align="center">

## Audio-Visual Robustness Benchmark to Common Corruptions

(Work in progress - feel free to build on top of this!) - JUST A STARTER CODE.

</div>


### Step 1 : Create AV-C benchmark dataset and JSON files.

- Run dataset_corruptions.sh in data_recipe/src to create the corruption dataset (would have to tweak the paths, etc). 
- Next, for your dataset, you WOULD need a reference_json file (as in data/vgg_test_refer.json). Same style pls!
- Third, run create_clean_json.py (MANDATORY) and link this reference_json file. After that, run create_both_c_json.py. This would create json files for your new dataset. Pls go through the code and make changes, as required. This worked in my case, might not work on yours. We'll figure out the final version later.


### Step 2 : src/ houses task wise models.
- As of now, CAV-MAE and UAVM are there for classification. Still building on this. You can add yours, as per the AV task.