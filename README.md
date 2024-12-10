# OthelloSAE
CS194-196 Course Project

## 1 Dependencies
Make sure to install `neel-plotly`. Clone from https://github.com/neelnanda-io/neel-plotly.git, then `cd neel-plotly && pip install -e .`

SAE-related files: Download datasets and trained models from https://drive.google.com/drive/folders/1xMkEctaqAUjoPXGY-9dBu-pE3SJjKx2K, place it in `/`(root folder).

Download trimmed trained linear probe weights at https://drive.google.com/drive/folders/1hYbOP4tzHeRmnxmu2rTO6v-qNaGMsO5Q?usp=sharing, place it in `/probes`.

Download self-trained OthelloGPT weights at https://drive.google.com/drive/folders/1LBu8BivQX1fO2yEV1OZdpNwTEoTUJLlm?usp=sharing, place it in `/`(root folder).

## 2 Training

### 2.1 Train OthelloGPT
We adopted OthelloGPT training code from existing code base. You can find fucntion `full_scale_training` in `model_training.py`. Change `num_layers` value in the code to train original OthelloGPT(8 layer) or bigger OthelloGPT in our experiment(12 layer).

### 2.2 Train Linear Probes
We adopted and modified linear probe training code from existing code base. You can find function `full_probe_run` in `model_training.py`. We trimmed down the size of the linear probe by removing the OthelloGPT weights inside the original implementation.

## 3 Experiments

### 3.1 Linear probe related experiments
Every block of code in `experiment.ipynb` have explanations about the usage of the code. See the `ipynb` file for more details.

### 3.2 Cosine Similarity
Every block of code in `cosine_similarity.ipynb` have explanations about the usage of the code. See the `ipynb` file for more details.

### 3.3 SAE stability
You can find all functions related to SAE stability in `stability_funcs.py`.

`save_activations_boards_and_legal_moves`: Get activations for a specific layer and save it as a `.pkl` file. (This part is the same for tile color, functionality might be repetitive.)

`compute_stability_maps`: Get stability maps for stability classifications.


`evaluate_all_stability_classification`: You can get AUROCs of a specific layer and a specific seed from the activations and the stability maps.

`compare_top_features_stability`: Get visualizations of top features predicting stability frequency with a specific layer and AUROC threshold as input.

Our usage: First run `save_activations_boards_and_legal_moves` to get activations for your specific layer. Pass `all_boards` into `compute_stability_maps` to get `stability_maps`. Run `evaluate_all_stability_classification` if you’re doing one seed by passing in activations, stability_maps, layer, seed or `evaluate_all_stability_classification_across_seeds` if you wanna do all seeds. Finally run `compare_top_features_stability` with a specific layer and AUROC threshold to get visualizations.
