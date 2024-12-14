import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataloaders import get_dataloader
from tqdm import tqdm
from utils.game_engine import history_to_legal_moves
import math
import matplotlib.pyplot as plt
from torcheval.metrics import BinaryAUROC
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import OrderedDict

device='cuda' if torch.cuda.is_available() else 'cpu'

def compute_stability_maps(all_boards):
    num_states, _ = all_boards.shape
    
    stability_maps = torch.zeros_like(all_boards)
    
    for state_idx in range(num_states):
        board = all_boards[state_idx].reshape(8, 8)
        stability_map = np.zeros((8, 8))
        
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        for x, y in corners:
            if board[x, y] != 0:
                stability_map[x, y] = 1
        
        for x in [0, 7]:
            for y in range(8):
                if board[x, y] != 0 and (
                    (y > 0 and stability_map[x, y - 1] == 1) or
                    (y < 7 and stability_map[x, y + 1] == 1)
                ):
                    stability_map[x, y] = 1
        
        for y in [0, 7]:
            for x in range(8):
                if board[x, y] != 0 and (
                    (x > 0 and stability_map[x - 1, y] == 1) or
                    (x < 7 and stability_map[x + 1, y] == 1)
                ):
                    stability_map[x, y] = 1
        
        for x in range(1, 7):
            for y in range(1, 7):
                if board[x, y] != 0:
                    neighbors = [
                        (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                        (x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)
                    ]
                    if all(0 <= nx < 8 and 0 <= ny < 8 and stability_map[nx, ny] == 1 for nx, ny in neighbors):
                        stability_map[x, y] = 1
        
        stability_maps[state_idx] = torch.tensor(stability_map.flatten(), dtype=torch.int32)
    
    return stability_maps


def calculate_f1_score(tp, fp, tn, fn):
    return 2*tp/(2*tp+fp+fn)


def index_to_pair(index, original_shape):
    num_tiles = original_shape[1]
    feature_number = index // num_tiles
    tile_number = index % num_tiles
    return (feature_number, tile_number)


def count_frequencies(iterable):
    to_return={}
    for x in iterable:
        if x in to_return:
            to_return[x]+=1
        else:
            to_return[x]=1
    return to_return


def evaluate_all_stability_classification(activations, stability_maps, layer, seed):
    final_dict = dict()
    f1_scores = torch.zeros((activations.shape[1], stability_maps.shape[1]))  # (num_features, num_tiles)
    aurocs = torch.zeros((activations.shape[1], stability_maps.shape[1]))  # (num_features, num_tiles)

    # Iterate over features and tiles
    for i, feature_activation in tqdm(enumerate(activations.transpose(0, 1))):
        
        features_dict = dict()

        for j, stability_map in enumerate(stability_maps.transpose(0, 1)):
            is_feature_active = feature_activation > 0
            is_stable_tile = stability_map == 1
            ended_game_mask = stability_map > -100

            tp = (is_feature_active * is_stable_tile * ended_game_mask).sum()
            fp = (is_feature_active * ~is_stable_tile * ended_game_mask).sum()
            tn = (~is_feature_active * ~is_stable_tile * ended_game_mask).sum()
            fn = (~is_feature_active * is_stable_tile * ended_game_mask).sum()

            f1_score = calculate_f1_score(tp, fp, tn, fn)
            f1_scores[i, j] = float(f1_score)

            metric = BinaryAUROC()
            metric.update(feature_activation[ended_game_mask], is_stable_tile[ended_game_mask].int())

            auroc_value = float(metric.compute())

            aurocs[i, j] = auroc_value
            features_dict[j] = auroc_value
        
        final_dict[i] = features_dict

    torch.save(aurocs, f"analysis_results/layer_{layer}/contents_aurocs_layer{layer}_seed{seed}.pkl")
    torch.save(f1_scores, f"analysis_results/layer_{layer}/contents_f1_scores_layer{layer}_seed{seed}.pkl")
    torch.save(final_dict, f"analysis_results/layer_{layer}/aurocs_dict_layer{layer}_seed{seed}.pkl")
    
    return aurocs, f1_scores, final_dict


def find_top_aurocs_contents(layer, seed):
    with open(f"analysis_results/layer_{layer}/contents_aurocs_layer{layer}_seed{seed}.pkl", 'rb') as f:
        aurocs = torch.load(f)

    top_values, top_indices = torch.topk(aurocs.flatten(), k=50)

    top_locations = [index_to_pair(idx.item(), aurocs.shape) for idx in top_indices]

    for (feature_number, tile_number), auroc in zip(top_locations, top_values):
        print(f"({feature_number}, {tile_number}), {auroc:.4f}")

    return top_locations, top_values


def compare_top_features_stability(layer, seed, auroc_threshold):
    top_locations, top_values = find_top_aurocs_contents_individual(layer, seed)

    tile_numbers = [tile_number for (feature_number, tile_number), auroc in zip(top_locations, top_values) if auroc >= auroc_threshold]
    print(tile_numbers)

    counts = count_frequencies(tile_numbers)

    data_to_plot = torch.zeros((8, 8))
    for tile_number, freq in counts.items():
        row = tile_number // 8
        col = tile_number % 8
        data_to_plot[row][col] = freq

    colors = ["#440154", "#30678D", "#35B778", "#FDE724", "#ffccb3"]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    im = ax.imshow(data_to_plot, cmap=cmap, norm=norm)

    ax.set_xticks(range(8), labels=list("ABCDEFGH"))
    ax.set_yticks(range(8), labels=range(1, 9))

    for i in range(8):
        for j in range(8):
            freq = int(data_to_plot[i, j])
            text = ax.text(j, i, freq, ha="center", va="center", color="white")

    ax.set_title(f"Frequency of Top Features Predicting Stability for Layer {layer} | Seed {seed}")
    fig.tight_layout()
    plt.show()
    plt.close()

def get_tile_feature_map_for_seed(layer, seed, threshold):
    """
    Given a layer, a seed, and a threshold, return a dictionary mapping:
    tile_number -> list_of_features that surpass the given AUROC threshold.
    """
    aurocs = torch.load(f"analysis_results/layer_{layer}/contents_aurocs_layer{layer}_seed{seed}.pkl")

    num_tiles = aurocs.shape[1]
    tile_to_features = {tile_idx: [] for tile_idx in range(num_tiles)}

    for feature_idx in range(aurocs.shape[0]):
        for tile_idx in range(num_tiles):
            if aurocs[feature_idx, tile_idx].item() >= threshold:
                tile_to_features[tile_idx].append(feature_idx)

    return tile_to_features
