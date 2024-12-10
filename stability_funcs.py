import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.dataloaders import get_dataloader
from tqdm import tqdm
from utils.game_engine import history_to_legal_moves
import math
import matplotlib.pyplot as plt
from torcheval.metrics import BinaryAUROC

device='cuda' if torch.cuda.is_available() else 'cpu'

def compute_stability_maps(all_boards):
    num_states, _ = all_boards.shape
    
    stability_maps = torch.zeros_like(all_boards)
    
    for state_idx in range(num_states):
        board = all_boards[state_idx].reshape(8, 8)
        stability_map = np.zeros((8, 8))
        
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)] # corners always stable

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
    
    return stability_maps # same shape as all_boards


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

def evaluate_all_stability_classification_across_seeds(activations_all, stability_maps, layer):
    final_aurocs_across_seeds = None
    final_f1_scores_across_seeds = None
    final_dict_across_seeds = dict()

    num_seeds = len(activations_all)

    for seed, activations in enumerate(activations_all, start=1):
        print(f"Processing seed {seed}")

        aurocs, f1_scores, current_dict = evaluate_all_stability_classification(activations, stability_maps, layer, seed)

        if final_aurocs_across_seeds is None:
            final_aurocs_across_seeds = aurocs.clone()
            final_f1_scores_across_seeds = f1_scores.clone()
        else:
            final_aurocs_across_seeds += aurocs
            final_f1_scores_across_seeds += f1_scores

        for feature, tile_dict in current_dict.items():
            if feature not in final_dict_across_seeds:
                final_dict_across_seeds[feature] = {}

            for tile, auroc_value in tile_dict.items():
                if tile not in final_dict_across_seeds[feature]:
                    final_dict_across_seeds[feature][tile] = auroc_value
                else:
                    final_dict_across_seeds[feature][tile] += auroc_value

    final_aurocs_across_seeds /= num_seeds
    final_f1_scores_across_seeds /= num_seeds

    averaged_dict = {
        feature: {
            tile: auroc_value / num_seeds
            for tile, auroc_value in tile_dict.items()
        }
        for feature, tile_dict in final_dict_across_seeds.items()
    }

    torch.save(final_aurocs_across_seeds, f"analysis_results/dictionaries/contents_aurocs_layer{layer}.pkl")
    torch.save(final_f1_scores_across_seeds, f"analysis_results/dictionaries/contents_f1_scores_layer{layer}.pkl")
    torch.save(averaged_dict, f"analysis_results/dictionaries/final_dict_layer{layer}.pkl")

    print(f"Averaged results saved for layer {layer}")


def find_top_aurocs_contents(layer):
    with open(f"analysis_results/dictionaries/contents_aurocs_layer{layer}.pkl", 'rb') as f:
        aurocs = torch.load(f)

    top_values, top_indices = torch.topk(aurocs.flatten(), k=50)

    top_locations = [index_to_pair(idx.item(), aurocs.shape) for idx in top_indices]

    print("(Feature_number, tile_number), AUROC:")
    for (feature_number, tile_number), auroc in zip(top_locations, top_values):
        print(f"({feature_number}, {tile_number}), {auroc:.4f}")

    return top_locations, top_values


def compare_top_features_stability(layer, auroc_threshold):
    top_locations, top_values = find_top_aurocs_contents(layer)

    tile_numbers = [tile_number for (feature_number, tile_number), auroc in zip(top_locations, top_values) if auroc >= auroc_threshold]
    print(tile_numbers)

    counts = count_frequencies(tile_numbers)
    print("\n".join(f"{pos}: {freq}" for pos, freq in sorted(counts.items(), key=lambda x: x[1], reverse=True)))

    data_to_plot = torch.zeros((8, 8))
    for tile_number, freq in counts.items():
        row = tile_number // 8
        col = tile_number % 8
        data_to_plot[row][col] = freq

    fig, ax = plt.subplots()
    im = ax.imshow(data_to_plot)

    ax.set_xticks(range(8), labels=list("ABCDEFGH"))
    ax.set_yticks(range(8), labels=range(1, 9))

    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, int(data_to_plot[i, j]), ha="center", va="center", color="w")

    ax.set_title(f"Frequency of Top Features Predicting Stability for Layer {layer}")
    fig.tight_layout()
    plt.show()
    plt.savefig(f"analysis_results/figures/feature_frequencies_stability_layer{layer}.png")
    plt.close()


def save_activations_boards_and_legal_moves(sae_location="saes/sae_layer_6.pkl", eval_dataset_type="probe_test", offset=0, save_directory="analysis_results", layer=1, seed=1):
    torch.manual_seed(1)
    with open(sae_location, 'rb') as f:
        sae=torch.load(f, map_location=device)
    test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=sae.window_length, batch_size=10))
    activations=[]
    boards=[]
    legal_moves=[]
    for test_input, test_labels in tqdm(test_dataloader):
        (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, normalized_logits), total_loss=sae(test_input, None)
        activations.append(hidden_layer)
        boards.append(sae.trim_to_window(test_labels, offset=offset)) # index n is the board state after before move n+start_window_length
        legal_moves.append(sae.trim_to_window(history_to_legal_moves(test_input.cpu()), offset=offset)) #index in are the legal moves on turn n+1+start_window_length
    all_activations=torch.cat(activations).flatten(end_dim=-2) # shape (dw,f), where d= dataset size (2000), w=trimmed window length (52), f=num_features (1024)
    
    with open(f"{save_directory}/layer_{layer}/activations_layer{layer}_seed{seed}.pkl", 'wb') as f:
        torch.save(all_activations, f)
