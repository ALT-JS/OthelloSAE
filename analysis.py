import torch
import matplotlib.pyplot as plt
from utils.dataloaders import get_dataloader
import utils.dataloaders
from tqdm import tqdm
from utils.game_engine import history_to_legal_moves
import utils.game_engine
from torcheval.metrics import BinaryAUROC
import os
import numpy as np
import seaborn as sns

device='cuda' if torch.cuda.is_available() else 'cpu'

def analyze_cosine_similarities(autoencoder_location="saes/sae.pkl", rectify=False, include_control=False):
    '''
    cosine similarities between autoencoder features and linear probe directions
    '''
    probe_directions=get_probe_directions(rectify=rectify)
    autoencoder_directions=get_autoencoder_directions(autoencoder_location)
    cosine_similarities=probe_directions@autoencoder_directions.transpose(0,1)
    max_cosine_similarities_of_probes=cosine_similarities.max(dim=1).values
    fig, ax=plt.subplots()
    plt.hist(max_cosine_similarities_of_probes, label="Probe/Autoencoder")
    plt.ylabel("Absolute Frequency")
    
    ax2=ax.twinx()
    if include_control:
        random_directions=get_random_directions(autoencoder_directions.shape)
        control_cosine_similarities=probe_directions@random_directions.transpose(0,1)
        max_cosine_similarities_control=control_cosine_similarities.max(dim=1).values
        sns.kdeplot(max_cosine_similarities_control, label="Probe/Random", ax=ax2, color='orange')
        plt.ylabel("Relative Frequency")
    plt.show()
    fig.legend(bbox_to_anchor=(0.4, 0.38, 0.5, 0.5))
    plt.title("Maximum Cosine Similarities between probe and feature directions")
    # plt.savefig("analysis_results/figures/hist_cosine_similarities.png")

def analyze_cosine_similarities_2(autoencoder_location="saes/sae.pkl"):
    '''
    cosine similarities between autoencoder features and model unembed directions
    '''
    model_directions=get_model_unembed_directions()
    autoencoder_directions=get_autoencoder_directions(autoencoder_location)
    cosine_similarities=model_directions@autoencoder_directions.transpose(0,1)
    max_cosine_similarities_of_probes=cosine_similarities.max(dim=1).values
    plt.hist(max_cosine_similarities_of_probes)
    plt.show()


def analyze_cosine_similarities_3(rectify=False):
    '''
    cosine similarities between probe directions and themselves
    '''
    probe_directions=get_probe_directions(rectify=rectify)

    cosine_similarities=probe_directions@probe_directions.transpose(0,1)
    cosine_similarities_masked_self=cosine_similarities-torch.eye(cosine_similarities.shape[0])
    max_cosine_similarities_of_probes=cosine_similarities_masked_self.max(dim=1).values


    plt.hist(max_cosine_similarities_of_probes.flatten())
    # plt.show()
    plt.savefig("analysis_results/figures/probe_probe_max_cosine_similarities.png")

def get_random_directions(shape):
    directions=torch.normal(torch.zeros(shape), torch.ones(shape))
    directions=normalize_rows(directions)
    return directions

def get_probe_directions(probe_location="probes/probe_layer_6.pkl", rectify=False):
    with open(probe_location, 'rb') as f:
        probe=torch.load(f, map_location=device)
    original_probe_directions=torch.stack([x.weight for x in probe.classifier])
    if rectify:
        original_probe_directions=1.5*original_probe_directions-0.5*original_probe_directions.sum(dim=0)
    probe_directions=original_probe_directions.flatten(end_dim=-2)
    probe_directions=normalize_rows(probe_directions)
    return probe_directions.detach()

def get_autoencoder_directions(autoencoder_location="saes/sae_layer_6.pkl"):
    with open(autoencoder_location, 'rb') as f:
        sae=torch.load(f, map_location=device)
    sae_directions=sae.encoder_decoder_matrix.data.transpose(0,1)
    sae_directions=normalize_rows(sae_directions)
    return sae_directions

def get_model_unembed_directions(model_location="trained_model_full.pkl"):
    with open(model_location, 'rb') as f:
        othello_gpt=torch.load(f, map_location=device)
    model_directions=othello_gpt.unembed.weight.data
    model_directions=normalize_rows(model_directions)
    return model_directions

def normalize_rows(data):
    return data/data.norm(p=2, dim=1, keepdim=True)

def calculate_f1_score(tp, fp, tn, fn):
    return 2*tp/(2*tp+fp+fn)

def evaluate_all_content_classification(directory="analysis_results", activations_file_name="all_activations.pkl", boards_file_name="all_boards.pkl"):
    with open(f"{directory}/{activations_file_name}", 'rb') as f:
        all_activations=torch.load(f)
    with open(f"{directory}/{boards_file_name}", 'rb') as f:
        all_boards=torch.load(f)
    f1_scores=torch.zeros((all_activations.shape[1], all_boards.shape[1], 3))
    aurocs=torch.zeros((all_activations.shape[1], all_boards.shape[1], 3))
    for i, feature_activation in tqdm(enumerate(all_activations.transpose(0,1))):
        for j, board_position in enumerate(all_boards.transpose(0,1)):
            for k, piece_class in enumerate([0,1,2]):
                is_feature_active=feature_activation>0
                is_target_piece=board_position==piece_class
                ended_game_mask= board_position>-100
                tp=(is_feature_active*is_target_piece* ended_game_mask).sum()
                fp=(is_feature_active* ~is_target_piece* ended_game_mask).sum()
                tn=(~is_feature_active* ~is_target_piece* ended_game_mask).sum()
                fn=(~is_feature_active*is_target_piece* ended_game_mask).sum()
                f1_score=calculate_f1_score(tp, fp, tn, fn)
                f1_scores[i,j,k]=float(f1_score)
                metric = BinaryAUROC()
                metric.update(feature_activation[ended_game_mask], is_target_piece[ended_game_mask].int())
                aurocs[i,j,k]=float(metric.compute())
                # print(f"F1 Score: {f1_score:.4f}")
    with open(f"{directory}/contents_aurocs.pkl", 'wb') as f:
        torch.save(aurocs, f)
    with open(f"{directory}/contents_f1_scores.pkl", 'wb') as f:
        torch.save(f1_scores, f)


def evaluate_all_legal_moves_classification(activations_location="analysis_results/all_activations.pkl", legal_moves_location="analysis_results/all_legal_moves.pkl"):
    with open(activations_location, 'rb') as f:
        all_activations=torch.load(f)
    with open(legal_moves_location, 'rb') as f:
        all_legal_moves=torch.load(f)
    f1_scores=torch.zeros((all_activations.shape[1], all_legal_moves.shape[1], 2))
    aurocs=torch.zeros((all_activations.shape[1], all_legal_moves.shape[1], 2))
    for i, feature_activation in tqdm(enumerate(all_activations.transpose(0,1))):
        for j, board_position in enumerate(all_legal_moves.transpose(0,1)):
            for k, legality_class in enumerate([0,1]):
                is_feature_active=feature_activation>0
                is_target_legality=board_position==legality_class
                ended_game_mask= board_position>-100
                tp=(is_feature_active*is_target_legality* ended_game_mask).sum()
                fp=(is_feature_active* ~is_target_legality* ended_game_mask).sum()
                tn=(~is_feature_active* ~is_target_legality* ended_game_mask).sum()
                fn=(~is_feature_active*is_target_legality* ended_game_mask).sum()
                f1_score=calculate_f1_score(tp, fp, tn, fn)
                f1_scores[i,j,k]=float(f1_score)
                metric = BinaryAUROC()
                metric.update(feature_activation[ended_game_mask], is_target_legality[ended_game_mask].int())
                aurocs[i,j,k]=float(metric.compute())
                # print(f"F1 Score: {f1_score:.4f}")
    with open("analysis_results/legal_aurocs.pkl", 'wb') as f:
        torch.save(aurocs, f)
    with open("analysis_results/legal_f1_scores.pkl", 'wb') as f:
        torch.save(f1_scores, f)

def evaluate_all_probe_classification(activations_location="analysis_results/probe_activations.pkl", board_locations="analysis_results/probe_all_boards.pkl"):
    with open(activations_location, 'rb') as f:
        probe_activations=torch.load(f)
    with open(board_locations, 'rb') as f:
        all_boards=torch.load(f)
    reshaped_activations=probe_activations.transpose(0,1).transpose(1,2)
    aurocs=torch.zeros((reshaped_activations.shape[0:2]))
    for board_position, all_class_logits in tqdm(enumerate(reshaped_activations)):
        boards=all_boards[:, board_position]
        all_class_odds=torch.nn.functional.softmax(all_class_logits, dim=0)
        for piece_class, this_class_odds in enumerate(all_class_odds):
            is_target_piece=boards==piece_class
            ended_game_mask= boards>-100
            metric = BinaryAUROC()
            metric.update(this_class_odds[ended_game_mask], is_target_piece[ended_game_mask].int())
            aurocs[board_position, piece_class]=float(metric.compute())
            # print(f"F1 Score: {f1_score:.4f}")
    with open("analysis_results/probe_aurocs.pkl", 'wb') as f:
        torch.save(aurocs, f)

def compare_probe_and_feature_aurocs():
    with open("analysis_results/legal_aurocs.pkl", 'rb') as f:
        legal_aurocs=torch.load(f)
    with open("analysis_results/contents_aurocs.pkl", 'rb') as f:
        contents_aurocs=torch.load(f)
    with open("analysis_results/probe_aurocs.pkl", 'rb') as f:
        probe_aurocs=torch.load(f)
    best_legal_aurocs=legal_aurocs.max(dim=0).values.flatten()
    best_contents_aurocs=contents_aurocs.max(dim=0).values.flatten()
    # best_legal_aurocs=legal_aurocs.flatten(start_dim=1).max(dim=1).values
    # best_contents_aurocs=contents_aurocs.flatten(start_dim=1).max(dim=1).values
    best_probe_aurocs=probe_aurocs.flatten()
    fig, ax = plt.subplots(1, 3, figsize=(18,4))

    plt.suptitle("AUROCs of...")
    
    plt.subplot(1, 3, 1, title="Probes on content prediction")
    plt.hist(x=best_probe_aurocs)

    plt.subplot(1, 3, 2, title="SAE Features on content prediction")
    plt.hist(x=best_contents_aurocs)

    plt.subplot(1, 3, 3, title="SAE Features on legal move prediction")
    plt.hist(x=best_legal_aurocs)

    plt.tight_layout()
    plt.savefig("analysis_results/figures/aurocs_comparison_hist.png")
    plt.close()



def plot_roc_curve(labels, scores):
    scores_sorted, order=torch.sort(scores, descending=True)
    labels_sorted=labels[order]
    points=[]
    tp=0
    fp=0
    previous_score=-1000000
    for score,label in zip(scores_sorted,labels_sorted):
        if score !=previous_score:
            points.append((tp,fp))
        previous_score=score
        if label:
            tp+=1
        else:
            fp+=1
    points.append((tp,fp))
    roc_y, roc_x=torch.tensor(points).transpose(0,1).float()
    roc_y/=tp
    roc_x/=fp
    plt.scatter(roc_x, roc_y)
    plt.show()

def create_roc(feature_number,board_position, piece_type):
    with open("analysis_results/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    scores=all_activations.transpose(0,1)[feature_number]
    labels=all_boards.transpose(0,1)[board_position]==piece_type
    plot_roc_curve(labels, scores)

def create_density_plot_contents(feature_number, board_position, separate_at_0=True):
    '''
    density plot for features as board classifiers
    '''
    with open("analysis_results/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    scores=all_activations.transpose(0,1)[feature_number].detach().numpy()
    labels=all_boards.transpose(0,1)[board_position]
    class_names=["empty", "own", "enemy"]
    empty_own_enemy_scores=[scores[np.where(labels==class_number)] for class_number in range(3)]

    if separate_at_0:
        fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 20]})
        ax=axs[1]
        axs[0].set_xticks([0],[0])
        for spine in axs[0].spines.values():
            spine.set_visible(False)
        empty_own_enemy_zero_scores=[class_scores[np.where(class_scores==0)] for class_scores in empty_own_enemy_scores]
        empty_own_enemy_zero_frequencies=[len(class_scores)/len(scores) for class_scores in empty_own_enemy_zero_scores]
        empty_own_enemy_nonzero_scores=[class_scores[np.where(class_scores!=0)] for class_scores in empty_own_enemy_scores]
        bottom=0
        axs[0].set_ylabel("Frequency")
        ax.set_ylabel(" ")

        for class_number, class_name in enumerate(class_names):
            sns.kdeplot(empty_own_enemy_nonzero_scores[class_number], fill=True, label=class_name, ax=axs[1])
            axs[0].bar(x=0, bottom=bottom, height=empty_own_enemy_zero_frequencies[class_number])
            bottom+=empty_own_enemy_zero_frequencies[class_number]
    else:
        fig, ax=plt.subplots()
        ax.set_ylabel("Frequency")
        sns.kdeplot(empty_own_enemy_scores, fill=True, label=["enemy", "empty", "own"])

    ax.set_xlabel("Feature Activation")
    plt.title(f'Feature {feature_number} activations against Position {board_position} contents')
    plt.legend()
    save_file_name=f"analysis_results/figures/hist_contents_feat_{feature_number}_pos_{board_position}.png"
    fig.savefig(save_file_name)

def create_density_plot_legal(feature_number, board_position, separate_at_0=True):
    '''
    histogram for features as playability classifiers
    '''
    with open("analysis_results/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/all_legal_moves.pkl", 'rb') as f:
        all_legal_moves=torch.load(f)
    scores=all_activations.transpose(0,1)[feature_number].detach().numpy()
    labels=all_legal_moves.transpose(0,1)[board_position]
    class_names=["illegal", "legal"]
    illegal_legal_scores=[scores[np.where(labels==class_number)] for class_number in range(2)]

    if separate_at_0:
        fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 20]})
        ax=axs[1]
        axs[0].set_xticks([0],[0])
        for spine in axs[0].spines.values():
            spine.set_visible(False)
        illegal_legal_zero_scores=[class_scores[np.where(class_scores==0)] for class_scores in illegal_legal_scores]
        illegal_legal_zero_frequencies=[len(class_scores)/len(scores) for class_scores in illegal_legal_zero_scores]
        illegal_legal_nonzero_scores=[class_scores[np.where(class_scores!=0)] for class_scores in illegal_legal_scores]
        bottom=0
        axs[0].set_ylabel("Frequency")
        ax.set_ylabel(" ")

        for class_number, class_name in enumerate(class_names):
            sns.kdeplot(illegal_legal_nonzero_scores[class_number], fill=True, label=class_name, ax=axs[1])
            axs[0].bar(x=0, bottom=bottom, height=illegal_legal_zero_frequencies[class_number])
            bottom+=illegal_legal_zero_frequencies[class_number]
    else:
        fig, ax=plt.subplots()
        ax.set_ylabel("Frequency")
        sns.kdeplot(illegal_legal_scores, fill=True, label=class_names)

    ax.set_xlabel("Feature Activation")
    plt.title(f'Feature {feature_number} activations against whether Position {board_position} is a legal move')
    plt.legend()
    save_file_name=f"analysis_results/figures/hist_legal_feat_{feature_number}_pos_{board_position}.png"
    fig.savefig(save_file_name)

    

def save_probe_data(probe_location="probes/probe_layer_6.pkl", offset=0):
    with open(probe_location, 'rb') as f:
        probe=torch.load(f, map_location=device)
    probe.print_evaluation(torch.tensor([1]), "probe_test")
    probe.save_state_on_dataset(offset=offset)

def create_hist_for_probe(board_position, primary_class=0, separate_at_0=False):
    with open("analysis_results/probe_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/probe_all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    n_bins=20
    # scores=torch.nn.functional.softmax(all_activations[:, board_position, :].transpose(0,1), dim=0)[primary_class]
    scores=2*all_activations[:,board_position,primary_class]-1*all_activations[:,board_position,:].sum(dim=-1)
    # scores=all_activations[:,board_position,primary_class]
    all_one_hot_predictions=torch.argmax(all_activations, dim=2)
    correctness=all_one_hot_predictions==all_boards
    this_position_labels=all_boards[:,board_position]
    class_names={0:"Empty", 1:"Own", 2:"Enemy"}
    empty, own, enemy=(scores[torch.where(this_position_labels==class_number)].detach().numpy() for class_number in class_names)
    fig, ax = plt.subplots()
    ax.hist((enemy, empty, own), n_bins, density=True, histtype='bar', stacked=True, label=["enemy", "empty", "own"])
    ax.set_xlabel("Feature Activation")
    ax.set_ylabel("Frequency")
    ax.set_title(f'Position {board_position} probe, trying to separate class {class_names[primary_class]}')
    plt.legend()
    save_file_name=f"analysis_results/figures/hist_probe_pos_{board_position}.png"
    fig.savefig(save_file_name)
    plt.close()

def create_triple_hist_for_probe(board_position, primary_class=0):
    with open("analysis_results/probe_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open("analysis_results/probe_all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    fig, ax=plt.subplots(1,3,figsize=(10,4))
    plt.suptitle(f"Position {board_position} rectified probe scores")
    for primary_class in range(3):
        # ax=plt.subplot(1,3,primary_class+1)
        scores=1.5*all_activations[:,board_position,primary_class]-0.5*all_activations[:,board_position,:].sum(dim=-1)
        this_position_labels=all_boards[:,board_position]
        class_names={0:"Empty", 1:"Own", 2:"Enemy"}
        empty, own, enemy=(scores[torch.where(this_position_labels==class_number)].detach().numpy() for class_number in class_names)
        # plt.hist((enemy, empty, own), n_bins, density=True, histtype='bar', stacked=True, label=["enemy", "empty", "own"])
        sns.set_style('whitegrid')
        sns.kdeplot(np.array(enemy), ax=ax[primary_class], fill=True, label="Enemy")
        sns.kdeplot(np.array(empty), ax=ax[primary_class], fill=True, label="Empty")
        sns.kdeplot(np.array(own), ax=ax[primary_class], fill=True, label="Own")

        # ax.set_xlabel("Feature Activation")
        ax[primary_class].set_title(f'Probe for class {class_names[primary_class]}')
        # if primary_class==0:
        #     # ax.set_ylabel("Frequency")
        #     ax.legend()
    ax[0].legend()
    plt.tight_layout()
    save_file_name=f"analysis_results/figures/triple_density_probe_pos_{board_position}.png"
    plt.savefig(save_file_name)
    plt.close()

def find_top_aurocs_contents(k=10, directory="analysis_results"):
    with open(f"{directory}/contents_aurocs.pkl", 'rb') as f:
        aurocs=torch.load(f)
    top_values, top_locations=torch.topk(aurocs.flatten(), k=k)
    top_locations_tuples=[location_to_triple(x, aurocs.shape) for x in top_locations]
    print("(Feature_number, board_position, class), AUROC:")
    print("\n".join([f"{top_locations_tuple}, {top_value:.4f}" for top_locations_tuple, top_value in zip(top_locations_tuples, top_values)]))
    return top_locations_tuples, top_values

def find_top_aurocs_legal(k=10, directory="analysis_results"):
    with open(f"{directory}/legal_aurocs.pkl", 'rb') as f:
        aurocs=torch.load(f)
    top_values, top_locations=torch.topk(aurocs.flatten(), k=k)
    top_locations_tuples=[location_to_triple(x, aurocs.shape) for x in top_locations]
    print("(Feature_number, board_position, class), AUROC:")
    print("\n".join([f"{top_locations_tuple}, {top_value:.4f}" for top_locations_tuple, top_value in zip(top_locations_tuples, top_values)]))
    return top_locations_tuples, top_values

def compare_legal_contents_aurocs(k=10, directory="analysis_results"):
    with open(f"{directory}/legal_aurocs.pkl", 'rb') as f:
        legal_aurocs=torch.load(f)
    with open(f"{directory}/contents_aurocs.pkl", 'rb') as f:
        contents_aurocs=torch.load(f)
    these_legal_aurocs, top_locations=torch.topk(legal_aurocs.flatten(), k=k)
    top_locations_tuples=[location_to_triple(x, legal_aurocs.shape) for x in top_locations]
    these_content_aurocs=[contents_aurocs[(a,b,c-1)] for a,b,c in top_locations_tuples]
    print("(Feature_number, board_position, class), Legal AUROC, Content AUROC, Difference")
    print("\n".join([f"{top_locations_tuple}, {legal_auroc:.4f}, {content_auroc:.4f}, {content_auroc-legal_auroc:.4f}" for top_locations_tuple, legal_auroc, content_auroc in zip(top_locations_tuples, these_legal_aurocs, these_content_aurocs)]))
    return top_locations_tuples, legal_aurocs


def location_to_triple(index, original_shape):
    c=int(index%original_shape[-1])
    b=int(index//original_shape[-1]%original_shape[-2])
    a=int(index//original_shape[-1]//original_shape[-2])
    return (a,b,c)

def show_top_activating(feature_number, top_k=5, marked_position=-1, directory="analysis_results"):
    with open(f"{directory}/all_activations.pkl", 'rb') as f:
        all_activations=torch.load(f)
    with open(f"{directory}/all_boards.pkl", 'rb') as f:
        all_boards=torch.load(f)
    num_data=all_boards.shape[0]
    filtered_boards=all_boards[torch.where(all_boards[:,0]>-100)]
    these_activations=all_activations[:, feature_number]
    filtered_activations=these_activations[torch.where(all_boards[:,0]>-100)]
    best_activations, best_indices=torch.topk(filtered_activations, k=top_k)
    random_indices=torch.randint(low=0, high=num_data, size=(top_k,))
    best_boards=filtered_boards[best_indices]
    # best_activations=sorted_activations[best_indices]
    random_boards=filtered_boards[random_indices]
    random_activations=filtered_activations[random_indices]

    fig, ax=plt.subplots(figsize=(10,5))
    plt.suptitle(f"Top Activating Boards and Random Boards for Feature {feature_number}, with Position {marked_position} marked")
    for n in range(top_k):
        ax_1=plt.subplot(2, top_k, n+1)
        plot_board(ax_1, best_boards[n], best_activations[n], marked_position=marked_position)
        ax_2=plt.subplot(2, top_k, n+1+top_k)
        plot_board(ax_2, random_boards[n], random_activations[n], marked_position=marked_position)

    plt.savefig(f"analysis_results/figures/top_act_boards_feature_{feature_number}.png")
    plt.close()


def plot_board(ax, board, activation, marked_position=-1):

    black_pieces_locations=[]
    white_pieces_locations=[]
    for n in range(64):
        x=n%8
        y=n//8
        if board[n]==1:
            black_pieces_locations.append((x,y))
        elif board[n]==2:
            white_pieces_locations.append((x,y))
    
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_aspect(1)
    ax.set_axisbelow(True)

    marker_size=50

    if black_pieces_locations:
        ax.scatter(x=torch.tensor(black_pieces_locations)[:,0],y=torch.tensor(black_pieces_locations)[:,1], c="black", s=marker_size, linewidths=1.5, edgecolors="black")
    if white_pieces_locations:
        ax.scatter(x=torch.tensor(white_pieces_locations)[:,0],y=torch.tensor(white_pieces_locations)[:,1], c="white", s=marker_size, linewidths=1.5, edgecolors="black")
    
    if marked_position>=0:
        x_center=marked_position%8
        y_center=marked_position//8
        ax.scatter(x=[x_center], y=[y_center], facecolors="none", edgecolors="red", s=marker_size*2)
    
    ax.set_title(f"Activation {activation:.2f}")
    ax.invert_yaxis()

    ax.set_yticks(range(8), labels=range(1, 8+1))
    ax.set_xticks(range(8), labels=["A", "B", "C", "D", "E", "F", "G", "H"])
    return

def save_activations_boards_and_legal_moves(sae_location="saes/sae_layer_6.pkl", eval_dataset_type="probe_test", offset=0, save_directory="analysis_results"):
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
    all_boards=torch.cat(boards).flatten(end_dim=-2) # shape (dw,b), where d= dataset size (2000), w=trimmed window length (52), b=board_size(64)
    all_legal_moves=torch.cat(legal_moves).flatten(end_dim=-2) # shape (dw,b), where d= dataset size (2000), w=trimmed window length (52), b=board_size(64)
    with open(f"{save_directory}/all_activations.pkl", 'wb') as f:
        torch.save(all_activations, f)
    with open(f"{save_directory}/all_boards.pkl", 'wb') as f:
        torch.save(all_boards, f)
    with open(f"{save_directory}/all_legal_moves.pkl", 'wb') as f:
        torch.save(all_legal_moves, f)

def combine(legal_moves, board):
    return (board+10*legal_moves).reshape((8,8))

def manual_test():
    '''
    old junk code
    '''
    with open("saes/sae_layer_6_trimmed_alpha_77_e-3.pkl", 'rb') as f:
        sae=torch.load(f, map_location=device)
    
    moves="D3 E3 F3 E2 F1 C2 F4 G3 B1 C6 H3 D2 C4 C3 F5 C5 E6 G4 B5 B2 A2 B6 G5 E1 C1 F6 F2 E7 B7 H4 E8 D1 G1 G2 F7 A1 H2 C7 B3 F8 D8 H1 G7 D7 H5 D6 C8 A4 A5 H8 G6 H6 A3 A7 A6 B4 A8 G8 B8"
    # moves="D3 E3 F3 E2 F2 C6 E6 F5 E1 F7 G6 G2 G4 H5 F6 D1 H6 H7 C1 F1 F8 C5 D6 D7 G5 F4 B4 B5 D8 B3 E7 B1 H3 C2 A5 A4 B6 C8 H1 H4 G1 G3 C4 A7 E8 B7 C7 D2 H2 G7 C3 A3 H8 A6 B2 G8 A1"
    dataloader=utils.dataloaders.OthelloDataset(None)
    input= torch.tensor(dataloader.string_to_int_list(moves)).unsqueeze(dim=0)
    (reconstruction,hidden_layer,reconstruction_loss, sparsity_loss, trimmed_logits), total_loss= sae(input, None)
    feature_number=111
    specific_hidden_layer=hidden_layer[0, 0, feature_number]
    print(specific_hidden_layer)
    return

def analyse_saes_across_seed_range():
    for idx in range(1, 11):
        autoencoder_location=f"saes/sae_layer_6_trimmed_seed_{idx}.pkl"
        dir=f"analysis_results/seed_{idx}/"
        os.makedirs(dir, exist_ok=True)
        save_activations_boards_and_legal_moves(sae_location=autoencoder_location, save_directory=dir, offset=1)
        evaluate_all_content_classification(directory=dir)
        

def compare_top_features_across_seeds():
    all_top_locations=[]
    for seed in range(1, 11):
        dir=f"analysis_results/seed_{seed}"
        print(f"Seed {seed}:")
        top_locations_tuples, top_values = find_top_aurocs_contents(k=50, directory=dir)
        all_top_locations.extend([position for (feature,position,piece_type),auroc in zip(top_locations_tuples, top_values) if auroc>=.9])
    counts=count_frequencies(all_top_locations)
    print("\n".join(f"{pos}: {freq}" for pos, freq in sorted(counts.items(), key=lambda x: x[1], reverse=True)))
    
    data_to_plot=torch.zeros((8,8))
    for position in range(64):
        if position in counts:
            data_to_plot[position//8][position%8]=counts[position]
    


    fig, ax = plt.subplots()
    im = ax.imshow(data_to_plot)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(8), labels=list("ABCDEFGH"))
    ax.set_yticks(range(8), labels=range(1, 9))

    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, int(data_to_plot[i, j]),
                        ha="center", va="center", color="w")

    ax.set_title("Across 10 starting seeds, number of times each feature is found")
    fig.tight_layout()
    plt.savefig("analysis_results/figures/feature_frequencies.png")
    plt.close()



def count_frequencies(iterable):
    to_return={}
    for x in iterable:
        if x in to_return:
            to_return[x]+=1
        else:
            to_return[x]=1
    return to_return

def graph_MCS_and_AUROC_correlation(autoencoder_location="saes/sae.pkl", rectify=False):
    with open("analysis_results/contents_aurocs.pkl", 'rb') as f:
        contents_aurocs=torch.load(f)
    probe_directions=get_probe_directions(rectify=rectify)
    autoencoder_directions=get_autoencoder_directions(autoencoder_location)
    cosine_similarities=probe_directions@autoencoder_directions.transpose(0,1)
    max_cosine_similarities_of_probes, locations=cosine_similarities.max(dim=1)
    corresponding_aurocs=torch.zeros(192)
    for probe_number, feature_number in enumerate(locations):
        corresponding_aurocs[probe_number]=contents_aurocs[feature_number, probe_number%64, probe_number//64]
        # corresponding_aurocs[probe_number]=contents_aurocs[feature_number, probe_number//3, probe_number%3]
    plt.scatter(x=max_cosine_similarities_of_probes, y=corresponding_aurocs)
    plt.ylabel("Feature AUROC")
    plt.xlabel("MCS of probe and feature")

    # cosine_similarities=(probe_directions@autoencoder_directions.transpose(0,1)).reshape((64, 3, 1024))
    # max_cosine_similarities_of_probes, locations=cosine_similarities.max(dim=-1)
    # aurocs=
    return


if __name__=="__main__":

    autoencoder_location="saes/sae_layer_6_trimmed.pkl"
    dir="analysis_results"
    # compare_probe_and_feature_aurocs()
    # evaluate_all_probe_classification()
    # evaluate_all_legal_moves_classification()


    # compare_top_features_across_seeds()

    # show_top_activating(468, marked_position=34)

    # analyze_cosine_similarities(autoencoder_location=autoencoder_location, rectify=True, include_control=True)
    # analyze_cosine_similarities_2(autoencoder_location=autoencoder_location)
    # find_top_aurocs_legal(k=20)
    # create_density_plot_legal(feature_number=722, board_position=26)
    # create_density_plot_legal(feature_number=395, board_position=43)
    # create_density_plot_legal(feature_number=1009, board_position=19)
    # create_density_plot_legal(feature_number=831, board_position=20)
    # create_density_plot_legal(feature_number=38, board_position=45)
    # show_top_activating(353, marked_position=18)
    # show_top_activating(525, marked_position=7)
    # show_top_activating(727, marked_position=37)
    # show_top_activating(722, marked_position=26)
    # show_top_activating(395, marked_position=43)
    # show_top_activating(612, marked_position=56)
    # show_top_activating(142, marked_position=43)


    # find_top_aurocs_contents(k=20, directory=dir)
    # create_density_plot_contents(feature_number=857, board_position=34)
    # create_density_plot_contents(feature_number=353, board_position=18)
    # create_density_plot_contents(feature_number=722, board_position=26)
    # create_density_plot_contents(feature_number=688, board_position=15)
    # create_density_plot_contents(feature_number=395, board_position=43)
    # create_density_plot_contents(feature_number=525, board_position=7)
    # create_density_plot_contents(feature_number=831, board_position=20)
    # create_density_plot_contents(feature_number=38, board_position=45)

    # create_hist_for_probe(board_position=26, primary_class=0)
    # create_triple_hist_for_probe(board_position=26)
    # create_hist_for_probe(board_position=3, primary_class=0)
    # create_hist_for_probe(board_position=29, primary_class=0)
    # create_hist_for_probe(board_position=11, primary_class=0)
    # create_hist_for_probe(board_position=12, primary_class=0)
    # create_hist_for_probe(board_position=13, primary_class=0)
    # analyse_saes_across_seed_range()

    # compare_legal_contents_aurocs()
    # graph_MCS_and_AUROC_correlation(autoencoder_location=autoencoder_location, rectify=True)
    # analyze_cosine_similarities_3(rectify=True)