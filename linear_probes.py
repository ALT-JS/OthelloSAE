import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.dataloaders import get_dataloader

class LinearProbe(torch.nn.Module):

    def __init__(self, othello_gpt_model, layer_num, window_start_trim=0, window_end_trim=0):
        super().__init__()
        self.othello_gpt_model=othello_gpt_model
        self.layer_num=layer_num
        self.board_size=64
        self.classifier=nn.ModuleList(nn.Linear(self.othello_gpt_model.d_model, self.board_size, bias=False) for _ in range(3))
        self.window_length=self.othello_gpt_model.window_length
        self.layer_norm=nn.LayerNorm(normalized_shape=(self.window_length-window_start_trim-window_end_trim, self.othello_gpt_model.d_model))
        self.window_start_trim=window_start_trim
        self.window_end_trim=window_end_trim
        # freeze othello gpt model weights
        for parameter in self.othello_gpt_model.parameters():
            parameter.requires_grad=False

    def forward(self, input, target=None):
        #input is B-by-W, where W is the length of the context window
        #target are B-W-64, and are 0,1,2 for the 3 classes, with -100 for indices that should be ignored (e.g after the game)
        logits=self.othello_gpt_model.intermediate_residual_stream(input, self.layer_num) #B-W-d_model
        logits=self.trim_to_window(logits) #B-w-d_model, where w is the length of the shrunk window
        normalized_logits=self.layer_norm(logits)
        predictions=torch.stack([classifier_shard(normalized_logits) for classifier_shard in self.classifier], dim=3) #B-w-64-3
        if target is None:
            loss=None
        else:
            target=self.trim_to_window(target)
            loss_function=torch.nn.CrossEntropyLoss()
            loss=loss_function(input=predictions.flatten(end_dim=-2), target=target.flatten())
        return predictions, loss

    def print_evaluation(self, train_loss, eval_dataset_type, step_number="N/A", details=False):
        del details
        accuracy=self.evaluate_top_one_board_state_accuracy(eval_dataset_type)
        print(f"Train loss and test accuracy after {step_number} steps: {train_loss.item():.4f}, {accuracy:.4%}")

    def evaluate_top_one_board_state_accuracy(self, eval_dataset_type="probe_test"):
        batch_size=8

        total_predictions=0
        total_correct_predictions=0
        window_length=self.window_length
        test_probe_dataloader=iter(get_dataloader(mode=eval_dataset_type, window_length=window_length, batch_size=batch_size))

        for input_batch,label_batch in test_probe_dataloader:
            label_batch=self.trim_to_window(label_batch)
            predictions, loss=self(input_batch)
            one_hot_predictions=predictions.argmax(dim=3)
            total_correct_predictions+=(label_batch==one_hot_predictions).sum()
            total_predictions+=(label_batch>=0).sum()

        return float(total_correct_predictions/total_predictions)
    
    def trim_to_window(self, x, offset=0):
        return x[:, (self.window_start_trim+offset):(self.window_length-self.window_end_trim+offset), :]
    
    def save_state_on_dataset(self, offset=0, eval_dataset_type="probe_test", activations_save_location="analysis_results/probe_activations.pkl", boards_save_location="analysis_results/probe_all_boards.pkl"):
        batch_size=8
        window_length=self.window_length
        test_probe_dataloader=iter(get_dataloader(mode=eval_dataset_type, window_length=window_length, batch_size=batch_size))
        activations=[]
        boards=[]
        for input_batch,label_batch in test_probe_dataloader:
            pred, loss=self(input_batch, None)
            activations.append(pred)
            boards.append(self.trim_to_window(label_batch, offset=offset))
        activations=torch.cat(activations).flatten(0,1)
        boards=torch.cat(boards).flatten(0,1)
        with open(activations_save_location, 'wb') as f:
            torch.save(activations, f)
        with open(boards_save_location, 'wb') as f:
            torch.save(boards, f)
    
