import torch
from utils.dataloaders import get_dataloader
from tqdm import tqdm

# depricated because this method is too memory-intensive to use 

def save_residual_stream_from_dataloader(dataloader_mode, othello_gpt_model,layer_num, destination_file="save_residual_streams.pkl"):
    for param in othello_gpt_model.parameters():
        param.requires_grad = False
    dataloader=iter(get_dataloader(dataloader_mode, window_length=othello_gpt_model.window_length, batch_size=8))
    all_residual_streams=[]
    all_labels=[]
    for input, label in tqdm(dataloader):
        logits=othello_gpt_model.intermediate_residual_stream(input, layer_num) #B-W-d_model
        all_labels.append(label)
        all_residual_streams.append(logits)
    all_residual_streams=torch.cat(all_residual_streams, dim=0)
    all_labels=torch.cat(all_labels, dim=0)
    with open(destination_file, 'wb') as f:
        torch.save((all_residual_streams, all_labels), f)