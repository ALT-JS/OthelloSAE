import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.game_engine import tokens_list
from utils.dataloaders import get_dataloader, get_othello_labels_and_legal_moves

device='cuda' if torch.cuda.is_available() else 'cpu'

class OthelloGPT(nn.Module):

    def __init__(self, num_layers, d_model, n_heads, window_length=64, dropout_chance=.2, tied_embed=False):
        super().__init__()
        self.num_layers=num_layers
        self.d_model=d_model
        self.n_heads=n_heads
        self.window_length=window_length
        self.vocab_size=len(tokens_list())
        self.d_head=int(d_model/n_heads)
        self.token_embed_table=nn.Embedding(self.vocab_size, d_model)
        self.position_embed_table=nn.Embedding(window_length, d_model)
        self.unembed=nn.Linear(d_model, self.vocab_size)
        self.dropout_chance=dropout_chance

        self.linear_activation=nn.GELU()
        self.blocks=nn.Sequential(
            *(MyTransformerBlock(d_model=d_model, 
                                 n_heads=n_heads, 
                                 linear_activation=self.linear_activation,
                                 dropout_chance=self.dropout_chance) 
                                    for _ in range(self.num_layers)))
        self.final_layer_norm=LayerNorm(dim=self.d_model)


    def forward(self, input, targets=None):
        '''
        '''
        if input.shape[1]>self.window_length:
            input=input[:,:self.window_length]
        if targets != None and targets.shape[1]>self.window_length:
            targets=targets[:,:self.window_length]
        model_device=next(self.parameters()).device
        input=input.to(model_device)
        positions=torch.arange(input.shape[1]).to(model_device)
        logits=self.token_embed_table(input)+self.position_embed_table(positions)
        logits=self.blocks(logits)
        logits=self.final_layer_norm(logits)
        logits=self.unembed(logits)
        if targets is None:
            loss=None
        else:
            loss=F.cross_entropy(torch.transpose(logits, dim0=1, dim1=2), targets)
        return logits,loss
    
    def generate(self, input, max_new_tokens):
        model_device=next(self.parameters()).device
        input.to(model_device)
        for _ in range(max_new_tokens):
            logits, loss=self(input)
            logits=logits[:,-1,:]
            probs=F.softmax(logits, dim=-1)
            idx_next=torch.multinomial(probs, num_samples=1)
            input=torch.concatenate((input, idx_next), dim=1)
        return input
    
    def intermediate_residual_stream(self, input, layer_num, require_grad=False):
        # returns the state of the residual stream after applying the first layer_num transformer blocks
        # so layer_num=0 is the initial stream, 1 is after the the first block, etc
        if input.shape[1]>self.window_length:
            input=input[:,:self.window_length]
        positions=torch.arange(input.shape[1])
        if input.get_device()>=0:
            positions=positions.to(input.get_device())
        logits=self.token_embed_table(input)+self.position_embed_table(positions) #B-W-d_model
        logits=self.blocks[:layer_num](logits)
        logits.requires_grad=require_grad
        return logits
    
    def print_evaluation(self, train_loss, eval_dataset_type, step_number="N/A", details=False):
        '''
        runs several evaluation methods on self, then prints the results and the train_loss
        '''
        self.eval()
        test_loss=self.evaluate_test_loss(eval_dataset_type=eval_dataset_type)
        divergence=self.evaluate_kl_divergence(eval_dataset_type=eval_dataset_type)
        accuracy=self.evaluate_top_one_accuracy(eval_dataset_type=eval_dataset_type)
        if details:
            accuracy_by_turn=self.evaluate_top_one_accuracy_by_turn(eval_dataset_type=eval_dataset_type)
            print(f"After training step {step_number}, accuracy on turns 1, 2, 3, ...: {accuracy_by_turn}")
        print(f"Train loss, test loss, divergence, accuracy after {step_number} steps: {train_loss.item():.4f}, {test_loss.item():.4f}, {divergence:.4f}, {accuracy:.4f}")
        self.train()

    def evaluate_test_loss(self, eval_dataset_type="gpt_test"):
        test_dataloader=iter(get_dataloader(eval_dataset_type, window_length=self.window_length, batch_size=1))
        test_labels, test_input= next(test_dataloader)
        logits, loss=self(test_labels, test_input)
        return loss

    def evaluate_kl_divergence(self, eval_dataset_type="gpt_test",  num_samples=80):
        batch_size=1
        batches=num_samples//batch_size
        divergences=[]
        for n in range(batches):
            xb, legal_moves=get_othello_labels_and_legal_moves(window_length=self.window_length, num_samples=batch_size, eval_dataset_type=eval_dataset_type,  key=n)
            legal_move_distribution=legal_moves/legal_moves.sum(dim=-1, keepdim=True)
            logits, loss=self(xb, None)
            kl_loss=torch.nn.KLDivLoss(reduction='batchmean')
            log_softmax=torch.nn.LogSoftmax(dim=-1)
            divergences.append(kl_loss(log_softmax(logits), legal_move_distribution))
        return float(torch.tensor(divergences).mean())

    def evaluate_top_one_accuracy(self, eval_dataset_type="gpt_test", num_samples=80):
        batch_size=1
        batches=num_samples//batch_size
        accuracies=[]
        for n in range(batches):
            xb, legal_moves=get_othello_labels_and_legal_moves(window_length=self.window_length, num_samples=batch_size, eval_dataset_type=eval_dataset_type,  key=n)
            logits, loss=self(xb, None)
            largest_entry_locations=logits.argmax(dim=-1, keepdim=True)
            one_hot_predictions = torch.zeros(legal_moves.shape).to(device)
            one_hot_predictions = one_hot_predictions.scatter(dim=-1, index=largest_entry_locations, src=torch.ones(legal_moves.shape, device=device))
            accuracies.append((legal_moves*one_hot_predictions).sum()/one_hot_predictions.sum())
        return float(torch.tensor(accuracies).mean())

    def evaluate_top_one_accuracy_by_turn(self, eval_dataset_type="gpt_test", num_samples=80):
        batch_size=1
        batches=num_samples//batch_size
        accuracies=[]
        for n in range(batches):
            xb, legal_moves=get_othello_labels_and_legal_moves(window_length=self.window_length, num_samples=batch_size, eval_dataset_type=eval_dataset_type,  key=n)
            logits, loss=self(xb, None)
            largest_entry_locations=logits.argmax(dim=-1, keepdim=True)
            one_hot_predictions = torch.zeros(legal_moves.shape).to(device)
            one_hot_predictions = one_hot_predictions.scatter(dim=-1, index=largest_entry_locations, src=torch.ones(legal_moves.shape, device=device))
            accuracies.append((legal_moves*one_hot_predictions).sum(dim=(0,2))/one_hot_predictions.sum(dim=(0,2)))
        return torch.stack(accuracies).mean(dim=0)

    
class MyAttentionHead(torch.nn.Module):

    def __init__(self, d_head, d_model, dropout_chance=.2, use_mask=True):
        super().__init__()
        self.d_model=d_model
        self.d_head=d_head
        self.scaling_factor=torch.nn.Parameter(1/torch.sqrt(torch.tensor([d_model])), requires_grad=False)
        self.Q = torch.nn.Linear(d_model, d_head, bias=False) 
        self.K = torch.nn.Linear(d_model, d_head, bias=False)
        self.V = torch.nn.Linear(d_model, d_head, bias=False)
        self.dropout_chance=.2
        self.dropout=nn.Dropout(p=dropout_chance)
        self.use_mask=use_mask


    def forward(self, residual_stream):
        attention=self.attention_pattern(residual_stream)
        attention=self.dropout(attention)
        x=self.V(residual_stream)
        return attention@x

    def attention_pattern(self, residual_stream):
        keys=self.K(residual_stream)
        queries=self.Q(residual_stream)
        pre_attention=queries@torch.transpose(keys, dim0=1, dim1=2)*self.scaling_factor
        if self.use_mask:
            upper_triangular=torch.tril(torch.ones(pre_attention.shape)).to(residual_stream.device)
            pre_attention=pre_attention.masked_fill(upper_triangular==0, float('-inf'))
        attention=F.softmax(pre_attention, dim=-1)
        return attention
        

class MyMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_chance=.2, use_mask=True):
        super().__init__()
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_head=d_model//n_heads
        self.dropout_chance=dropout_chance
        self.heads=nn.ModuleList([MyAttentionHead(d_head=self.d_head, d_model=d_model, dropout_chance=dropout_chance, use_mask=use_mask) for _  in range(n_heads)])
        self.proj=nn.Linear(self.n_heads*self.d_head, d_model)
        self.dropout=nn.Dropout(p=dropout_chance)

    def forward(self, residual_stream):
        all_heads_output=torch.cat([head(residual_stream) for head in self.heads], dim=-1)
        output=self.proj(all_heads_output)
        output=self.dropout(output)
        return output

class myMLPLayer(torch.nn.Module):

    def __init__(self, d_model, activation, dropout_chance=.2):
        super().__init__()
        self.encode=nn.Linear(in_features=d_model, out_features=4*d_model, bias=True)
        self.proj=nn.Linear(in_features=4*d_model, out_features=d_model, bias=True)
        self.activation=activation
        self.dropout_chance=dropout_chance
        self.dropout=torch.nn.Dropout(dropout_chance)

    def forward(self, residual_stream):
        hidden_layer=self.activation(self.encode(residual_stream))
        output=self.proj(hidden_layer)
        output=self.dropout(output)
        return output


class MyTransformerBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads, linear_activation, dropout_chance, use_mask=True):
        super().__init__()
        self.dropout_chance=dropout_chance
        self.attention_sublayer=MyMultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_chance=self.dropout_chance, use_mask=use_mask)
        self.mlp_sublayer=myMLPLayer(d_model=d_model, activation=linear_activation, dropout_chance=self.dropout_chance, )
        self.layernorm_1=LayerNorm(dim=d_model)
        self.layernorm_2=LayerNorm(dim=d_model)


    def forward(self, residual_stream):
        residual_stream=residual_stream+self.attention_sublayer(self.layernorm_1(residual_stream))
        residual_stream=residual_stream+self.mlp_sublayer(self.layernorm_2(residual_stream))
        return residual_stream
    
class LayerNorm(torch.nn.Module):

    def __init__(self, dim, eps=1e-10):
        super().__init__()
        self.eps=nn.Parameter(torch.tensor([eps],requires_grad=False))
        self.beta=nn.Parameter(torch.zeros(dim))
        self.gamma=nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        std, mean=torch.std_mean(x, dim=-1, keepdim=True)
        return ((x-mean)/(std+self.eps))*self.gamma+self.beta


