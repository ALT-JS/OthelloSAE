import torch
from utils.dataloaders import get_dataloader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

device='cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train_dataset_type, eval_dataset_type, batch_size=64, num_epochs=2, report_every_n_steps=500, fixed_seed=1337):
    '''
    model be a nn.Module object, and have a print_evaluation() method
    train_dataset_type and eval_dataset_type must be in the list of valid types defined in the recognized_dataset() method in utils/dataloaders 
    '''
    if fixed_seed:
        torch.manual_seed(fixed_seed)
        model.to(device)
        model.train()
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        step=0
        print(f"Beginning model training on {device}!")

        for epoch in range(num_epochs):
            train_dataloader=iter(get_dataloader(mode=train_dataset_type, window_length=model.window_length, batch_size=batch_size))
            print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps, will evaluate every {report_every_n_steps} steps.")
            
            for input_batch,label_batch in tqdm(train_dataloader):
                step+=1
                optimizer.zero_grad(set_to_none=True)
                output, loss=model(input_batch, label_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if step % report_every_n_steps==0:
                    model.print_evaluation(loss, eval_dataset_type, step_number=step)
    else:
        loss = torch.zeros(1)
        model.print_evaluation(train_loss=loss, eval_dataset_type=eval_dataset_type, step_number="Omega", details=True)