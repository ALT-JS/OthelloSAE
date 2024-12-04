import othello_gpt
import autoencoder
import linear_probes
from utils.tokenizer import encode, decode
from utils.save_residual_streams import save_residual_stream_from_dataloader
import torch
from train import train_model
import cProfile


device='cuda' if torch.cuda.is_available() else 'cpu'

def test_small_training(save=True):
    num_layers=2
    d_model=32
    n_heads=8
    window_length=4
    num_epochs=1
    report_every_n_steps=100
    batch_size=64
    train_corpus="gpt_train_small"
    eval_corpus="gpt_test"
    model=othello_gpt.OthelloGPT(num_layers=num_layers, d_model=d_model, n_heads=n_heads, window_length=window_length)

    train_model(model, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    if save:
        to_save_location="trained_model_test.pkl"
        with open(to_save_location, 'wb') as f:
            torch.save(model, f)

def full_scale_training(save=False, seed=0):
    num_layers=12
    d_model=512
    n_heads=8
    window_length=64

    train_corpus="gpt_train"
    eval_corpus="gpt_test"
    batch_size=64
    report_every_n_steps=2000
    num_epochs=6
    model=othello_gpt.OthelloGPT(num_layers=num_layers, d_model=d_model, n_heads=n_heads, window_length=window_length)
    train_model(model, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size, fixed_seed=seed)
    
    to_save_location=f"trained_model_full_12_{seed}.pkl"
    if save:
        with open(to_save_location, 'wb') as f:
            torch.save(model, f)

def test_generation():
    model=othello_gpt.OthelloGPT(num_layers=1, d_model=8, n_heads=2)
    start_text="C4"
    x=decode(model.generate(torch.unsqueeze(encode(start_text),dim=0), max_new_tokens=10)[0])
    print(x)

def test_unpickle():
    # model=othello_gpt.OthelloGPT(8,512,8)

    with open("trained_model_full.pkl", 'rb') as f:
        model=torch.load(f)
    start_text="XX C4"
    model_input=torch.unsqueeze(encode(start_text),dim=0).to(device)
    # xb,yb=train.get_batch("train", block_size=model.window_length)
    x=decode(model.generate(model_input, max_new_tokens=10)[0])
    print(x)


def test_sae_training(target_layer, save=False):
    trained_model_location="trained_model_test.pkl"
    with open(trained_model_location, 'rb') as f:
        language_model=torch.load(f, map_location=device)
    num_epochs=1
    report_every_n_steps=5
    batch_size=8
    train_corpus="sae_train"
    eval_corpus="probe_test"
    feature_ratio=2
    sparsity_coeff=1e-3
    window_start=1
    window_end=1

    sparse_autoencoder=autoencoder.SparseAutoencoder(language_model, layer_num=target_layer, feature_ratio=feature_ratio, sparsity_coeff=sparsity_coeff, window_start_trim=window_start, window_end_trim=window_end)
    train_model(sparse_autoencoder, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)

    if save:
        to_save_location=f"saes/sae_layer_{target_layer}.pkl"
        with open(to_save_location, 'wb') as f:
            torch.save(sparse_autoencoder, f)

def full_sae_training(target_layer, save=False):
    trained_model_location="trained_model_full.pkl"
    with open(trained_model_location, 'rb') as f:
        language_model=torch.load(f, map_location=device)
    num_epochs=4
    report_every_n_steps=500
    batch_size=64
    train_corpus="sae_train"
    eval_corpus="probe_test"
    feature_ratio=2
    sparsity_coeff=7.7e-2
    window_start=4
    window_end=8
    normalize_inputs=False

    sparse_autoencoder=autoencoder.SparseAutoencoder(language_model, layer_num=target_layer, feature_ratio=feature_ratio, sparsity_coeff=sparsity_coeff, window_start_trim=window_start, window_end_trim=window_end, normalize_inputs=normalize_inputs)
    train_model(sparse_autoencoder, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    if save:
        to_save_location=f"saes/sae_layer_{target_layer}_trimmed.pkl"
        with open(to_save_location, 'wb') as f:
            torch.save(sparse_autoencoder, f)

def sae_hyperparameter_sweep(target_layer):
    trained_model_location="trained_model_full.pkl"
    with open(trained_model_location, 'rb') as f:
        language_model=torch.load(f, map_location=device)
    num_epochs=1
    report_every_n_steps=320
    batch_size=64
    train_corpus="sae_train"
    eval_corpus="probe_test"
    feature_ratio=2
    window_start=4
    window_end=8
    normalize_inputs=False

    sparsity_coeff_choices=torch.logspace(-2.0, -1.0, steps=10, base=10.0)

    for sparsity_coeff in sparsity_coeff_choices:
        print(f"Training autoencoder with sparsity coefficient {sparsity_coeff}\n")

        sparse_autoencoder=autoencoder.SparseAutoencoder(language_model, layer_num=target_layer, feature_ratio=feature_ratio, sparsity_coeff=sparsity_coeff, window_start_trim=window_start, window_end_trim=window_end, normalize_inputs=normalize_inputs)
        sparse_autoencoder.write_updates_to="hyperparameter_results.txt"
        with open(sparse_autoencoder.write_updates_to, 'a') as f:
            f.write(f"Training autoencoder with sparsity coefficient {sparsity_coeff}\n")
        train_model(sparse_autoencoder, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)

def test_diff_probes(othellogpt_layer, probe_layer, seed = 9999):
        
    trained_model_location=f"trained_model_full_{seed}.pkl"
    with open(trained_model_location, 'rb') as f:
        language_model=torch.load(f, map_location=device)
    
    probe_model_location = f"probes/probe_layer_{probe_layer}_{seed}_trimmed.pkl"
    with open(probe_model_location, 'rb') as f:
        probe_model=torch.load(f, map_location=device)
        
    num_epochs=1
    report_every_n_steps=10
    batch_size=16
    train_corpus="probe_train_small"
    eval_corpus="probe_test"
    window_start=1
    window_end=1
    
    linear_probe_model=linear_probes.LinearProbe(language_model, layer_num=othellogpt_layer, window_start_trim=window_start, window_end_trim=window_end).cuda()
    linear_probe_model.classifier[0].weight.data = probe_model[f'classifier.0.weight']
    linear_probe_model.classifier[1].weight.data = probe_model[f'classifier.1.weight']
    linear_probe_model.classifier[2].weight.data = probe_model[f'classifier.2.weight']
    
    train_model(linear_probe_model, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size, fixed_seed=False)
    

def test_linear_probes(target_layer, save=True):
    trained_model_location="trained_model_test.pkl"
    with open(trained_model_location, 'rb') as f:
        language_model=torch.load(f, map_location=device)
    num_epochs=1
    report_every_n_steps=100
    batch_size=16
    train_corpus="probe_train_small"
    eval_corpus="probe_test"
    window_start=1
    window_end=1
    linear_probe_model=linear_probes.LinearProbe(language_model, layer_num=target_layer, window_start_trim=window_start, window_end_trim=window_end)
    train_model(linear_probe_model, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    if save:
        to_save_location=f"probes/probe_layer_{target_layer}.pkl"
        with open(to_save_location, 'wb') as f:
            torch.save(linear_probe_model, f)

def full_probe_run(target_layer, save=True, trained_model_location="trained_model_full.pkl"):
    with open(trained_model_location, 'rb') as f:
        language_model=torch.load(f, map_location=device)
    
    id = trained_model_location.split('_')[-1][:-4]
    num_epochs=1
    report_every_n_steps=2000
    batch_size=64
    train_corpus="probe_train"
    eval_corpus="probe_test"
    window_start=4
    window_end=8
    linear_probe_model=linear_probes.LinearProbe(language_model, layer_num=target_layer, window_start_trim=window_start, window_end_trim=window_end)
    print(num_epochs, batch_size, report_every_n_steps)
    train_model(linear_probe_model, train_dataset_type=train_corpus, eval_dataset_type=eval_corpus, num_epochs=num_epochs, report_every_n_steps=report_every_n_steps, batch_size=batch_size)
    
    trimmed_probe = {}
    for mode in list(range(0, 3)):
        trimmed_probe[f'classifier.{mode}.weight'] = linear_probe_model.state_dict()[f'classifier.{mode}.weight'] # (64, 512)
    
    if save:
        to_save_location=f"probes/12layer/12_probe_layer_{target_layer}_{id}_trimmed.pkl"
        with open(to_save_location, 'wb') as f:
            torch.save(trimmed_probe, f)

if __name__ == '__main__':
    # test_small_training(save=True)
    
    # for gpt_layer in list(range(0, 8)):
    #     for probe_layer in ["6"]:
    #         print(f"Different probe test on GPT layer {gpt_layer} and Probe layer {probe_layer}:")
    #         test_diff_probes(gpt_layer, probe_layer, seed=9999)
    
    full_scale_training(save=True, seed=9999)
    # full_scale_training(save=True, seed=114514)
    # full_scale_training(save=True, seed=20040805)
    
    # test_unpickle()

    # test_sae_training(target_layer=6)
    # sae_hyperparameter_sweep(6)
    # full_probe_run(target_layer=6)
    # full_sae_training(target_layer=6, save=True)

    # test_linear_probes(6)
    # for n in range(1, 9):
    #     full_probe_run(target_layer=n)

