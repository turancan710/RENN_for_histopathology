import pickle
import torch
import numpy as np
import random

def save_data(X, y, groups , beta , b , e, sigma2_b, sigma2_e, n, p, q, seed, path):

    data_dict = {
    "X": X ,
    "y": y ,
    "groups": groups ,
    "beta": beta,
    "b": b,
    "e": e,
    "config": {
        "sigma2_b": sigma2_b,
        "sigma2_e": sigma2_e,
        "n": n,
        "p": p,
        "q": q,
        "sim_seed": seed
        }
        }

    with open(path, "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"saved simulated data to {path}")

def global_seed(seed):
    random.seed(seed) #python cpu
    np.random.seed(seed) #np cpu
    torch.manual_seed(seed) #torch cpu
    torch.cuda.manual_seed_all(seed) # CUDA gpu
    torch.backends.cudnn.deterministic = True #deterministic torch methods
    torch.backends.cudnn.benchmark = False #no switch between CUDA NN algorithms 

#dataloader utilizes multiple workers -> assign a seed to them:
def worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)