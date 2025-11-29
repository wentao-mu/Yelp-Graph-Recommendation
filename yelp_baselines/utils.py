import numpy as np, torch, random
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_user_pos_items(interactions):
    # interactions: list of (u,i)
    pos = defaultdict(set)
    for u,i in interactions:
        pos[u].add(i)
    return pos

def sample_neg(user_pos_items, n_items, u, num=1):
    res = []
    for _ in range(num):
        while True:
            j = np.random.randint(0, n_items)
            if j not in user_pos_items[u]:
                res.append(j); break
    return res

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    print(f"[device] using {dev}") 
    return dev
