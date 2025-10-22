import random
import numpy as np
import torch
import os
import psutil




def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def memory_info():
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    return {"rss": info.rss, "vms": info.vms}