import sys
import torch
print(sys.version)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())