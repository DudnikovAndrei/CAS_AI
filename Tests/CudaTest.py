
import torch
print(torch.backends.cuda.is_built())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.version())
print('gpu', torch.cuda.is_available())
if torch.backends.cudnn.enabled:
    print("enabled")
else:
    print("not enabled")

if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu only")
