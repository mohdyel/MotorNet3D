# save_vit_checkpoint.py
import torch
from transformers import ViTModel

# 1. Load the pretrained ViT (base) model from Hugging Face
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#    └─ this automatically downloads weights into ~/.cache/huggingface/...

# 2. Save only the state_dict as vit_base_2d.pth
torch.save(model.state_dict(), "vit_base_2d.pth")
print("Saved vit_base_2d.pth in the current directory.")
