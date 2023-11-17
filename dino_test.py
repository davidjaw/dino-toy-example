import torch

# DINOv2
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

dummy_img = torch.zeros(1, 3, 224, 224)
pred = dinov2_vits14(dummy_img)  # returns logits

print()

