import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision import transforms

class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "google/vit-base-patch16-224-in21k",
        dropout_rate: float = 0.0,
    ):
        """
        Wrap the pure ImageNet-21k ViT and attach a new classification head.
        """
        super().__init__()

        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            # torch_dtype=torch.float16,
        )
        # enable gradient checkpointing
        self.vit.vit.gradient_checkpointing_enable()

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B, 3, H, W)
        Returns:
            logits: torch.Tensor of shape (B, num_classes)
        """
        outputs = self.vit(pixel_values=x)
        return outputs.logits

    def clip_gradients(self, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)