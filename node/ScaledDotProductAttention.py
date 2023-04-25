import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k:int) -> None:
        super().__init__()
        self.d_k = d_k
        
    def forward(
        self,
        q: torch.Tensor, # = Q
        k: torch.Tensor, # = X
        v: torch.Tensor, # = X
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scalar = np.sqrt(self.d_k)
        attention_weight = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar # Q * X^T / (D^0.5)
        
        if mask is not None: 
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            ) 

        attention_weight = nn.functional.softmax(attention_weight, dim=2) # Attention weight
        return torch.matmul(attention_weight, v) # (Attention weight) * X

