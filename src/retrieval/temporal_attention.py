import torch
import torch.nn as nn
import numpy as np


class TemporalAttention(nn.Module):
    """
    Applies multi-head self-attention over a sequence of CLIP frame embeddings.
    Each frame embedding attends to all other frames — giving every vector
    temporal context from its neighbors before storing in FAISS.
    
    Input:  [N x D] numpy array  (N frames, D=512 CLIP dims)
    Output: [N x D] numpy array  (same shape, temporally attended)
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        self.register_buffer(
            "pos_enc",
            self._build_pos_enc(max_len=2000, dim=embed_dim)
        )

    def _build_pos_enc(self, max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N x D] tensor of frame embeddings
        returns: [N x D] attended embeddings
        """
        x = x.unsqueeze(0)                          # [1 x N x D] batch dim
        x = x + self.pos_enc[:x.shape[1]].unsqueeze(0)  # add positional encoding
        attended, _ = self.attn(x, x, x)            # self-attention
        return attended.squeeze(0)                   # back to [N x D]

    @torch.no_grad()
    def attend(self, vectors: np.ndarray) -> np.ndarray:
        """
        Convenience method — numpy in, numpy out.
        Handles single-frame case gracefully (no-op).
        """
        if vectors.shape[0] == 1:
            return vectors  

        t = torch.from_numpy(vectors).float()
        attended = self.forward(t)
        attended = attended / attended.norm(dim=-1, keepdim=True)
        return attended.numpy().astype("float32")