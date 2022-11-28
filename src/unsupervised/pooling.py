import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Pools the input sequence to a fixed number of vectors
    using attention pooling with `n_pools` learned queries.
    """
    def __init__(self, d_model: int, n_pools: int):
        super().__init__()
        self.d_model = d_model
        self.n_pools = n_pools
        self.register_parameter("query", nn.Parameter(torch.randn(n_pools, d_model)))

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len]
        Returns:
            pooled: [batch_size, n_pools, d_model]
        """
        # x: [batch, seq_len, d_model]
        # mask: [batch, seq_len]
        query = self.query.unsqueeze(0).expand(x.size(0), -1, -1)
        # query: [batch, n_pools, d_model]
        attn = torch.bmm(query, x.transpose(1, 2))
        # attn: [batch, n_pools, seq_len]
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        # attn: [batch, n_pools, seq_len]
        pooled = torch.bmm(attn, x)
        # pooled: [batch, n_pools, d_model]
        return pooled


class MaxPooling(nn.Module):
    def __init__(self, d_model: int, n_pools: int):
        """
        Slices the input sequence into `n_pools` and
        projects each slice to the same dimension
        before taking the max across the sequence length.
        If `n_pools` is 1, this behaves like regular max-pooling.
        """
        super().__init__()
        if d_model % n_pools != 0:
            raise ValueError("d_model must be divisible by n_pools")

        self.d_model = d_model
        self.n_pools = n_pools
        if self.n_pools > 1:
            self.proj = nn.Linear(d_model // n_pools, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len]
        Returns:
            pooled: [batch, n_pools, d_model]
        """
        # x: [batch, seq_len, d_model]
        # mask: [batch, seq_len]
        if self.n_pools > 1:
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            # x: [batch, seq_len, d_model]
            x = x.view(x.size(0), x.size(1), self.n_pools, -1)
            # x: [batch, n_pools, seq_len, n_pools, d_model // n_pools]
            x, _ = x.max(dim=2)
            # x: [batch, n_pools, d_model // n_pools]
            x = self.proj(x)
            # x: [batch, n_pools, d_model]
            return x
        else:
            return x.max(dim=1, keepdim=True)[0]

class MeanPooling(nn.Module):
    def __init__(self, d_model: int, n_pools: int):
        """
        Slices the input sequence into `n_pools` and
        projects each slice to the same dimension
        before taking the mean across the sequence length.
        If `n_pools` is 1, this behaves like regular mean-pooling.
        """
        super().__init__()
        if d_model % n_pools != 0:
            raise ValueError("d_model must be divisible by n_pools")

        self.d_model = d_model
        self.n_pools = n_pools
        if self.n_pools > 1:
            self.proj = nn.Linear(d_model // n_pools, d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len]
        Returns:
            pooled: [batch, n_pools, d_model]
        """
        # x: [batch, seq_len, d_model]
        # mask: [batch, seq_len]
        if self.n_pools > 1:
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, 0)
            # x: [batch, seq_len, d_model]
            x = x.view(x.size(0), x.size(1), self.n_pools, -1)
            # x: [batch, n_pools, seq_len, n_pools, d_model // n_pools]
            if mask is not None:
                x = x.sum(dim=2)
                x = x / mask.unsqueeze(-1).sum(dim=1, keepdim=True)
            else:
                x = x.mean(dim=2)
            # x: [batch, n_pools, d_model // n_pools]
            x = self.proj(x)
            # x: [batch, n_pools, d_model]
            return x
        else:
            return x.mean(dim=1, keepdim=True)
