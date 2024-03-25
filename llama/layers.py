# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

import math
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
  """RMSNorm module."""

  def __init__(self, dim: int, eps: float = 1e-6, device='meta'):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, device=device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output = self._norm(x.float()).type_as(x)
    return output * self.weight




def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  ndim = x.ndim
  assert 1 < ndim
  assert freqs_cis.shape == (x.shape[-3], x.shape[-1]), x.shape
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # bs, seqlen, heads, dim
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)."""

  bs, n_kv_heads, slen, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:, :, None, :, :]
      .expand(bs, n_kv_heads, n_rep, slen, head_dim)
      .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
  )


class Attention(nn.Module):
  """Attention module."""

  def __init__(self, args):
    super().__init__()

    self.n_kv_heads = (
        args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    )
    self.n_local_heads = args.n_heads
    self.n_local_kv_heads = self.n_kv_heads
    self.n_rep = self.n_local_heads // self.n_local_kv_heads
    self.head_dim = args.dim // args.n_heads
    self.max_seq_len = args.max_seq_len
    self.n_heads = args.n_heads

    LinearLayer = nn.Linear


    self.wo = LinearLayer(
        args.n_heads * self.head_dim,
        args.dim,
        bias=False,
        device=args.device,
    )
    self.q_size = args.n_heads * self.head_dim
    self.kv_size = self.n_kv_heads * self.head_dim

    self.wq = LinearLayer(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wk = LinearLayer(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wv = LinearLayer(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
        device=args.device,
    )  
    
  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache,
  ):
    # bsz, seqlen, _ = x.shape
    
    bsz, seqlen = x.shape[0], x.shape[-2]

    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    

    
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    if seqlen == 1:
        xq = torch.broadcast_to(xq, (xq.shape[0], 2, xq.shape[2], xq.shape[3])) 

    #if not self.env.enable_kv_quantization:
    keys, values = cache.update(xk, xv)

    ## Attention start
    scores = torch.einsum("ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
    #scores = torch_xla2.extra.call_jax(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)

    if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)

    scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    output = torch.einsum(
        "ikjm,ikml->ikjl", scores, values
    )  # (bs, n_local_heads, seqlen, head_dim)
    #output = torch_xla2.extra.call_jax(jnp.einsum,"ikjm,ikml->ikjl", scores, values)
    # For XLA matmul performance boost
    if seqlen == 1:
        output = output[:, :, 0, :]
    #output = torch.matmul(scores, values)
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

    
