"""This version contains modification to make it easier to trace and support batch."""

import math
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from . import model_args 
from llama.layers import Attention, RMSNorm



class FeedForward(nn.Module):
  """Feed-forward module."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
      device = 'meta',
      quantize = False,
      env = None,
  ):
    super().__init__()
    self.env = env
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    LinearLayer = nn.Linear

    self.w1 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
    )
    self.w2 = LinearLayer(
        hidden_dim,
        dim,
        bias=False,
        device=device,
    )
    self.w3 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
    )

  def forward(self, x):
    result = self.w2(F.silu(self.w1(x)) * self.w3(x))
    return result


class TransformerBlock(nn.Module):
  """Transformer block."""

  def __init__(
      self,
      layer_id: int,
      args: model_args.ModelArgs
  ):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads

    self.attention = Attention(
        args
    )
    self.feed_forward = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        device=args.device,
        quantize=args.quantize,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache
  ):

    attn = self.attention.forward(
        self.attention_norm(x),
        freqs_cis,
        mask,
        cache
    )

    h = x + attn
    ffns = self.ffn_norm(h)

    out = h + self.feed_forward.forward(ffns)
    return out

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis

class Transformer(nn.Module):
  """Transformer module."""

  def __init__(
      self,
      params: model_args.ModelArgs,
  ):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    Embedding = nn.Embedding
    self.tok_embeddings = Embedding(
        params.vocab_size,
        params.dim,
        device=params.device,
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(
          TransformerBlock(
              layer_id,
              params
          )
      )
    self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=params.device)
    
    LinearLayer = nn.Linear

    self.output = LinearLayer(
        params.dim,
        params.vocab_size,
        bias=False,
        device=params.device,
    )
    # TODO what to do with this
    freqs_cis = precompute_freqs_cis(
        self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )
    # added
    #self.freqs_cis = freqs_cis 

    # changed
    self.register_buffer("freqs_cis", freqs_cis)


  @torch.no_grad()
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      caches: List[Any],
      mask,
  ):

    seqlen = tokens.shape[-1]
    h = self.tok_embeddings(tokens)

    freqs_cis = self.freqs_cis[input_pos]
    #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

    for layer in self.layers:
        h = layer(
            h,
            freqs_cis,
            mask,
            caches
        )

    h = self.norm(h)
    output = self.output(h).float()
    return output
