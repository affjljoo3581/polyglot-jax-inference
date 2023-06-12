from __future__ import annotations

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class Attention(nn.Module):
    dim: int
    heads: int
    rotary: int

    def setup(self):
        head_dim = self.dim // self.heads
        self.wq = nn.DenseGeneral((self.heads, head_dim), dtype=jnp.bfloat16)
        self.wk = nn.DenseGeneral((self.heads, head_dim), dtype=jnp.bfloat16)
        self.wv = nn.DenseGeneral((self.heads, head_dim), dtype=jnp.bfloat16)
        self.wo = nn.DenseGeneral(self.dim, axis=(-2, -1), dtype=jnp.bfloat16)

        x = np.arange(0, self.rotary, 2) / self.rotary
        x = np.outer(np.arange(10000), 1.0 / 10000.0**x)
        self.freqs_cis = jnp.asarray(np.cos(x) + 1j * np.sin(x), dtype=jnp.complex64)

    def update_cache(self, name: str, x: chex.Array) -> chex.Array:
        if (cache := self.get_variable("cache", name)) is not None:
            x = jnp.roll(cache, -x.shape[1], axis=1).at[:, -x.shape[1] :].set(x)
        self.put_variable("cache", name, x)
        return x

    def apply_rotary_embedding(self, x: chex.Array) -> chex.Array:
        z = x.astype(jnp.float32).reshape(*x.shape[:-1], 2, -1)
        z = jax.lax.complex(z[..., 0, :], z[..., 1, :])

        z = z * self.freqs_cis[None, -x.shape[1] :, None, :]
        z = jnp.stack((jnp.real(z), jnp.imag(z)), axis=-1)
        return z.reshape(x.shape).astype(x.dtype)

    def __call__(self, x: chex.Array, attn_bias: chex.Array) -> chex.Array:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        k = self.update_cache("k", k)
        v = self.update_cache("v", v)

        q_rot = self.apply_rotary_embedding(q[..., : self.rotary])
        k_rot = self.apply_rotary_embedding(k[..., : self.rotary])

        q = jnp.concatenate((q_rot, q[..., self.rotary :]), axis=3)
        k = jnp.concatenate((k_rot, k[..., self.rotary :]), axis=3)

        p = jnp.einsum("bqhd,bkhd->bhqk", q, k) / k.shape[3] ** 0.5
        x = jnp.einsum("bhqk,bkhd->bqhd", nn.softmax(p + attn_bias, axis=3), v)
        return self.wo(x)


class FeedForward(nn.Module):
    dim: int
    hidden: int

    def setup(self):
        self.w1 = nn.Dense(self.hidden, dtype=jnp.bfloat16)
        self.w2 = nn.Dense(self.dim, dtype=jnp.bfloat16)

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.w2(nn.gelu(self.w1(x), approximate=False))


class TransformerLayer(nn.Module):
    dim: int
    heads: int
    hidden: int
    rotary: int
    eps: float = 1e-5

    def setup(self):
        self.attn = Attention(self.dim, self.heads, self.rotary)
        self.ff = FeedForward(self.dim, self.hidden)

        self.attn_norm = nn.LayerNorm(self.eps, dtype=jnp.bfloat16)
        self.ff_norm = nn.LayerNorm(self.eps, dtype=jnp.bfloat16)

    def __call__(self, x: chex.Array, attn_bias: chex.Array) -> chex.Array:
        return x + self.attn(self.attn_norm(x), attn_bias) + self.ff(self.ff_norm(x))


class Transformer(nn.Module):
    vocab_size: int
    layers: int
    dim: int
    heads: int
    rotary: int
    hidden: int
    eps: float = 1e-5

    def setup(self):
        layer_args = (self.dim, self.heads, self.hidden, self.rotary, self.eps)

        self.wte = nn.Embed(self.vocab_size, self.dim, dtype=jnp.bfloat16)
        self.layer = [TransformerLayer(*layer_args) for _ in range(self.layers)]
        self.head_norm = nn.LayerNorm(self.eps, dtype=jnp.bfloat16)
        self.head = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.bfloat16)

    def __call__(self, x: chex.Array, mask: chex.Array | None = None) -> chex.Array:
        if mask is None:
            mask = self.get_variable("cache", "mask")
            mask = jnp.roll(mask, -x.shape[1], axis=1).at[:, -x.shape[1] :].set(True)
        self.put_variable("cache", "mask", mask)

        # Create an attention bias to mask the attention probability which should be
        # ignored. To mask the future tokens, `jnp.tril` is used to the extended
        # attention bias array. We use `-1e9` which is relatively high penalty to make
        # the exponential value to zero.
        attn_bias = jnp.repeat(mask[:, None, None, :], x.shape[1], axis=2)
        attn_bias = jnp.tril(attn_bias, k=attn_bias.shape[3] - attn_bias.shape[2])
        attn_bias = -1e9 * (1 - attn_bias.astype(jnp.bfloat16))

        x = self.wte(x)
        for layer in self.layer:
            x = layer(x, attn_bias)
        return self.head(self.head_norm(x))
