"""A small transformer that knows only the corpus it is given.

Adapted from nanoGPT by Andrej Karpathy (https://github.com/karpathy/nanoGPT),
MIT License, copyright (c) 2022 Andrej Karpathy: see THIRD_PARTY_NOTICES.md and
LICENSES/nanoGPT-MIT.txt. The architecture is nanoGPT's, written compactly:
pre-norm blocks, causal self-attention, a GELU MLP, and the token embedding
tied to the output head. It learns the caller's sentences (already split into
words) for a fixed number of steps, long enough to fit them: the corpus is its
whole world, so fitting it is the point. Then it is read back, for each key the
caller names:
  wte     the embedding row of the word named for the key (the weights)
  layers  the hidden state at each layer at the positions named for the key,
          averaged (the weights at work on the corpus)
Nothing is written to disk: the sentences, the network and what it gives back
live for one call. One network trains at a time; the threads it may use are
set for that time and given back after.
"""

import math
import random
import threading
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2
_one_at_a_time = threading.Lock()


class Cancelled(Exception):
    """The caller went away while the network trained."""


class _Attention(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.heads, self.drop = heads, drop
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (
            z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class _Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.ln1, self.attn, self.ln2 = (
            nn.LayerNorm(d),
            _Attention(d, heads, drop),
            nn.LayerNorm(d),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class _GPT(nn.Module):
    def __init__(self, n_vocab, block, d, layers, heads, drop=0.1):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(n_vocab, d), nn.Embedding(block, d)
        self.blocks = nn.ModuleList(_Block(d, heads, drop) for _ in range(layers))
        self.ln_f, self.drop = nn.LayerNorm(d), nn.Dropout(drop)
        self.head = nn.Linear(d, n_vocab, bias=False)
        self.head.weight = self.wte.weight  # tied, as in nanoGPT
        self.apply(
            lambda m: nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, (nn.Linear, nn.Embedding))
            else None
        )

    def forward(self, idx, states=False):
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(idx.shape[1])))
        layers = []
        for block in self.blocks:
            x = block(x)
            layers.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, layers) if states else logits


def train_corpus_gpt(
    sentences: list[list[str]],
    words: dict[str, str] | None = None,
    positions: dict[str, list[tuple[int, int]]] | None = None,
    steps: int = 1500,
    seed: int = 7,
    dimensions: int = 64,
    layers: int = 4,
    heads: int = 4,
    threads: int = 2,
    progress: Callable[[int, int, float], None] | None = None,
    every: int = 25,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Train on the sentences and read the network back for each key.

    `words` names, for a key, the word whose embedding row it gets; `positions`
    names, for a key, (sentence, word) places whose hidden states it gets,
    averaged, at each layer. A key whose word is not in the sentences, or with
    no positions, is left out. `progress(step, steps, loss)` is called every
    `every` steps and at the end; when `cancelled()` turns true the training
    stops with Cancelled.
    """
    words, positions = words or {}, positions or {}
    if not sentences or not any(sentences):
        raise ValueError("No sentences to learn.")
    if dimensions % heads:
        raise ValueError("The dimensions must divide among the heads.")
    for key, places in positions.items():
        for s, w in places:
            if not (0 <= s < len(sentences) and 0 <= w < len(sentences[s])):
                raise ValueError(
                    f"A position of {key!r} is outside the sentences: {[s, w]}."
                )
    vocab = ["<pad>", "<bos>", "<eos>"] + sorted({w for ws in sentences for w in ws})
    index = {w: i for i, w in enumerate(vocab)}
    block = max(len(ws) for ws in sentences) + 2

    def batch(rows):
        x = torch.full((len(rows), block), PAD, dtype=torch.long)
        for k, r in enumerate(rows):
            seq = [BOS] + [index[w] for w in sentences[r]] + [EOS]
            x[k, : len(seq)] = torch.tensor(seq)
        return x

    with _one_at_a_time:
        before = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            torch.manual_seed(seed)
            pick = random.Random(seed)
            gpt = _GPT(len(vocab), block, dimensions, layers, heads)
            opt = torch.optim.AdamW(
                gpt.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95)
            )
            rows = list(range(len(sentences)))
            loss = torch.tensor(0.0)
            for step in range(steps):
                if cancelled and cancelled():
                    raise Cancelled()
                lr = (
                    3e-3
                    * min(1, (step + 1) / 100)
                    * 0.5
                    * (1 + math.cos(math.pi * step / steps))
                )
                for group in opt.param_groups:
                    group["lr"] = lr
                gpt.train()
                x = batch(pick.sample(rows, min(32, len(rows))))
                logits = gpt(x[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, len(vocab)),
                    x[:, 1:].reshape(-1),
                    ignore_index=PAD,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
                opt.step()
                if progress and ((step + 1) % every == 0 or step + 1 == steps):
                    progress(step + 1, steps, float(loss.detach()))

            gpt.eval()
            wte = {
                k: gpt.wte.weight[index[w]].tolist()
                for k, w in words.items()
                if w in index
            }
            sums: dict[tuple[str, int], torch.Tensor] = {}
            counts: dict[str, int] = {}
            by_sentence: dict[int, list[tuple[str, int]]] = {}
            for key, places in positions.items():
                for s, w in places:
                    by_sentence.setdefault(s, []).append((key, w))
            with torch.no_grad():
                for s, wanted in sorted(by_sentence.items()):
                    _, states = gpt(batch([s]), states=True)
                    for key, w in wanted:
                        counts[key] = counts.get(key, 0) + 1
                        for k, h in enumerate(states):
                            at = h[0, w + 1]  # after <bos>
                            sums[(key, k)] = (
                                sums[(key, k)] + at if (key, k) in sums else at.clone()
                            )
            means = [
                {key: (sums[(key, k)] / counts[key]).tolist() for key in counts}
                for k in range(layers)
            ]
            return {
                "model": {
                    "architecture": f"nanoGPT, {layers} layers, {heads} heads, "
                    f"{dimensions} dimensions, tied embedding",
                    "parameters": sum(p.numel() for p in gpt.parameters()),
                    "vocab": len(vocab),
                    "tokens": sum(len(ws) for ws in sentences),
                    "steps": steps,
                    "seed": seed,
                    "trainLoss": round(float(loss.detach()), 4),
                },
                "wte": wte,
                "layers": means,
            }
        finally:
            torch.set_num_threads(before)
