"""Train a small causal Transformer with a coupled mixture-of-experts router."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


VOCAB_SIZE = 256


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_wikitext(config, max_train_bytes=0):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    ds = load_dataset("Salesforce/wikitext", config)

    def encode(split, limit=0):
        raw = "\n".join(ds[split]["text"]).encode("utf-8")
        if limit:
            raw = raw[:limit]
        return torch.tensor(bytearray(raw), dtype=torch.long)

    train = encode("train", max_train_bytes)
    valid = encode("validation")
    return train, valid


def get_batch(data, batch_size, seq_len, device, generator):
    if data.numel() <= seq_len + 1:
        raise ValueError("Dataset split is shorter than seq_len + 1.")

    starts = torch.randint(
        0,
        data.numel() - seq_len - 1,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(seq_len + 1)
    tokens = data[starts[:, None] + offsets[None, :]]
    x = tokens[:, :-1].to(device)
    y = tokens[:, 1:].to(device)
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(y)


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class CoupledMoE(nn.Module):
    """Top-k MoE with one learned expert-interaction correction to the router."""

    def __init__(
        self,
        d_model,
        d_ff,
        n_experts,
        top_k,
        metric_rank,
        lambda_coupling,
        dropout,
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must lie between 1 and n_experts.")
        if metric_rank < 1:
            raise ValueError("metric_rank must be positive.")

        self.n_experts = n_experts
        self.top_k = top_k
        self.metric_rank = metric_rank
        self.lambda_coupling = lambda_coupling

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.metric_basis = nn.Parameter(torch.empty(d_model, metric_rank))
        self.metric_weight = nn.Parameter(torch.zeros(metric_rank))
        self.experts = nn.ModuleList(
            Expert(d_model, d_ff, dropout) for _ in range(n_experts)
        )

        nn.init.normal_(self.metric_basis, std=1.0 / math.sqrt(d_model))

    def interaction_matrix(self):
        # q_i are the rows of the ordinary router matrix Q.
        q_u = self.router.weight @ self.metric_basis
        c = (q_u * self.metric_weight) @ q_u.transpose(0, 1)
        return c

    def route(self, x):
        logits0 = self.router(x)
        p0 = F.softmax(logits0, dim=-1)

        c = self.interaction_matrix()
        coupling = p0 @ c.transpose(0, 1)
        logits = logits0 + self.lambda_coupling * coupling
        p = F.softmax(logits, dim=-1)
        return p, p0, c

    def forward(self, x):
        p, p0, c = self.route(x)
        top_p, top_i = torch.topk(p, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        flat_x = x.reshape(-1, x.size(-1))
        flat_i = top_i.reshape(-1, self.top_k)
        flat_p = top_p.reshape(-1, self.top_k)
        out = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(flat_i == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat_x[token_idx])
            weighted = expert_out * flat_p[token_idx, slot_idx, None]
            out = out.index_add(0, token_idx, weighted)

        out = out.view_as(x)

        importance = p.mean(dim=(0, 1))
        top1 = top_i[..., 0]
        load = F.one_hot(top1, self.n_experts).float().mean(dim=(0, 1))
        aux = self.n_experts * torch.sum(importance * load)

        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1).mean()
        kl = (
            p
            * (
                torch.log(p.clamp_min(1e-9))
                - torch.log(p0.clamp_min(1e-9))
            )
        ).sum(dim=-1).mean()

        stats = {
            "aux": aux,
            "entropy": entropy,
            "kl": kl,
            "load": load,
            "coupling_norm": c.norm() / self.n_experts,
        }
        return out, stats


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.d_model)
        self.attn = CausalSelfAttention(args.d_model, args.n_heads, args.dropout)
        self.ln2 = nn.LayerNorm(args.d_model)
        self.moe = CoupledMoE(
            args.d_model,
            args.d_ff,
            args.n_experts,
            args.top_k,
            args.metric_rank,
            args.lambda_coupling,
            args.dropout,
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        moe_out, stats = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x, stats


class TransformerLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len
        self.token_embedding = nn.Embedding(VOCAB_SIZE, args.d_model)
        self.position_embedding = nn.Embedding(args.seq_len, args.d_model)
        self.blocks = nn.ModuleList(Block(args) for _ in range(args.n_layers))
        self.ln_f = nn.LayerNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens):
        length = tokens.size(1)
        if length > self.seq_len:
            raise ValueError("Input is longer than the configured sequence length.")

        pos = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(pos)[None, :, :]

        stats = []
        for block in self.blocks:
            x, block_stats = block(x)
            stats.append(block_stats)

        logits = self.lm_head(self.ln_f(x))
        return logits, stats


def summarize_router(stats):
    aux = torch.stack([s["aux"] for s in stats]).mean()
    entropy = torch.stack([s["entropy"] for s in stats]).mean()
    kl = torch.stack([s["kl"] for s in stats]).mean()
    coupling_norm = torch.stack([s["coupling_norm"] for s in stats]).mean()
    load = torch.stack([s["load"] for s in stats]).mean(dim=0)
    load_cv = load.std(unbiased=False) / load.mean().clamp_min(1e-9)

    return {
        "aux": aux,
        "entropy": entropy,
        "kl": kl,
        "coupling_norm": coupling_norm,
        "load_cv": load_cv,
    }


@torch.no_grad()
def evaluate(model, data, args, device, generator):
    model.eval()
    totals = {
        "loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "coupling_norm": 0.0,
        "load_cv": 0.0,
    }

    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.seq_len, device, generator)
        logits, stats = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        router = summarize_router(stats)

        totals["loss"] += loss.item()
        for key in ("entropy", "kl", "coupling_norm", "load_cv"):
            totals[key] += router[key].item()

    for key in totals:
        totals[key] /= args.eval_batches
    totals["ppl"] = math.exp(min(totals["loss"], 20.0))
    return totals


def smoke_test(device):
    class Args:
        seq_len = 32
        d_model = 64
        d_ff = 128
        n_heads = 4
        n_layers = 2
        n_experts = 4
        top_k = 2
        metric_rank = 8
        lambda_coupling = 0.0
        dropout = 0.0

    args = Args()
    moe = CoupledMoE(
        args.d_model,
        args.d_ff,
        args.n_experts,
        args.top_k,
        args.metric_rank,
        0.0,
        0.0,
    ).to(device)
    h = torch.randn(2, 8, args.d_model, device=device)
    p, p0, _ = moe.route(h)
    if not torch.allclose(p, p0, atol=1e-7, rtol=1e-6):
        raise AssertionError("lambda=0 must recover the ordinary router.")

    model = TransformerLM(args).to(device)
    for block in model.blocks:
        block.moe.lambda_coupling = 1.0
        with torch.no_grad():
            block.moe.metric_weight.normal_(std=0.05)

    x = torch.randint(0, VOCAB_SIZE, (2, args.seq_len), device=device)
    y = torch.randint(0, VOCAB_SIZE, (2, args.seq_len), device=device)
    logits, stats = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    loss = loss + 0.01 * summarize_router(stats)["aux"]
    loss.backward()

    grads = [block.moe.metric_weight.grad for block in model.blocks]
    if any(g is None or not torch.isfinite(g).all() for g in grads):
        raise AssertionError("Coupling parameters did not receive finite gradients.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--max-train-bytes", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--metric-rank", type=int, default=16)
    parser.add_argument("--lambda-coupling", type=float, default=0.0)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is not available. Check the NVIDIA driver and PyTorch CUDA build.")
    seed_all(args.seed)

    if args.smoke:
        smoke_test(device)
        print(f"smoke test passed on {device}")
        return

    train_data, valid_data = load_wikitext(
        args.dataset_config,
        max_train_bytes=args.max_train_bytes,
    )

    model = TransformerLM(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 1)

    history = []
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} params={n_params:,} "
        f"lambda={args.lambda_coupling:g} rank={args.metric_rank}"
    )

    progress = tqdm(
        range(1, args.steps + 1),
        desc=f"lambda={args.lambda_coupling:g}",
        dynamic_ncols=True,
    )

    for step in progress:
        model.train()
        x, y = get_batch(
            train_data,
            args.batch_size,
            args.seq_len,
            device,
            train_gen,
        )
        logits, stats = model(x)
        task_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
        )
        router = summarize_router(stats)
        loss = task_loss + args.aux_weight * router["aux"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 20 == 0:
            progress.set_postfix(train=f"{task_loss.item():.3f}")

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, valid_data, args, device, eval_gen)
            metrics["step"] = step
            history.append(metrics)
            progress.set_postfix(
                train=f"{task_loss.item():.3f}",
                val=f"{metrics['loss']:.3f}",
                kl=f"{metrics['kl']:.3g}",
            )

    result = {
        "config": vars(args),
        "device": str(device),
        "parameters": n_params,
        "history": history,
        "final": history[-1],
    }

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
