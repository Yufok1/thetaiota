#!/usr/bin/env python3
"""
Simple local trainer for the tiny LM used by ChatEngine.
Trains on project text files and saves to checkpoints/tiny_lm.pt

Usage:
  python train_tiny_lm.py --epochs 2 --lr 3e-4 --context 128 --batch 8
"""

import os
import glob
import argparse
import random
import math
import torch
import torch.nn as nn
from chat_engine import ByteTokenizer, TinyCausalLM


def load_corpus(paths: list[str]) -> str:
    texts = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
        except Exception:
            continue
    return "\n\n".join(texts)


def make_batches(ids: list[int], context: int, batch_size: int):
    # Yield random batches of token ids
    total = len(ids)
    while True:
        xs = []
        ys = []
        for _ in range(batch_size):
            if total <= context + 1:
                start = 0
            else:
                start = random.randint(0, total - context - 2)
            chunk = ids[start:start+context+1]
            x = chunk[:-1]
            y = chunk[1:]
            xs.append(x)
            ys.append(y)
        yield torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--d_model', type=int, default=1024)
    ap.add_argument('--d_ff', type=int, default=4096)
    ap.add_argument('--n_layers', type=int, default=12)
    ap.add_argument('--steps_per_epoch', type=int, default=250)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    os.makedirs('checkpoints', exist_ok=True)

    # Collect project text
    candidates = []
    for pat in ['*.md', '*.txt', '*.py']:
        candidates.extend(glob.glob(pat))
    corpus = load_corpus(candidates)
    tok = ByteTokenizer()
    ids = tok.encode(corpus, add_special=False)
    print(f"Loaded corpus tokens: {len(ids)} from {len(candidates)} files")

    # Model
    model = TinyCausalLM(vocab_size=tok.vocab_size, d_model=args.d_model, d_ff=args.d_ff, max_len=args.context, n_layers=args.n_layers)
    model.to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    batches = make_batches(ids, context=args.context, batch_size=args.batch)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step in range(args.steps_per_epoch):
            x, y = next(batches)
            x = x.to(args.device)
            y = y.to(args.device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            if (step + 1) % 50 == 0:
                print(f"epoch {epoch+1} step {step+1}/{args.steps_per_epoch} loss {total_loss/(step+1):.4f}")
        print(f"epoch {epoch+1} avg loss {total_loss/args.steps_per_epoch:.4f}")

    # Save
    save_path = os.path.join('checkpoints', 'tiny_lm.pt')
    torch.save({'state_dict': model.state_dict(), 'config': vars(args)}, save_path)
    print(f"Saved LM to {save_path}")


if __name__ == '__main__':
    main()


