#!/usr/bin/env python3
"""
Minimal native chat engine for the Phase 4/5 agent.

Design goals:
- No external runtimes (pure Python / PyTorch optional)
- Default "reflective" mode uses agent memory to answer
- Optional tiny causal LM scaffold (greedy) for future local text gen

Notes:
- The LM is intentionally tiny and untrained by default; reflective mode
  provides useful answers today. We keep the LM path for future training.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Deque, Tuple
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = object  # type: ignore


# -------------------- Tokenizer --------------------

class ByteTokenizer:
    """Simple byte-level tokenizer (UTF-8) with BOS/EOS specials.
    Vocab: 0..255 (bytes) + 256=BOS + 257=EOS
    """

    def __init__(self):
        self.BOS = 256
        self.EOS = 257
        self.vocab_size = 258

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        b = text.encode("utf-8", errors="ignore")
        ids = list(b)
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        # strip specials
        filtered = [i for i in ids if i < 256]
        return bytes(filtered).decode("utf-8", errors="ignore")


# -------------------- Tiny Causal LM --------------------

if TORCH_AVAILABLE:
    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.1):
            super().__init__()
            self.scale = math.sqrt(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D]
            q = self.q(x); k = self.k(x); v = self.v(x)
            att = (q @ k.transpose(-2, -1)) / self.scale
            # causal mask: allow j <= i
            L = x.size(1)
            mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0)
            att = att.masked_fill(mask == 0, float('-inf'))
            w = torch.softmax(att, dim=-1)
            w = self.dropout(w)
            out = w @ v
            return self.o(out)

    class TinyCausalLM(nn.Module):
        def __init__(self, vocab_size: int = 258, d_model: int = 1024, d_ff: int = 4096, max_len: int = 512, dropout: float = 0.1, n_layers: int = 12):
            super().__init__()
            self.max_len = max_len
            self.n_layers = n_layers
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(max_len, d_model)
            
            # Multiple transformer layers for better capacity
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'ln1': nn.LayerNorm(d_model),
                    'attn': CausalSelfAttention(d_model, dropout),
                    'ln2': nn.LayerNorm(d_model),
                    'ff': nn.Sequential(
                        nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
                    )
                }) for _ in range(n_layers)
            ])
            
            self.ln_final = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
            self.drop = nn.Dropout(dropout)
            self._init()

        def _init(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            # idx: [B, L]
            B, L = idx.shape
            if L > self.max_len:
                idx = idx[:, -self.max_len:]
                L = self.max_len
            pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1)
            x = self.drop(self.tok(idx) + self.pos(pos))
            
            # Pass through multiple transformer layers
            for layer in self.layers:
                x = layer['ln1'](x + layer['attn'](x))
                x = layer['ln2'](x + layer['ff'](x))
            
            x = self.ln_final(x)
            logits = self.head(x)
            return logits

        @torch.no_grad()
        def generate(self, idx: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
            self.eval()
            for _ in range(max_new_tokens):
                logits = self(idx)  # [B,L,V]
                logits = logits[:, -1, :]
                if temperature and temperature > 0.0 and temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                if top_k and top_k > 0:
                    k = min(top_k, probs.size(-1))
                    vals, inds = torch.topk(probs, k=k, dim=-1)
                    vals = vals / vals.sum(dim=-1, keepdim=True)
                    cat = torch.distributions.Categorical(vals)
                    choice = cat.sample().unsqueeze(-1)
                    nxt = inds.gather(-1, choice)
                else:
                    nxt = probs.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt], dim=1)
                # stop early if EOS
                if int(nxt[0, 0].item()) == 257:
                    break
                if idx.size(1) > self.max_len:
                    idx = idx[:, -self.max_len:]
            return idx


@dataclass
class ChatConfig:
    mode: str = "reflect"  # "reflect" | "lm"
    max_new_tokens: int = 64
    device: str = "cpu"
    history_max_turns: int = 10
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.92
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    d_model: int = 1024
    d_ff: int = 4096
    lm_max_len: int = 512
    n_layers: int = 12


class ChatEngine:
    def __init__(self, agent_ref=None, config: Optional[ChatConfig] = None):
        self.agent_ref = agent_ref
        self.config = config or ChatConfig()
        self.tokenizer = ByteTokenizer()
        self.lm: Optional[TinyCausalLM] = None
        # session_id -> deque of (role, text)
        self.sessions: Dict[str, Deque[Tuple[str, str]]] = {}

        if TORCH_AVAILABLE:
            try:
                self.lm = TinyCausalLM(
                    vocab_size=self.tokenizer.vocab_size,
                    d_model=self.config.d_model,
                    d_ff=self.config.d_ff,
                    max_len=self.config.lm_max_len,
                    n_layers=self.config.n_layers,
                )
                self.lm.to(self.config.device)
                # Try to load fine-tuned weights if available
                self._try_load_lm_weights_default()
            except Exception:
                self.lm = None

    def attach_agent(self, agent_obj):
        self.agent_ref = agent_obj

    def _try_load_lm_weights_default(self):
        try:
            import os
            path = os.path.join("checkpoints", "tiny_lm.pt")
            if os.path.exists(path) and self.lm is not None:
                state = torch.load(path, map_location=self.config.device)
                if isinstance(state, dict) and 'state_dict' in state:
                    self.lm.load_state_dict(state['state_dict'], strict=False)
                else:
                    self.lm.load_state_dict(state, strict=False)
        except Exception:
            pass

    def chat(self, text: str, mode: Optional[str] = None, session: Optional[str] = None) -> Dict[str, Any]:
        mode = (mode or self.config.mode).lower()
        session_id = session or "default"
        self._append_history(session_id, "user", text)
        if mode == "lm" and self.lm is not None and TORCH_AVAILABLE:
            result = self._chat_lm(text, session_id=session_id)
            # If LM reply looks too weak, fall back to reflective
            if not result.get("reply") or len(result.get("reply", "").strip()) < 2:
                result = self._chat_reflect(text, session_id=session_id)
                result["mode"] = "reflect_fallback"
            self._append_history(session_id, "assistant", result.get("reply", ""))
            return result
        # default: reflective
        result = self._chat_reflect(text, session_id=session_id)
        self._append_history(session_id, "assistant", result.get("reply", ""))
        return result

    def _chat_reflect(self, text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        # Use the agent's explainer and memory to craft a conversational response
        reply_parts: List[str] = []
        human_summary: Optional[str] = None
        last_decision: Optional[str] = None
        remembers_context = False
        if self.agent_ref is not None:
            try:
                result = self.agent_ref.explainer.query_agent_memory(text)
                human_summary = (result.get("summary") or "").strip() or None
            except Exception:
                pass

            # Attempt to pull last decision
            try:
                mem = self.agent_ref.memory
                cursor = mem.conn.cursor()
                cursor.execute("""
                    SELECT * FROM meta_events 
                    WHERE event_type = 'meta_decision' 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    info = json.loads(row[3]) if isinstance(row[3], str) else {}
                    last_decision = str(info.get('action', None)) or None
            except Exception:
                pass

            # Check session memory for context
            try:
                if session_id and hasattr(self.agent_ref.memory, 'get_chat_history'):
                    hist = self.agent_ref.memory.get_chat_history(session_id, limit=6)
                    remembers_context = len(hist) >= 2
            except Exception:
                remembers_context = False

        # Build a friendlier reply
        if human_summary:
            reply_parts.append(human_summary)
        else:
            reply_parts.append(f"Got it.")
        if last_decision:
            reply_parts.append(f"Last decision: {last_decision}.")
        if remembers_context:
            reply_parts.append("I’m following the thread.")
        # Light echo to feel responsive on first turns
        if text and len(text) <= 120:
            reply_parts.append(f"You said: '{text}'.")
        final = " ".join(reply_parts).strip()
        if not final:
            final = "I’m here. How can I help?"
        return {"reply": final, "mode": "reflect"}

    def _chat_lm(self, text: str, session_id: str) -> Dict[str, Any]:
        assert self.lm is not None and TORCH_AVAILABLE
        # Build rolling prompt from session history
        history = self.sessions.get(session_id)
        prompt_lines: List[str] = []
        if history:
            for role, msg in list(history)[-self.config.history_max_turns:]:
                if role == "user":
                    prompt_lines.append(f"User: {msg}")
                else:
                    prompt_lines.append(f"Assistant: {msg}")
        prompt_lines.append(f"User: {text}")
        prompt_lines.append("Assistant:")
        prompt = "\n".join(prompt_lines)
        ids = self.tokenizer.encode(prompt, add_special=True)
        idx = torch.tensor([ids], dtype=torch.long, device=self.config.device)
        out = self.lm.generate(idx, max_new_tokens=self.config.max_new_tokens, temperature=self.config.temperature, top_k=self.config.top_k)
        gen = out[0].tolist()
        resp = self.tokenizer.decode(gen[len(ids):])
        resp = resp.strip()
        if not resp:
            resp = "(thinking…)"
        return {"reply": resp, "mode": "lm"}

    def _append_history(self, session_id: str, role: str, text: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.config.history_max_turns * 2)
        self.sessions[session_id].append((role, text))
        try:
            if self.agent_ref is not None and hasattr(self.agent_ref, 'memory'):
                self.agent_ref.memory.log_chat_message(session_id=session_id, role=role, text=text, mode=None)
        except Exception:
            pass


