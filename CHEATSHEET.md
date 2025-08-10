# ThetaIota Cheatsheet (Comprehensive)

One brain, three lobes. Agent A is the leader/user surface; Agents B and C provide quorum and weight sync. Everything runs locally. No external models/APIs.

---

## TL;DR
- Launch servers: double-click `start_all.bat`
- Start services: `.\djinn.bat start --agent ALL`
- Talk naturally: `.\djinn.bat` (type sentences; Ctrl+C to exit)
  - Toggle local generator: `!lm`  (back to reflective: `!reflect`)
- One-shot chat: `.\djinn.bat "hello there"`
- Train enhanced LM (150M params): `.\djinn-train.bat` OR `python train_conversational_lm_windows.py --epochs 100 --use_amp`, then restart A/B/C windows and `start --agent ALL`

---

## Modes: what they are
- **Reflect (default)**: replies from agent memory/state (no generator). Deterministic, status-friendly.
- **LM (local generator)**: replies written by a 150M parameter on-device Transformer (12 layers). Silver-tongued conversational AI. Reflective fallback if unsure.
- **Training (learning)**: background epochs improve the base learner and meta-controller. Independent of chat mode.

---

## Federation: one brain across three processes
- A = leader/user-facing chat surface. B/C forward chat & tasks to A.
- Quorum: 2-of-3 approvals for self-updates (e.g., `FINE_TUNE_NOW`).
- Heartbeats: leader + official checkpoint step advertised; followers auto-pull if newer.
- Memory: chat turns and meta-events recorded on A and replicated for shared auditability.

---

## Natural-language prompt (recommended)
- Open prompt: `.\djinn.bat`
  - Type sentences. No flags needed.
  - Inline toggles: `!lm`, `!reflect`
- One-shot chat: `.\djinn.bat "hello there"`

---

## Advanced CLI (same wrapper: `.\djinn.bat`)

### Lifecycle & status
- Start all: `start --agent ALL`
- Pause/resume: `pause --agent ALL` / `resume --agent ALL`
- Stop all: `stop --agent ALL`
- Status all: `status --agent ALL`

### Chat (one-shot)
- Reflect: `chat --agent A --text "hello" --session main --mode reflect`
- LM: `chat --agent A --text "tell a short story" --mode lm --temperature 0.8 --top_k 20`

### Tasks
- Enqueue: `task --agent C --task_type fine_tune --priority 1 --objective "stabilize" --payload "{}"`

### Quorum & checkpoints
- Quorum test: `quorum --val_loss 0.74`
- Promote official: `checkpoint --agent A --op promote`
- Pull official if newer (all): `checkpoint-pull-if-newer --agent ALL`

### Memory
- Query: `query --agent A --text "What was my last decision?"`

---

## Training the local generator (enhanced LM)
- Train: `.\djinn-train.bat` → trains 150M parameter model (100 epochs) → `checkpoints\tiny_lm.pt`
- Enhanced: `python train_conversational_lm_windows.py --epochs 100 --use_amp` → live text generation display with 84 diverse test prompts
- Takes 1-2 hours (CUDA recommended for speed, 3GB+ VRAM)
- Restart A/B/C windows, then `start --agent ALL`
- In prompt, `!lm` to use the silver-tongued generator

Decode knobs:
- `--temperature 0.7..1.2`
- `--top_k 10..50`

Architecture: 12 layers, 1024 d_model, 4096 d_ff, 512 context (150M parameters)

---

## Health & diagnostics
- API health: `http://127.0.0.1:8081/health` (and 8082/8083)
- Liveness: `/healthz`, Readiness: `/readyz`
- Metrics (if `starlette_exporter` installed): `/metrics`
- Status: `.\djinn.bat status --agent ALL`

Protected write endpoints (when `RA_AUTH_SECRET` is set):
- `/agent/start`, `/agent/pause`, `/agent/resume`, `/agent/stop`
- `/feedback/submit`, `/task`, `/quorum/test`, `/checkpoint`

---

## .env keys (prod)
- `RA_BIND_HOST=127.0.0.1`
- `RA_PORT_BASE=8081` (replicas use 8081..8083)
- `RA_REPLICAS=3`
- `RA_DB_PATH` (per-replica DB path; server_main defaults if unset)
- `RA_CHECKPOINT_DIR`
- `RA_AUTH_SECRET=<long-random>` (enables bearer token auth)
- `RA_RATE_LIMIT=20/minute`

---

## Prod launcher
- Use `server_main.py` to run a replica with `.env`:
  - `python server_main.py --replica-index 1|2|3`
- Linux: systemd `recursive-ai@.service` ExecStart → python server_main.py --replica-index %i
- Windows: NSSM services with the same command

---

## Troubleshooting (quick)
- "Agent service not initialized" → `start --agent ALL`
- B/C timeout → retry start; check `/health` on ports 8082/8083
- Only A replies → by design (leader surface)
- New LM weights not used → run trainer, restart A/B/C windows, start services

---

## File map
- `start_all.bat`, `server_main.py`
- `djinn.bat`, `cli_control.py`, `djinn-train.bat`
- `phase4_api_server.py`, `phase4_agent_service.py`
- `chat_engine.py`, `train_tiny_lm.py`, `train_conversational_lm_windows.py`
- `memory_db.py`

---

## Roadmap
- ✅ Enhanced 150M parameter conversational LM (12-layer transformer)
- ✅ 100-epoch training for silver-tongued conversation quality
- ✅ Activation checkpointing, grad accumulation, AMP optimizations
- ✅ Live text generation display with 84 diverse conversation prompts
- LoRA/QLoRA adapters; longer context; top‑p; stop sequences
- Retrieval‑augmented replies from `MemoryDB` and reflections
- Optional RLHF later
