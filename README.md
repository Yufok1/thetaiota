## ThetaIota — One Brain, Three Lobes

A native, on‑device, self‑reflective AI that learns, explains itself, and converses. It runs as three cooperating agents (A/B/C) that together form one brain:
- Agent A: leader and user‑facing surface
- Agents B and C: peers that vote (2‑of‑3 quorum), send heartbeats, and auto‑sync weights to the leader’s official checkpoint

No external models or APIs. All conversation and learning run locally.

---

### System map
- **Federation runtime**
  - `start_all.bat`: launches three API servers (A:8081, B:8082, C:8083)
  - `server_main.py`: production launcher (reads .env, computes port/peers per replica)
  - `phase4_api_server.py`: FastAPI server exposing control/chat/memory/ops (now with auth/rate‑limit/healthz/readyz/metrics)
  - `phase4_agent_service.py`: lifecycle, events, leader routing, quorum, replication
- **Agent cognition & learning**
  - `phase3_agent.py`: self‑aware learner (meta‑controller, reflections)
  - `task_spawner.py`, `meta_controller.py`, `curriculum_dataset.py`
  - `transformer_model.py`, `toy_dataset.py`
- **Memory & introspection**
  - `memory_db.py` (SQLite): `introspection`, `tasks`, `meta_events`, `config`, `chat_sessions`, `chat_messages`; view: `memories`
  - Hardened: WAL, NORMAL sync, busy_timeout
  - `reflection_explainer.py`, `memory_summarizer.py`, `human_feedback_system.py`
- **Conversation**
  - `chat_engine.py`: reflective chat + enhanced local LM (150M params, 12-layer transformer), session memory, temperature/top‑k
  - `train_tiny_lm.py`: trains 150M parameter conversational LM on project text (100 epochs default, saves `checkpoints/tiny_lm.pt`)
  - `train_conversational_lm_windows.py`: Windows-compatible training with live text generation display and 84 diverse test prompts
- **CLI**
  - `djinn.bat`: unified CLI shim (no `python` needed)
  - `cli_control.py`: natural‑language prompt + advanced commands
  - `djinn-train.bat`: batch wrapper for the tiny LM trainer

---

### Quick start (Windows)
1) Launch federation servers: double‑click `start_all.bat` (opens 3 windows)
2) Start services (training/decision loops):
   - `.\djinn.bat start --agent ALL`
3) Talk naturally (no flags):
   - `.\djinn.bat` → opens a simple prompt (type sentences; Ctrl+C to exit)
   - One‑shot: `.\djinn.bat "hello there"`
   - Toggle local generator in the prompt: `!lm` (switch back: `!reflect`)
4) **Recommended**: train the enhanced local generator:
   - `.\djinn-train.bat` → trains 150M parameter model for 100 epochs → writes `checkpoints\tiny_lm.pt`
   - OR: `python train_conversational_lm_windows.py --epochs 100 --use_amp` → enhanced training with live text generation display
   - Takes 1-2 hours depending on hardware (CUDA recommended, 3GB+ VRAM)
   - Restart A/B/C windows so new weights are loaded automatically

Dependencies (install once):
- Python 3.10+
- pip install: `fastapi uvicorn pydantic aiohttp numpy torch python-dotenv slowapi starlette-exporter`
  - (slowapi/starlette‑exporter are optional but recommended)

---

### Modes
- **Reflect (default)**: conversational replies from the agent's own memory/state (no generator). Deterministic; great for status/explanations.
- **LM (local generator)**: replies written by a 150M parameter on‑device Transformer (12 layers, 1024 d_model). Silver-tongued conversational AI; reflective fallback if the LM is unsure.
- **Training (learning)**: background epochs improving the base learner/meta‑controller; independent of chat mode (can run while chatting).

---

### Federation (one brain, three lobes)
- **Leader/user surface**: A is the single chat surface; B/C forward chat/tasks to A.
- **Quorum voting**: 2‑of‑3 across A/B/C gates sensitive self‑updates (e.g., `FINE_TUNE_NOW`).
- **Heartbeats & convergence**: leader identity and official step advertised; followers auto‑pull the official checkpoint when newer.
- **Memory replication**: chat turns and meta‑events recorded on the leader and replicated to peers for shared view.

---

### .env (production config)
Common keys (optional in dev):
- `RA_BIND_HOST=127.0.0.1`
- `RA_PORT_BASE=8081` (replicas use 8081..8083)
- `RA_REPLICAS=3`
- `RA_DB_PATH=/opt/recursive-ai/app/agent_A.db` (server_main computes defaults if unset)
- `RA_CHECKPOINT_DIR=/opt/recursive-ai/checkpoints`
- `RA_AUTH_SECRET=<long-random>` (when set, protected endpoints require `Authorization: Bearer <token>`)
- `RA_RATE_LIMIT=20/minute`

If `RA_AUTH_SECRET` is unset, auth is disabled (for local development). The current CLI does not attach a bearer token; use unset secret for local dev, set it for production behind your edge.

---

### CLI — natural language
- **Interactive prompt (recommended)**:
  - `.\djinn.bat` → just type sentences
  - Toggles inside the prompt: `!lm` (generator on), `!reflect` (back to reflective)
- **One‑shot chat**:
  - `.\djinn.bat "hello there"`

### CLI — advanced
- **Start/stop/pause/resume/status**
  - `start --agent ALL`
  - `stop  --agent ALL`
  - `pause --agent ALL`
  - `resume --agent ALL`
  - `status --agent ALL`
- **Chat (one‑shot)**
  - `chat --agent A --text "hello" [--mode reflect|lm] [--session s] [--temperature 0.8] [--top_k 20]`
- **Tasks**
  - `task --agent C --task_type fine_tune --priority 1 --objective "stabilize" --payload "{}"`
- **Quorum test**
  - `quorum --val_loss 0.74`
- **Checkpoints**
  - `checkpoint --agent A --op promote|pull`
  - `checkpoint-pull-if-newer --agent ALL`
- **Memory queries**
  - `query --agent A --text "What was my last decision?"`

PowerShell tip: prefix local scripts with `.\` (e.g., `.\djinn.bat`).

---

### API (selected)
- **Lifecycle**: POST `/agent/start|pause|resume|stop` (protected), GET `/agent/status`, GET `/health`
- **Chat**: POST `/chat` with `{ text, mode?, session?, temperature?, top_k? }`
- **Memory**: POST `/memory/query` with `{ query }`, GET `/memory/decisions`, GET `/memory/reflections`
- **Federation & ops**: POST `/messages/receive` (heartbeat, vote_request, chat_replicate, task_replicate), POST `/quorum/test` (protected), POST `/checkpoint` (protected), POST `/checkpoint/pull_if_newer`
- **Ops**: GET `/healthz` (fast liveness), GET `/readyz` (DB+agent check), GET `/metrics` (if `starlette_exporter` installed)

---

### Production rollout (dockerless)
- **Linux (systemd)**
  - Create `/etc/systemd/system/recursive-ai@.service` executing:
    - `ExecStart=/opt/recursive-ai/env/bin/python /opt/recursive-ai/app/server_main.py --replica-index %i`
  - Use `.env` in `/opt/recursive-ai/config/.env` to set ports, secret, rate limits.
  - Enable/start `recursive-ai@1`, `recursive-ai@2`, `recursive-ai@3`.
- **Windows services**
  - Use NSSM or Task Scheduler to run three services:
    - `python server_main.py --replica-index 1|2|3`
- **Edge**
  - Put Nginx/Caddy in front (TLS, rate‑limit if desired). Our app also supports internal rate limiting.

---

### Troubleshooting
- **"Agent service not initialized" in status** → `.\djinn.bat start --agent ALL`
- **B/C timeout on start** → retry; check `http://127.0.0.1:8082/health` or `:8083/health` (`agent_connected: false` = API up, service not started yet)
- **404s at startup** → wait a few seconds after `start_all.bat`
- **Only A responds** → by design (A is leader/user surface). B/C contribute quorum and weight sync.
- **New LM weights not applied** → run `djinn-train.bat`, restart A/B/C windows, then `start --agent ALL`.

---

### File map
- `start_all.bat`, `server_main.py`
- `djinn.bat`, `cli_control.py`, `djinn-train.bat`
- `phase4_api_server.py`, `phase4_agent_service.py`
- `chat_engine.py`, `train_tiny_lm.py`
- `memory_db.py`

---

### Roadmap (scaling safely)
- Fine‑tune tiny LM on domain text; keep reflective fallback
- Activation checkpointing, gradient accumulation, 8‑bit optimizers
- LoRA/QLoRA adapters; longer context; top‑p/stop‑sequences
- Retrieval‑augmented replies from MemoryDB and reflections
- Optional RLHF later
