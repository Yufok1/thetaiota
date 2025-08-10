#!/usr/bin/env python3
"""
Server launcher for production-style runs without Docker.

Usage (example):
  python server_main.py --replica-index 1

Reads environment variables (via .env if present):
  RA_BIND_HOST, RA_PORT_BASE, RA_REPLICAS, RA_DB_PATH,
  RA_CHECKPOINT_DIR, RA_AUTH_SECRET, RA_MAX_VRAM_MB

Computes port and peers from RA_PORT_BASE and replica index, then launches
Phase4APIServer on that port with appropriate config.
"""

import argparse
import asyncio
import os
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):  # fallback no-op
        return False

from phase4_api_server import Phase4APIServer


def build_config(replica_index: int) -> Dict[str, Any]:
    load_dotenv()  # best-effort
    bind_host = os.getenv("RA_BIND_HOST", "127.0.0.1")
    base_port = int(os.getenv("RA_PORT_BASE", "8081"))
    num_replicas = int(os.getenv("RA_REPLICAS", "3"))
    db_path = os.getenv("RA_DB_PATH", f"agent_{chr(64+replica_index)}.db")
    ckpt_dir = os.getenv("RA_CHECKPOINT_DIR", "checkpoints")
    auth_secret = os.getenv("RA_AUTH_SECRET")

    this_port = base_port + replica_index - 1
    peers: List[str] = []
    for i in range(1, num_replicas + 1):
        if i == replica_index:
            continue
        peers.append(f"http://{bind_host}:{base_port + i - 1}")

    # Leader is replica 1 by default
    is_leader = (replica_index == 1)
    leader_url = f"http://{bind_host}:{base_port}" if not is_leader else None

    agent_id = f"agent_{chr(64+replica_index)}"

    config: Dict[str, Any] = {
        'agent_id': agent_id,
        'db_path': db_path,
        'human_feedback_enabled': True,
        'peers': peers,
        'heartbeat_interval_s': 5,
        'leader_id': 'agent_A',
    }
    if leader_url:
        config['leader_url'] = leader_url

    # Pass through for Path config used by agent for checkpoints (Phase2/3)
    config['checkpoint_dir'] = ckpt_dir

    # Attach auth secret into process env for the API layer to read
    if auth_secret:
        os.environ['RA_AUTH_SECRET'] = auth_secret

    return config, this_port


async def main_async(replica_index: int):
    config, port = build_config(replica_index)
    server = Phase4APIServer(config, port=port)
    await server.start_server()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replica-index', type=int, required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.replica_index))


if __name__ == "__main__":
    main()


