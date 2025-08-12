#!/usr/bin/env python3
"""
Native CLI to control local agents (no browser, no external deps).

Quick use:
  djinn.bat "hello there"   -> sends chat to leader (Agent A)
  djinn.bat                  -> opens NL prompt (type sentences; Ctrl+C to exit)

Advanced:
  python cli_control.py start --agent A  (use ALL to target all)
  python cli_control.py status --agent B   (use ALL to target all)
  python cli_control.py task --agent C --task_type fine_tune --priority 0 --objective "User requested fine-tune"
  python cli_control.py query --agent A --text "What was my last decision?"
  python cli_control.py chat --agent A --text "hello" [--mode reflect|lm] [--session mychat] [--temperature 0.8] [--top_k 20]
  python cli_control.py ask --agent A [--mode reflect|lm] [--session mychat]
  python cli_control.py quorum [--agent A|B|C] [--val_loss 0.70]
  python cli_control.py checkpoint --agent A --op promote|pull
  python cli_control.py checkpoint-pull-if-newer --agent A   (use ALL to target all)

Agents mapping:
  A -> http://127.0.0.1:8081
  B -> http://127.0.0.1:8082
  C -> http://127.0.0.1:8083
"""

import sys
import os
import json
import time
from urllib import request, parse, error


DEFAULT_MAP = {
    'A': 'http://127.0.0.1:8081',
    'B': 'http://127.0.0.1:8082',
    'C': 'http://127.0.0.1:8083',
}

# Simple HTTP client tuning
TIMEOUT_SEC = 120.0
RETRIES = 3
RETRY_DELAY_SEC = 1.0


def _auth_headers():
    headers = {"Accept": "application/json"}
    tok = os.getenv("RA_AUTH_SECRET")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def http_get(url: str):
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = request.Request(url, headers=_auth_headers())
            with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                return True, resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            last_err = e
            time.sleep(RETRY_DELAY_SEC)
    return False, str(last_err)


def http_post_json(url: str, payload: dict | None = None):
    data = None
    headers = _auth_headers()
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = request.Request(url, method='POST', data=data, headers=headers)
            with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                return True, resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            last_err = e
            time.sleep(RETRY_DELAY_SEC)
    return False, str(last_err)


def resolve_base(agent: str | None, port: str | None) -> str:
    if port:
        return f"http://127.0.0.1:{port}"
    if agent:
        agent = agent.upper()
        if agent == 'ALL':
            # Placeholder base, multi-target handled in commands
            return 'ALL'
        if agent in DEFAULT_MAP:
            return DEFAULT_MAP[agent]
    # Fallback
    return DEFAULT_MAP['A']


def cmd_start(base: str):
    if base == 'ALL':
        for k, url in DEFAULT_MAP.items():
            ok, body = http_post_json(f"{url}/agent/start")
            print(f"[{k}] " + (body if ok else f"ERROR: {body}"))
    else:
        ok, body = http_post_json(f"{base}/agent/start")
        print(body if ok else f"ERROR: {body}")


def cmd_pause(base: str):
    if base == 'ALL':
        for k, url in DEFAULT_MAP.items():
            ok, body = http_post_json(f"{url}/agent/pause")
            print(f"[{k}] " + (body if ok else f"ERROR: {body}"))
    else:
        ok, body = http_post_json(f"{base}/agent/pause")
        print(body if ok else f"ERROR: {body}")


def cmd_resume(base: str):
    if base == 'ALL':
        for k, url in DEFAULT_MAP.items():
            ok, body = http_post_json(f"{url}/agent/resume")
            print(f"[{k}] " + (body if ok else f"ERROR: {body}"))
    else:
        ok, body = http_post_json(f"{base}/agent/resume")
        print(body if ok else f"ERROR: {body}")


def cmd_stop(base: str):
    if base == 'ALL':
        for k, url in DEFAULT_MAP.items():
            ok, body = http_post_json(f"{url}/agent/stop")
            print(f"[{k}] " + (body if ok else f"ERROR: {body}"))
    else:
        ok, body = http_post_json(f"{base}/agent/stop")
        print(body if ok else f"ERROR: {body}")


def cmd_status(base: str):
    if base == 'ALL':
        for k, url in DEFAULT_MAP.items():
            ok, body = http_get(f"{url}/agent/status")
            print(f"[{k}] " + (body if ok else f"ERROR: {body}"))
    else:
        ok, body = http_get(f"{base}/agent/status")
        print(body if ok else f"ERROR: {body}")


def cmd_task(base: str, task_type: str, priority: int | None, objective: str | None, payload: dict | None):
    data = {
        'task_type': task_type,
        'priority': priority,
        'objective': objective,
        'payload': payload or {},
    }
    ok, body = http_post_json(f"{base}/task", data)
    print(body if ok else f"ERROR: {body}")


def cmd_query(base: str, text: str):
    # Uses existing memory query endpoint
    data = { 'query': text }
    ok, body = http_post_json(f"{base}/memory/query", data)
    print(body if ok else f"ERROR: {body}")


def cmd_chat(base: str, text: str, mode: str | None):
    data = { 'text': text }
    if mode:
        data['mode'] = mode
    ok, body = http_post_json(f"{base}/chat", data)
    print(body if ok else f"ERROR: {body}")


def cmd_chat_with_session(base: str, text: str, mode: str | None, session: str | None):
    data = { 'text': text }
    if mode:
        data['mode'] = mode
    if session:
        data['session'] = session
    ok, body = http_post_json(f"{base}/chat", data)
    print(body if ok else f"ERROR: {body}")




def cmd_ask(base: str, mode: str | None, session: str | None):
    print("Type 'exit' or 'quit' to leave. Press Ctrl+C to stop.")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            # Inline toggles: !lm and !reflect
            local_mode = mode
            if line.startswith('!lm'):
                local_mode = 'lm'
                line = line[3:].strip() or ""
            elif line.startswith('!reflect'):
                local_mode = 'reflect'
                line = line[9:].strip() or ""
            data = { 'text': line }
            if local_mode:
                data['mode'] = local_mode
            if session:
                data['session'] = session
            ok, body = http_post_json(f"{base}/chat", data)
            print(body if ok else f"ERROR: {body}")
    except KeyboardInterrupt:
        pass


def cmd_quorum(base: str, action: str, val_loss: float | None):
    payload = { 'action': action, 'metrics': {} }
    if val_loss is not None:
        payload['metrics']['val_loss'] = float(val_loss)
    if base == 'ALL':
        for k, url in DEFAULT_MAP.items():
            print(f"\n== Quorum test on {k} ({url}) ==")
            ok, body = http_post_json(f"{url}/quorum/test", payload)
            print(body if ok else f"ERROR: {body}")
    else:
        ok, body = http_post_json(f"{base}/quorum/test", payload)
        print(body if ok else f"ERROR: {body}")


def cmd_checkpoint(base: str, op: str):
    data = { 'op': op }
    ok, body = http_post_json(f"{base}/checkpoint", data)
    print(body if ok else f"ERROR: {body}")

def cmd_checkpoint_pull_if_newer(base: str):
    ok, body = http_post_json(f"{base}/checkpoint/pull_if_newer", {})
    print(body if ok else f"ERROR: {body}")


def parse_args(argv: list[str]):
    if len(argv) < 2:
        # Will fall back to interactive NL mode in main()
        return 'nl', None, None, None, None, None, None, None, 'FINE_TUNE_NOW', None, None, None, None, None
    cmd = argv[1].lower()
    # simple arg parser
    agent = None
    port = None
    task_type = None
    priority = None
    objective = None
    text = None
    chat_mode = None
    session = None
    temperature = None
    top_k = None
    payload = None
    action = 'FINE_TUNE_NOW'
    val_loss = None
    op = None
    i = 2
    while i < len(argv):
        a = argv[i]
        if a in ('--agent', '-a') and i + 1 < len(argv):
            agent = argv[i+1]
            i += 2
        elif a == '--port' and i + 1 < len(argv):
            port = argv[i+1]
            i += 2
        elif a == '--task_type' and i + 1 < len(argv):
            task_type = argv[i+1]
            i += 2
        elif a == '--priority' and i + 1 < len(argv):
            try:
                priority = int(argv[i+1])
            except ValueError:
                priority = None
            i += 2
        elif a == '--objective' and i + 1 < len(argv):
            objective = argv[i+1]
            i += 2
        elif a == '--text' and i + 1 < len(argv):
            text = argv[i+1]
            i += 2
        elif a == '--mode' and i + 1 < len(argv):
            chat_mode = argv[i+1]
            i += 2
        elif a == '--session' and i + 1 < len(argv):
            session = argv[i+1]
            i += 2
        elif a == '--temperature' and i + 1 < len(argv):
            try:
                temperature = float(argv[i+1])
            except ValueError:
                temperature = None
            i += 2
        elif a == '--top_k' and i + 1 < len(argv):
            try:
                top_k = int(argv[i+1])
            except ValueError:
                top_k = None
            i += 2
        elif a == '--action' and i + 1 < len(argv):
            action = argv[i+1]
            i += 2
        elif a == '--val_loss' and i + 1 < len(argv):
            try:
                val_loss = float(argv[i+1])
            except ValueError:
                val_loss = None
            i += 2
        elif a == '--payload' and i + 1 < len(argv):
            try:
                payload = json.loads(argv[i+1])
            except Exception:
                payload = None
            i += 2
        elif a == '--op' and i + 1 < len(argv):
            op = argv[i+1]
            i += 2
        else:
            i += 1
    return cmd, agent, port, task_type, priority, objective, text, chat_mode, action, val_loss, op, payload, session, temperature, top_k


def main():
    # If called with no args: open a natural-language prompt to leader A
    if len(sys.argv) == 1:
        base = DEFAULT_MAP['A']
        print("Natural-language prompt (Agent A). Type anything; Ctrl+C to exit.")
        print("Toggle modes: !lm (local generator) or !reflect (memory-based)")
        current_mode = 'reflect'
        try:
            while True:
                line = input("> ").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit"):
                    break
                    
                # Handle mode toggles
                if line.startswith('!lm'):
                    current_mode = 'lm'
                    line = line[3:].strip()
                    if not line:
                        print("Switched to LM mode")
                        continue
                elif line.startswith('!reflect'):
                    current_mode = 'reflect'
                    line = line[9:].strip()
                    if not line:
                        print("Switched to reflect mode")
                        continue
                        
                data = { 'text': line, 'session': 'main', 'mode': current_mode }
                ok, body = http_post_json(f"{base}/chat", data)
                if ok:
                    try:
                        resp = json.loads(body)
                        if resp.get('success') and resp.get('data', {}).get('reply'):
                            print(resp['data']['reply'])
                        else:
                            print(f"Agent: {resp.get('message', 'No response')}")
                    except json.JSONDecodeError:
                        print(body)
                else:
                    print(f"ERROR: {body}")
        except KeyboardInterrupt:
            return

    cmd, agent, port, task_type, priority, objective, text, chat_mode, action, val_loss, op, payload, session, temperature, top_k = parse_args(sys.argv)
    base = resolve_base(agent, port)
    if cmd == 'start':
        cmd_start(base)
    elif cmd == 'pause':
        cmd_pause(base)
    elif cmd == 'resume':
        cmd_resume(base)
    elif cmd == 'stop':
        cmd_stop(base)
    elif cmd == 'status':
        cmd_status(base)
    elif cmd == 'task':
        if not task_type:
            print('ERROR: --task_type required')
            sys.exit(2)
        cmd_task(base, task_type, priority, objective, payload)
    elif cmd == 'query':
        if not text:
            print('ERROR: --text required')
            sys.exit(2)
        cmd_query(base, text)
    elif cmd == 'chat':
        if not text:
            print('ERROR: --text required')
            sys.exit(2)
        data = { 'text': text }
        if chat_mode:
            data['mode'] = chat_mode
        if session:
            data['session'] = session
        if temperature is not None:
            data['temperature'] = temperature
        if top_k is not None:
            data['top_k'] = top_k
        ok, body = http_post_json(f"{base}/chat", data)
        print(body if ok else f"ERROR: {body}")
    elif cmd == 'ask':
        cmd_ask(base, chat_mode, session)
    elif cmd == 'quorum':
        # If agent not provided, run for all A/B/C
        if not agent and not port:
            for k, url in DEFAULT_MAP.items():
                print(f"\n== Quorum test on {k} ({url}) ==")
                cmd_quorum(url, action, val_loss)
        else:
            cmd_quorum(base, action, val_loss)
    elif cmd == 'checkpoint':
        if not op:
            print('ERROR: --op promote|pull required')
            sys.exit(2)
        if base == 'ALL':
            for k, url in DEFAULT_MAP.items():
                print(f"\n== Checkpoint {op} on {k} ({url}) ==")
                cmd_checkpoint(url, op)
        else:
            cmd_checkpoint(base, op)
    elif cmd == 'checkpoint-pull-if-newer':
        if base == 'ALL':
            for k, url in DEFAULT_MAP.items():
                print(f"\n== Checkpoint pull_if_newer on {k} ({url}) ==")
                cmd_checkpoint_pull_if_newer(url)
        else:
            cmd_checkpoint_pull_if_newer(base)
    else:
        # Natural-language fallback: treat remaining args as a message to leader A
        base = DEFAULT_MAP['A']
        msg = " ".join(sys.argv[1:])
        ok, body = http_post_json(f"{base}/chat", { 'text': msg, 'session': 'main' })
        print(body if ok else f"ERROR: {body}")


if __name__ == '__main__':
    main()


