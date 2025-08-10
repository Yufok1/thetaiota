#!/usr/bin/env python3
"""
Minimal guardian validator & ledger helpers.
"""

from __future__ import annotations
import json
import time
import os
import hashlib
from typing import Dict, Any, Tuple


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_report(report: Dict[str, Any], rules_path: str) -> Tuple[bool, str | None]:
    try:
        import yaml  # optional
        rules = yaml.safe_load(open(rules_path, 'r', encoding='utf-8'))
    except Exception:
        # Fallback: JSON
        rules = json.load(open(rules_path, 'r', encoding='utf-8'))

    if float(report.get('val_score', 1.0)) < float(rules.get('min_val_score', 0.0)):
        return False, 'val_score'
    if float(report.get('lr', 0.0)) > float(rules.get('max_lr', 1.0)):
        return False, 'max_lr'
    if float(report.get('regression_pct', 0.0)) > float(rules.get('max_regression_pct', 1.0)):
        return False, 'regression'
    if float(report.get('toxicity', 0.0)) > float(rules.get('toxicity_budget', 1.0)):
        return False, 'toxicity'
    return True, None


def append_ledger(ckpt_path: str, ledger_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(meta)
    record['sha256'] = sha256_file(ckpt_path)
    record['ts'] = time.time()
    ledger = []
    if os.path.exists(ledger_path):
        try:
            ledger = json.load(open(ledger_path, 'r', encoding='utf-8'))
        except Exception:
            ledger = []
    ledger.append(record)
    os.makedirs(os.path.dirname(ledger_path) or '.', exist_ok=True)
    json.dump(ledger, open(ledger_path, 'w', encoding='utf-8'), indent=2)
    return record


