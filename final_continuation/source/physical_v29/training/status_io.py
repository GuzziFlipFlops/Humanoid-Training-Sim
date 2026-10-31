"""Atomic local status writes tolerate Windows readers that briefly lock rename."""
import json
import os
from pathlib import Path
import time


def atomic_json(path, data, timeout=5.):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
    finally:
        if temporary.exists():
            temporary.unlink()
