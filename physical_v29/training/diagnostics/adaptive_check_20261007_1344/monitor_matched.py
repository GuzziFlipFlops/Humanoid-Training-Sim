"""Compact read-only progress; no simulator or checkpoint load."""
import json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;root=OUT/'matched_heading'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
done=[];active=[]
for f in sorted(root.iterdir()):
 if not f.is_dir():continue
 if (f/'result.json').exists():
  done.append(dict(name=f.name,completed=read(f/'result.json')['completed']))
 elif (f/'progress.jsonl').exists():
  rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
  if rows:
   r=rows[-1];active.append(dict(name=f.name,updates=len(rows),GPU=r['gpu']['free_mb'],host=r['gpu']['ram_available_mb'],KL=r['ppo']['pre_update_kl'],command_mismatch=r['behavior']['action_command_difference']))
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),completed=done,active=active,all_done=(root/'completion.json').exists()),indent=2))
