"""Standard-library mock only: healthy progress and save after elapsed budget."""
import argparse,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args,_=p.parse_known_args()
out=Path(args.output)
(out/'progress.json').write_text(json.dumps(dict(iteration=1,time=time.time(),collect_seconds=.001,learn_seconds=.001)))
time.sleep(3.2)
(out/'result.json').write_text(json.dumps(dict(completed=True,cpu_mock=True,checkpoint='mock-save-after-deadline')))
