"""Single snapshot of owned diagnostic progress, no simulator imports."""
import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(OUT/'monitor_check.py')],check=True,stdout=subprocess.DEVNULL)
s=json.loads((OUT/'monitor_snapshot.json').read_text());ROOT=OUT/'matched_full_state'
done=[p.parent.name for p in sorted(ROOT.glob('*/operation_receipt.json'),key=lambda p:p.stat().st_mtime)]
compare=[n for n in done if n.startswith(('seed317_','seed331_'))]
print(json.dumps(dict(utc=s['recorded_utc'],one_worker=s['actual_worker_count'],
    worker=[dict(output=Path(w['output']).name,updates=w.get('observed_updates'),GPU=w.get('free_GPU_MiB'),host=w.get('available_host_MiB'),KL=w.get('KL'),command=w.get('command_mismatch')) for w in s['actual_workers']],
    all_completed=len(done),comparison_completed=len(compare),comparison_expected=32,
    latest=done[-2:],failure=(ROOT/'failure.json').exists(),completed=(ROOT/'completion.json').exists(),resumed=(OUT/'diagnostic_resume.json').exists())))
