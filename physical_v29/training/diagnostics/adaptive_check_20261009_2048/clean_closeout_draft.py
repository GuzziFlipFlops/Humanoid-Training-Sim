"""Remove superseded draft strings in this check's CPU publication helper only."""
from pathlib import Path
root = Path(__file__).resolve().parent
path = (root / 'record_current_check.py').resolve()
assert path.parent == root and root.name == 'adaptive_check_20261009_2048'
text = path.read_text(encoding='utf-8')
start = text.index('priority=[\n')
end = text.index("priority=read(OUT/'next_priorities_draft.json')['priorities']", start)
text = text[:start] + text[end:]
start = text.index("report=f'''# Five-hour check")
end = text.index('from closeout_report import render', start)
text = text[:start] + text[end:]
compile(text, str(path), 'exec')
path.write_text(text, encoding='utf-8')
print('Removed superseded report and priority draft strings; execution sources untouched.')
