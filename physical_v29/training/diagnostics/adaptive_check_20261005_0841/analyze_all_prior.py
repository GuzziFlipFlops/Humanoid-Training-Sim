from pathlib import Path
OUT=Path(__file__).resolve().parent
source=(OUT/'analyze_matched_transfer.py').read_text()
source=source.replace("['control','candidate']", "['control','all_prior']")
source=source.replace("reload=OUT/'candidate_full_reload'", "reload=OUT/'all_prior_full_reload'")
source=source.replace("OUT/'matched_comparison.json'", "OUT/'all_prior_comparison.json'")
exec(compile(source,str(OUT/'analyze_matched_transfer.py'),'exec'))
