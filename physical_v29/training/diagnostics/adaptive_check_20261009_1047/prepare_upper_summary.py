from pathlib import Path
OUT=Path(__file__).resolve().parent
text=(OUT.parent/'adaptive_check_20261008_1946/summarize_upper_comparison.py').read_text(encoding='utf-8')
text=text.replace('for seed in [157,167]:','for seed in [179,191]:')
text=text.replace("(f/'visual_inspection.json').exists()","(f/'visual_QA.json').exists()")
(OUT/'summarize_upper_comparison.py').write_text(text,encoding='utf-8')
compile(text,str(OUT/'summarize_upper_comparison.py'),'exec')
print('Prepared exact two-seed motion comparison.')
