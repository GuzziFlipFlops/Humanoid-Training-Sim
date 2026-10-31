from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261005_0540'
text=(OLD/'analyze_matched_exposure.py').read_text()
text=text.replace("        'max_pre_update_KL':", "        'phase_transfer_last':rows[-1].get('phase_transfer'),\n        'max_pre_update_KL':")
text=text.replace("print(json.dumps({k:","print(json.dumps({k:")
(OUT/'analyze_matched_transfer.py').write_text(text)
print('Prepared matched independent gait/posture/reload analysis.')
