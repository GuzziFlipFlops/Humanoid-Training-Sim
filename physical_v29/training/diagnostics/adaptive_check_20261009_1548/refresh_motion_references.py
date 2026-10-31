"""Primary-source metadata only; never execute fetched repository code."""
import hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
urls=['https://api.github.com/repos/unitreerobotics/unitree_rl_mjlab/commits/main',
      'https://api.github.com/repos/unitreerobotics/unitree_rl_mjlab/contents/src/assets/motions/g1',
      'https://api.github.com/repos/unitreerobotics/unitree_rl_mjlab/contents/src/assets/motions/g1_23dof']
records=[]
for i,url in enumerate(urls):
    row=dict(url=url,fetched=False)
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'humanoid-motion-reference-audit','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(request,timeout=8) as r:data=r.read()
        file=OUT/f'official_motion_metadata_{i}.json';file.write_bytes(data);parsed=json.loads(data)
        row.update(fetched=True,file=str(file),sha256=hashlib.sha256(data).hexdigest(),
            summary=dict(commit=parsed['sha']) if isinstance(parsed,dict) else
            dict(files=[dict(name=x['name'],size=x['size'],sha=x['sha'],url=x['html_url']) for x in parsed]))
    except Exception as error:row['error']=repr(error)
    records.append(row)
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),official_repository_metadata=records,
    primary_human=dict(url='https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/',pubmed='https://pubmed.ncbi.nlm.nih.gov/19640879/',
        direct_fetch='Browser challenge; PubMed direct rendering did not expose article text.',
        indexed_primary_Figure6_caption_confirms_shoulder_elbow_angle_time_plots=True,
        usable_numerical_curves_obtained=False,axes_signs_rest_poses_speed_morphology_mapping_verified=False),
    sequential_learning_lead=dict(url='https://proceedings.mlr.press/v15/ross11a.html',
        primary_source_read=True,finding='The paper studies sequential prediction when actions change the subsequent observation distribution. It motivates checking teacher labels on student states. Our conditional PPO auxiliary is not DAgger and does not inherit its formal guarantees.'),
    usable_speed_matched_official_Unitree_locomotion_encoder_reference_obtained=False,
    no_remote_code_executed=True,no_reference_curve_fabricated=True,
    upper_reference_comparison='Actual robot timelines show bilateral/ipsilateral phase with amplitude and held/slow drift. Published human shoulder/elbow mechanics are feature references only. No mapped speed-matched human/Unitree encoder curve or calibrated optical gaze is available, so no normal-motion certification is made.')
(OUT/'updated_motion_reference_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
