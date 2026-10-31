"""Bounded primary-source metadata review, no remote code execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,urllib.request,urllib.error
OUT=Path(__file__).resolve().parent
urls=[
 'https://api.github.com/repos/unitreerobotics/unitree_rl_mjlab/commits/main',
 'https://api.github.com/repos/unitreerobotics/unitree_rl_mjlab/contents/src/assets/motions/g1',
 'https://api.github.com/repos/unitreerobotics/unitree_rl_mjlab/contents/src/assets/motions/g1_23dof']
receipts=[]
for idx,url in enumerate(urls):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Humanoid-research-reference-audit','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(req,timeout=20) as response:data=response.read(2*1024*1024)
        file=OUT/f'official_motion_metadata_{idx}.json';file.write_bytes(data)
        value=json.loads(data)
        summary=dict(commit=value.get('sha')) if isinstance(value,dict) else dict(files=[dict(name=x['name'],size=x['size'],sha=x['sha'],url=x['html_url']) for x in value])
        receipts.append(dict(url=url,fetched=True,file=str(file),sha256=hashlib.sha256(data).hexdigest(),summary=summary))
    except (urllib.error.URLError,TimeoutError) as error:
        receipts.append(dict(url=url,fetched=False,error=str(error)))
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),official_repository_metadata=receipts,
    primary_human=dict(url='https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/',pubmed='https://pubmed.ncbi.nlm.nih.gov/19640879/',
        indexed_primary_text_confirms_human_walking_speed_m_s=1.25,indexed_primary_text_subjects=10,
        shoulder_elbow_time_figure=6,direct_article_fetch='browser challenge',usable_numerical_curves_obtained=False,
        axes_signs_rest_poses_morphology_mapping_verified=False),
    usable_speed_matched_official_Unitree_locomotion_encoder_reference_obtained=False,
    no_remote_code_executed=True,no_reference_curve_fabricated=True,
    scope='Only official repository metadata was fetched. Dance or manipulation assets, simulator replay/conversion code and joint-index documents do not establish speed-matched measured locomotion encoder trajectories. Published human1.25m/s is closer to walking cases; it cannot certify3.5m/s robot arm proportions.')
(OUT/'updated_motion_reference_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
