"""Read primary paper metadata; never synthesize reference trajectories."""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import requests
out=Path(__file__).resolve().parent
url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2817299/fullTextXML'
record=dict(primary_paper='Collins, Adamczyk and Kuo (2009), doi:10.1098/rspb.2009.0664',
    abstract_url='https://pubmed.ncbi.nlm.nih.gov/19640879/',api_url=url,
    mapped_reference_curve_generated=False,Unitree_speed_matched_encoders_obtained=False)
try:
    response=requests.get(url,timeout=25);response.raise_for_status()
    root=ET.fromstring(response.content)
    (out/'primary_arm_study.xml').write_bytes(response.content)
    record['source_sha256']=hashlib.sha256(response.content).hexdigest()
    paragraphs=[' '.join(''.join(p.itertext()).split()) for p in root.findall('.//p')]
    relevant=[p for p in paragraphs if '1.25' in p or ('speed' in p.lower() and 'walking' in p.lower())]
    record['speed_evidence_paragraphs']=relevant
    record['figure6_caption']=[' '.join(''.join(f.itertext()).split()) for f in root.findall('.//fig') if f.attrib.get('id')=='F6']
    record['retrieval_complete']=True
except Exception as exc:
    record.update(retrieval_complete=False,error=str(exc))
(out/'primary_motion_reference_review.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
