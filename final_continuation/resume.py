"""Verify the bundle and run one explicit evaluation/training block."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p.write_text(json.dumps(value, indent=2))
    return str(p.resolve())

def prepare():
    for name, expected in read(ROOT/'manifest.json').items():
        assert sha(ROOT/name) == expected, f'Changed package file: {name}'
    runtime = ROOT/'runtime'
    runtime.mkdir(exist_ok=True)
    cert = read(ROOT/'config/qualified_prior_certificate.json')
    teacher = ROOT/'checkpoints/qualified_349_teacher.pt'
    assert sha(teacher) == cert['checkpoint_sha256']
    cert['checkpoint'] = str(teacher)
    for i, gate in enumerate(cert['gates']):
        for key in ('result','config'):
            original = ROOT/f'config/gates/{i}_{key}.json'
            assert sha(original) == gate[key+'_sha256']
            value = read(original)
            if key == 'config': value['arguments']['checkpoint'] = str(teacher)
            target = runtime/f'{i}_{key}.json'
            gate[key] = write(target,value)
            gate[key+'_sha256'] = sha(target)
    write(runtime/'relocated_certificate.json',cert)
    sys.path.insert(0,str(ROOT/'source/physical_v29/training'))
    from curriculum_provenance import training_reward_signature
    from standing_retention import validate_qualification
    signature = training_reward_signature(ROOT/'source/physical_v29/training')
    validate_qualification(cert,teacher,signature,cert['gate_version'])
    return runtime

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['verify','eval','train'],default='verify')
    p.add_argument('--num-envs',type=int,default=64)
    p.add_argument('--iterations',type=int,default=64)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); runtime=prepare()
    if a.mode=='verify':
        print('Package hashes and relocated teacher certificate pass.')
        sys.exit(0)
    assert a.output, '--output is required'
    cmd=[sys.executable,str(ROOT/'source/physical_v29/training/worker_unitree.py'),
         '--mode',a.mode,'--headless','--output',str(a.output.resolve()),
         '--checkpoint',str(ROOT/'checkpoints/final_1226.pt'),
         '--speed','3.5','--learning_rate','0.0001','--seed','477',
         '--num_envs',str(a.num_envs),'--iterations',str(a.iterations),
         '--termination_proof',str(ROOT/'config/termination_proof.json')]
    if a.mode=='train':
        standing=ROOT/'checkpoints/standing_teacher.pt'
        cmd += ['--standing_prior',str(standing),'--standing_prior_sha256',sha(standing),
                '--locomotion_prior_certificate',str(runtime/'relocated_certificate.json')]
    subprocess.run(cmd,check=True,cwd=ROOT)
