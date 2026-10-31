"""Diagnostic wrapper; original reward signature does not certify the new loss."""
import builtins,json,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(OUT),str(TRAIN),str(TRAIN.parent)]
from sequence_shoulder_refinement import SequenceShoulderRefinement
from upper_body_refinement import UpperBodyRefinement
from rsl_rl.algorithms import PPO
apply='--apply-sequence-mean' in sys.argv
if apply:sys.argv.remove('--apply-sequence-mean')
ENV=None;installed=False;original_import=builtins.__import__
def after_import(name,*args,**kwargs):
    global installed
    result=original_import(name,*args,**kwargs)
    if name=='environment_unitree' and not installed:
        installed=True;cls=sys.modules[name].LocomotionEnv;old_init=cls.__init__
        def initialize_env(self,*a,**k):
            global ENV
            old_init(self,*a,**k);ENV=self
        cls.__init__=initialize_env;builtins.__import__=original_import
    return result
builtins.__import__=after_import
old_init=UpperBodyRefinement.__init__;old_read=UpperBodyRefinement.read_audit;old_close=UpperBodyRefinement.close
def initialize(self,actor,names,*a,**k):
    old_init(self,actor,names,*a,**k)
    self.experimental=SequenceShoulderRefinement(actor,names,apply,lambda:ENV)
def read(self):
    result=old_read(self);result['experimental_sequence_mean']=self.experimental.read_audit()
    seq,cycles=self.experimental.cache.batch()
    if seq is not None:
        import torch
        torch.save(dict(observations=seq,commands=[c['command'] for c in cycles],env_ids=[c['env_id'] for c in cycles],
            physical_targets=[c['physical_targets'] for c in cycles],start_ticks=[c['start_tick'] for c in cycles],
            end_ticks=[c['end_tick'] for c in cycles],truthful=True),Path(sys.argv[sys.argv.index('--output')+1])/'actual_collection_cycles.pt')
    return result
def close(self):self.experimental.close();old_close(self)
UpperBodyRefinement.__init__=initialize;UpperBodyRefinement.read_audit=read;UpperBodyRefinement.close=close
old_act=PPO.act;old_process=PPO.process_env_step
def act(self,observations):
    result=old_act(self,observations)
    helper=getattr(self.actor,'_sequence_shoulder_diagnostic',None)
    if helper is not None:helper.record(observations)
    return result
def process(self,observations,rewards,dones,extras):
    result=old_process(self,observations,rewards,dones,extras)
    helper=getattr(self.actor,'_sequence_shoulder_diagnostic',None)
    if helper is not None:helper.cache.reset(dones)
    return result
PPO.act=act;PPO.process_env_step=process
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
