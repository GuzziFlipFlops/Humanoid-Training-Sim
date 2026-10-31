"""Diagnostic wrapper: observe first genuine PPO/prior gradient, do not alter it."""
import json, runpy, sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'diagnostic_pause.json').read_text());RUN=Path(d['run'])
TRAIN=RUN/'source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
import torch
from standing_retention import StandingRetention
captured=False
original=StandingRetention._regularize
def audited(self,module,inputs,mean):
    global captured
    observations=self._observations
    if self.command_range is None or captured or observations is None or not mean.requires_grad:
        return original(self,module,inputs,mean)
    raw=observations['policy'];index=raw.shape[-1]-80+6
    speed=raw[:,index:index+2].norm(dim=1)
    ids=((speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)).nonzero().flatten()
    if not ids.numel():return original(self,module,inputs,mean)
    stride=max(1,(ids.numel()+self.max_samples-1)//self.max_samples);ids=ids[::stride][:self.max_samples]
    with torch.no_grad():
        target=self.teacher(observations[ids]).clamp(-3,3)
        difference=mean[ids].detach()-target
        derivative=2*self.weight*difference/difference.numel()
    saved={'observations':raw.detach().cpu().clone(),'mean':mean.detach().cpu().clone(),
           'ids':ids.cpu(),'prior_derivative':derivative.cpu(),
           'mlp_state':{k:v.detach().cpu().clone() for k,v in module.state_dict().items()},
           'latest_commands':raw[:,index:index+3].detach().cpu().clone()}
    def capture(gradient):
        saved['incoming_gradient']=gradient.detach().cpu().clone()
        torch.save(saved,OUT/'gradient_batch.pt')
        return gradient
    mean.register_hook(capture)
    captured=True
    # Original hook adds its original exact derivative after our passive capture.
    return original(self,module,inputs,mean)
StandingRetention._regularize=audited
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
