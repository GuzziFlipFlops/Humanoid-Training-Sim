"""CPU parity against installed Isaac quaternion helpers; no simulator imports."""
import ast, hashlib, json
from pathlib import Path
import torch
from unitree_reward_port import yaw_quat, quat_apply_inverse

HERE=Path(__file__).parent
source=HERE.parent.parent/'IsaacLab/source/isaaclab/isaaclab/utils/math.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
functions=[]
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['normalize','yaw_quat','quat_apply_inverse']:
        node.decorator_list=[]
        functions.append(node)
module=ast.Module(body=functions,type_ignores=[])
ast.fix_missing_locations(module)
scope={'torch':torch}
exec(compile(module,str(source),'exec'),scope)
torch.manual_seed(7010)
rows=[]
for shape in [(4096,), (8,17), ()]:
    q=torch.randn((*shape,4),dtype=torch.float64)
    q=q/q.norm(dim=-1,keepdim=True)
    v=torch.randn((*shape,3),dtype=torch.float64)
    y_error=float((yaw_quat(q)-scope['yaw_quat'](q)).abs().max())
    v_error=float((quat_apply_inverse(q,v)-scope['quat_apply_inverse'](q,v)).abs().max())
    assert y_error<1e-12,(shape,y_error)
    assert v_error<1e-12,(shape,v_error)
    rows.append(dict(shape=shape,yaw_max_abs_error=y_error,inverse_max_abs_error=v_error))
report=dict(passed=True,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            unit_quaternion_samples=4096+8*17+1,checks=rows,scope='CPU math parity; not behavioral qualification')
(HERE/'diagnostics/unitree_port_20261002/math_helper_parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
