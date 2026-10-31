"""Physics-settle candidate lying starts; only retain independently audited poses."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from isaaclab.app import AppLauncher
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'source'))
parser = argparse.ArgumentParser()
parser.add_argument('--per_group', type=int, default=256)
parser.add_argument('--batch', type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
import torch
from isaaclab.envs import ManagerBasedRLEnv
from candidate import make_cfg
from telemetry import Telemetry
from geometry import Geometry


def main():
    dest = HERE / 'assets/pose_bank.pt'
    if dest.exists(): raise FileExistsError('Do not overwrite an audited pose bank')
    cfg = make_cfg(args.batch, probe=True)
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    robot, geom = env.scene['robot'], Geometry()
    telemetry = Telemetry(env)
    rng = np.random.default_rng(20260903)
    saved_q, saved_root, groups, splits = [], [], [], []
    audit = dict(groups={}, enabled_self_collisions=True, filtered_ground_sensors=len(geom.links))
    try:
        f = telemetry.measure()
        print('[POSE] Ground sensor shape OK; nominal floor/penetration', telemetry.last_floor.tolist()[:3],
              telemetry.last_penetration.tolist()[:3], flush=True)
        for group in range(5):
            accepted = 0
            attempts = 0
            rejected = 0
            while accepted < args.per_group and attempts < args.per_group * 6:
                qrows, roots = [], []
                while len(qrows) < args.batch:
                    if group == 4:
                        q = geom.nominal.copy()
                        q = {j['name']: float(np.clip(q[j['name']] + rng.uniform(-.025, .025),
                                    j['lower'] + .02, j['upper'] - .02)) for j in geom.joints}
                        rotation = np.eye(3)
                        z = -geom.floor_z(geom.fk(q)) + .003
                    else:
                        q, rotation, z = geom.pose(group, rng, jitter=.075)
                    if geom.penetrations(geom.fk(q, rotation)): continue
                    xyzw = Rotation.from_matrix(rotation).as_quat()
                    roots.append([0., 0., z, xyzw[3], *xyzw[:3], 0., 0., 0., 0., 0., 0.])
                    qrows.append([q[n] for n in robot.joint_names])
                q = torch.tensor(qrows, device=env.device)
                root = torch.tensor(roots, device=env.device)
                root[:, :3] += env.scene.env_origins
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                robot.reset()
                robot.set_joint_position_target(q)
                max_pen = torch.zeros(args.batch, device=env.device)
                min_floor = torch.ones(args.batch, device=env.device)
                for step in range(400):  # 2 s at 200 Hz, no external support or gravity removal
                    robot.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    if step % 20 == 0:
                        depth, floor = telemetry.geometry_check()
                        max_pen = torch.maximum(max_pen, depth)
                        min_floor = torch.minimum(min_floor, floor)
                f = telemetry.measure()
                support_ratio = telemetry.ground[:, :, 2].sum(1) / (telemetry.total_mass * 9.81)
                if not (support_ratio > .5).any():
                    raise RuntimeError('Ground force filter has no measured load; do not label poses validated')
                valid = f.safe & (max_pen < .015) & (min_floor > -.025)
                valid &= (support_ratio > .6) & (support_ratio < 1.4)
                valid &= robot.data.root_lin_vel_w.norm(dim=1) < .15
                valid &= robot.data.root_ang_vel_w.norm(dim=1) < .4
                valid &= robot.data.joint_vel.square().mean(1).sqrt() < .5
                # Label from the settled torso orientation, not the requested pose.
                from isaaclab.utils.math import matrix_from_quat
                rot = matrix_from_quat(robot.data.body_link_quat_w[:, telemetry.torso])
                dirs = torch.stack((rot[:, 2, 0], -rot[:, 2, 0], rot[:, 2, 1], -rot[:, 2, 1]), 1)
                actual_group = dirs.argmax(1)
                if group < 4:
                    valid &= (f.height < .45) & (f.torso_tilt > 1.05) & (actual_group == group)
                else:
                    valid &= (f.height > .76) & (f.torso_tilt < .25) & (f.foot_load > .1).all(1)
                ids = valid.nonzero().flatten()[:args.per_group - accepted]
                qcpu = robot.data.joint_pos[ids].cpu()
                rcpu = robot.data.root_state_w[ids].clone()
                rcpu[:, :3] -= env.scene.env_origins[ids]
                rcpu[:, 7:] = 0.
                for i in range(len(ids)):
                    saved_q.append(qcpu[i]); saved_root.append(rcpu[i].cpu())
                    groups.append(group)
                    splits.append(0 if accepted % 2 == 0 else 1)
                    accepted += 1
                attempts += args.batch
                rejected += args.batch - int(valid.sum())
                print(f'[POSE] group {group} accepted {accepted}/{args.per_group}, attempted {attempts}, rejected {rejected}', flush=True)
            audit['groups'][group] = dict(accepted=accepted, attempted=attempts, rejected=rejected)
            if accepted < args.per_group:
                raise RuntimeError(f'Pose family {group} did not settle reliably: {audit["groups"][group]}')
        bank = dict(validated=True, joint_names=robot.joint_names, q=torch.stack(saved_q), root=torch.stack(saved_root),
                    group=torch.tensor(groups), split=torch.tensor(splits),
                    asset_sha256=hashlib.sha256((HERE / 'assets/recovery.usda').read_bytes()).hexdigest())
        torch.save(bank, dest)
        audit['bank_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        audit['pose_count'] = len(groups)
        audit['settling_seconds'] = 2.
        audit['mass_kg'] = float(telemetry.total_mass[0])
        (HERE / 'assets/pose_audit.json').write_text(json.dumps(audit, indent=2))
        print('[POSE] COMPLETE ' + json.dumps(audit), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try: main()
    finally: app.app.close()
