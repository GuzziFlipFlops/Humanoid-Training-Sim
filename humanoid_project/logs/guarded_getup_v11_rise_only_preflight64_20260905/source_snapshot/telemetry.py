"""Measured geometry/support telemetry; no success defaults for missing sensors."""
import torch
from isaaclab.utils.math import quat_apply, matrix_from_quat
from geometry import Geometry
from reward_core import Frame


def support_margin(points, com):
    """Signed distance to convex hull of eight sole corners, batched on GPU.

    Every directed pair with all vertices on its left is a hull support line.
    Taking the minimum signed distance over these lines is exact for the hull;
    collinear edges are harmless. This is not a bounding-box approximation.
    """
    n, k, _ = points.shape
    i, j = torch.where(~torch.eye(k, dtype=torch.bool, device=points.device))
    p = points[:, i]
    edge = points[:, j] - p
    length = edge.norm(dim=-1)
    rel = points[:, None] - p[:, :, None]
    cross = edge[:, :, None, 0] * rel[..., 1] - edge[:, :, None, 1] * rel[..., 0]
    hull_edge = (cross >= -1e-7).all(-1) & (length > 1e-5)
    delta = com[:, None] - p
    distance = (edge[..., 0] * delta[..., 1] - edge[..., 1] * delta[..., 0]) / length.clamp_min(1e-6)
    return torch.where(hull_edge, distance, torch.inf).amin(1)


class Telemetry:
    def __init__(self, env):
        self.env, self.robot = env, env.scene['robot']
        self.geom = Geometry()
        self.names = self.robot.body_names
        self.feet = [self.names.index(n) for n in ('left_foot', 'right_foot')]
        self.torso = self.names.index('torso')
        self.head = self.names.index('head')
        self.nonfeet = [i for i, n in enumerate(self.names) if n not in ('left_foot', 'right_foot')]
        self.centers = torch.tensor([self.geom.links[n]['center'] for n in self.names], device=env.device, dtype=torch.float)
        self.half = torch.tensor([self.geom.links[n]['half'] for n in self.names], device=env.device, dtype=torch.float)
        assert all(abs(self.geom.links[n]['rotation'] - torch.eye(3).numpy()).max() < 1e-7 for n in self.names)
        self.pair_a = torch.tensor([self.names.index(a) for a, b in self.geom.pairs], device=env.device)
        self.pair_b = torch.tensor([self.names.index(b) for a, b in self.geom.pairs], device=env.device)
        self.masses = self.robot.root_physx_view.get_masses().to(env.device)
        self.total_mass = self.masses.sum(1)
        self.last_penetration = torch.zeros(env.num_envs, device=env.device)
        self.last_floor = torch.zeros_like(self.last_penetration)
        self.sample = None

    def geometry_check(self):
        d = self.robot.data
        rot = matrix_from_quat(d.body_link_quat_w)
        center = d.body_link_pos_w + (rot @ self.centers[None, :, :, None]).squeeze(-1)
        radius = (rot.abs() @ self.half[None, :, :, None]).squeeze(-1)
        self.last_floor = (center[..., 2] - radius[..., 2]).amin(1) - self.env.scene.env_origins[:, 2]
        a, b = self.pair_a, self.pair_b
        delta = center[:, b] - center[:, a]
        broad = (radius[:, a] + radius[:, b] - delta.abs()).amin(-1) > .001
        rows, pairs = broad.nonzero(as_tuple=True)
        depths = torch.zeros(self.env.num_envs, device=self.env.device)
        if rows.numel():
            ia, ib = a[pairs], b[pairs]
            ra, rb = rot[rows, ia], rot[rows, ib]
            axes_a, axes_b = ra.transpose(-1, -2), rb.transpose(-1, -2)
            cross = torch.cross(axes_a[:, :, None].expand(-1, 3, 3, 3),
                                axes_b[:, None].expand(-1, 3, 3, 3), dim=-1).flatten(1, 2)
            axes = torch.cat((axes_a, axes_b, cross), 1)
            length = axes.norm(dim=-1)
            axes = axes / length.clamp_min(1e-8)[..., None]
            extent_a = (axes @ ra).abs() @ self.half[ia, :, None]
            extent_b = (axes @ rb).abs() @ self.half[ib, :, None]
            sep = (axes * (center[rows, ib] - center[rows, ia])[:, None]).sum(-1).abs()
            depth = torch.where(length > 1e-7, extent_a.squeeze(-1) + extent_b.squeeze(-1) - sep, torch.inf).amin(-1)
            depths.scatter_reduce_(0, rows, depth.clamp_min(0.), reduce='amax', include_self=True)
        self.last_penetration = depths
        return depths, self.last_floor

    def measure(self, full_geometry=True):
        env, d = self.env, self.robot.data
        if full_geometry: self.geometry_check()
        # One sensor per body, explicitly filtered against ground. Self-contact
        # must not be interpreted as body support or upward foot-ground force.
        forces = []
        for name in self.names:
            data = env.scene.sensors['ground_' + name].data.force_matrix_w
            if data is None or data.shape[2] != 1:
                raise RuntimeError(f'Missing/ambiguous ground filter: {name}')
            forces.append(data[:, 0, 0])
        ground = torch.stack(forces, 1)
        self.ground = ground
        rot = matrix_from_quat(d.body_link_quat_w)
        pelvis_tilt = torch.acos((1. - 2. * (d.root_quat_w[:, 1].square() + d.root_quat_w[:, 2].square())).clamp(-1., 1.))
        torso_tilt = torch.acos(rot[:, self.torso, 2, 2].clamp(-1., 1.))
        fq = d.body_link_quat_w[:, self.feet]
        sole_offset = self.centers[self.feet].clone()
        sole_offset[:, 2] -= self.half[self.feet, 2]
        offset = quat_apply(fq, sole_offset[None].expand(env.num_envs, -1, -1))
        soles = d.body_link_pos_w[:, self.feet] + offset
        velocities = d.body_link_lin_vel_w[:, self.feet] + torch.cross(d.body_link_ang_vel_w[:, self.feet], offset, dim=-1)
        corners = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]], device=env.device)
        local = corners[None] * self.half[self.feet, None] + sole_offset[:, None]
        all_corners = d.body_link_pos_w[:, self.feet, None] + quat_apply(fq[:, :, None].expand(-1, -1, 4, -1),
                                                           local[None].expand(env.num_envs, -1, -1, -1))
        com = (d.body_com_pos_w * self.masses[..., None]).sum(1) / self.total_mass[:, None]
        margin = support_margin(all_corners[..., :2].flatten(1, 2), com[:, :2])
        self.com = com
        yaw = torch.atan2(2 * (d.root_quat_w[:, 0] * d.root_quat_w[:, 3] + d.root_quat_w[:, 1] * d.root_quat_w[:, 2]),
                          1. - 2 * (d.root_quat_w[:, 2].square() + d.root_quat_w[:, 3].square()))
        safe = (self.last_penetration < .015) & (self.last_floor > -.025)
        safe &= (d.root_lin_vel_w.norm(dim=1) < 8.) & (d.root_ang_vel_w.norm(dim=1) < 20.)
        safe &= ground[:, self.head].norm(dim=1) < self.total_mass * 9.81 * 2.
        foot_load = ground[:, self.feet, 2].clamp_min(0.) / (self.total_mass * 9.81)[:, None]
        # A support polygon is meaningful only with BOTH feet actually loaded.
        margin = torch.where((foot_load >= .1).all(1), margin, -.1)
        self.sample = Frame(d.root_pos_w[:, 2] - env.scene.env_origins[:, 2], pelvis_tilt, torso_tilt,
            d.root_pos_w[:, :2], yaw, d.root_lin_vel_w[:, :2].norm(dim=1), d.root_lin_vel_w[:, 2],
            d.root_ang_vel_w.norm(dim=1), d.joint_vel.square().mean(1).sqrt(), foot_load,
            torch.acos(rot[:, self.feet, 2, 2].clamp(-1., 1.)), velocities[..., :2].norm(dim=2),
            soles[..., :2], ground[:, self.nonfeet].norm(dim=2).amax(1), margin,
            torch.ones(env.num_envs, dtype=torch.bool, device=env.device), safe)
        return self.sample
