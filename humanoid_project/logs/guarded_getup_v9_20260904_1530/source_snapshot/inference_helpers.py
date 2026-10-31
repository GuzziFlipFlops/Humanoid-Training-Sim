"""Simulator buffers created during inference must also be mutated in inference."""
import torch


@torch.inference_mode()
def push_laterally(robot, speed_delta):
    velocity = robot.data.root_vel_w.clone()
    velocity[:, 1] += speed_delta
    robot.write_root_velocity_to_sim(velocity)
