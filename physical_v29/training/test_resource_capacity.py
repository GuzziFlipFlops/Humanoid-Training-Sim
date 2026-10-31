"""Replay the observed 8k host-pressure failure without starting Isaac/CUDA."""
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import supervisor_unitree as supervisor
from vram_ramp import VramRamp, estimate_increment

R28 = Path(__file__).parent / 'runs/locomotion_20261004_r28_heading_alignment_cache_fixed'


class CapacityReplay(unittest.TestCase):
    def test_measured_8k_recovery_is_not_sustained_capacity(self):
        measured = json.loads((R28 / 'benchmarks.json').read_text())
        eligible = [b for b in measured if supervisor.benchmark_capacity_eligible(b, 300)]
        self.assertEqual([b['environments'] for b in eligible], [4096, 6144])
        self.assertEqual(supervisor.select_environment_count(eligible, 8192), 6144)
        trial = next(b for b in measured if b['environments'] == 8192)
        self.assertGreaterEqual(trial['minimum_free_mb'], 300)
        self.assertTrue(trial['host_ram_recycled'])
        self.assertLess(trial['minimum_host_available_mb'], 2048)

    def test_each_memory_and_completion_guard_is_required(self):
        good = dict(environments=8192, minimum_free_mb=300,
                    minimum_host_available_mb=2048, updates=24, host_ram_recycled=False)
        self.assertTrue(supervisor.benchmark_capacity_eligible(good, 300))
        for change in [dict(minimum_free_mb=299), dict(minimum_host_available_mb=2047),
                       dict(updates=23), dict(host_ram_recycled=True)]:
            with self.subTest(change=change):
                self.assertFalse(supervisor.benchmark_capacity_eligible(dict(good, **change), 300))

    def test_saved_recovery_lowers_count_and_resets_hold_without_gpu_cost_inflation(self):
        result = json.loads((R28 / '008_benchmark_8192/result.json').read_text())
        ramp = VramRamp()
        for now in range(0, 181, 3):
            ramp.observe(now, 6144, 3000, True, 8000)
        self.assertEqual(ramp.ready, 8192)
        self.assertEqual(supervisor.fallback_after_recycle(8192, result, ramp, 181), 6144)
        self.assertIsNone(ramp.ready)
        self.assertEqual(ramp.mib_per_env, .5)
        self.assertEqual(ramp.cooldown_until, 361)
        for now in range(183, 361, 3):
            ramp.observe(now, 6144, 3000, True, 8000)
        self.assertIsNone(ramp.since)
        for now in range(363, 544, 3):
            ramp.observe(now, 6144, 3000, True, 8000)
        self.assertEqual(ramp.ready, 8192)

    def test_lowest_count_or_missing_checkpoint_cannot_loop_forever(self):
        result = dict(recycled=True, checkpoint='preserved.pt', recycle_reason='host_ram_below_2048MiB')
        with self.assertRaises(RuntimeError):
            supervisor.fallback_after_recycle(512, result, VramRamp(), 0)
        with self.assertRaises(AssertionError):
            supervisor.fallback_after_recycle(8192, dict(result, checkpoint=''), VramRamp(), 0)


class SupervisorContinuation(unittest.TestCase):
    def run_recycle(self, recovered_updates):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = supervisor.Chain.__new__(supervisor.Chain)
            chain.root = root
            chain.validated_checks = chain.migrate_run = chain.reward_migrate_run = chain.resume_run = None
            chain.optimization_proof = chain.curriculum_proof = ''
            # This is a scheduler fixture with mocked workers, not permission
            # to reuse historical capacity after the authored runtime changes.
            prior = root / 'matching-capacity-fixture'
            shutil.copytree(R28 / 'source/physical_v29', prior / 'source/physical_v29',
                            ignore=shutil.ignore_patterns('__pycache__'))
            for name in ['environment_unitree.py','worker_unitree.py','unitree_reward_port.py',
                         'environment.py','worker.py','evaluation.py','evaluation_steps.py',
                         'walking_rewards.py','reference_geometry.py','mass_estimate.json',
                         'assets/robot_training.urdf','assets/motion_reference.npz']:
                shutil.copy2(Path(__file__).parent / name, prior / 'source/physical_v29/training' / name)
            shutil.copy2(R28 / 'benchmarks.json', prior / 'benchmarks.json')
            chain.prior_run = prior
            chain.actor_warm_start = False
            chain.initial_checkpoint = ''
            chain.train_envs = 8192
            chain.reserve = 300
            chain.ramp = VramRamp()
            chain.step_seconds = chain.step_count = None
            chain.chunk = 64
            chain.deadline = time.time() + 60
            chain.standing_prior = chain.standing_prior_sha256 = None
            chain.state = dict(promotions=[], total_training_iterations=0)
            smoke = root / 'smoke.pt'
            smoke.write_bytes(b'isolated scheduler fixture; not a real policy')
            recovery = root / 'saved-full-optimizer.pt'
            recovery.write_bytes(b'isolated saved-optimizer fixture')
            calls = []
            completed = 0

            def checked(*args, **kwargs):
                return dict(completed=True, passed=True, checkpoint=str(smoke)), root

            def child(mode, count, speed, checkpoint, **kwargs):
                nonlocal completed
                calls.append(dict(count=count, checkpoint=checkpoint, iterations=kwargs['iterations']))
                folder = root / f'train_{len(calls)}'
                folder.mkdir()
                updates = recovered_updates if len(calls) == 1 else kwargs['iterations']
                completed += updates
                result = dict(completed=True, checkpoint=str(recovery), iterations=updates,
                              recycled=len(calls) == 1, recycle_reason='host_ram_below_2048MiB', gpu={})
                return result, dict(memory_failure=False), folder

            def gates(*args):
                return [dict(passed=completed >= 64, cases=[])]

            chain.checked, chain.child, chain.gates = checked, child, gates
            with patch.object(supervisor, 'STAGES', [('stand', 0.)]), \
                    patch.object(supervisor, 'HERE', prior / 'source/physical_v29/training'):
                chain.run()
            self.assertEqual(chain.state['total_training_iterations'], 64)
            self.assertEqual(chain.state['checkpoint'], str(recovery))
            self.assertEqual(chain.state['next_training_environment_count'], 4096)
            self.assertEqual(chain.state['last_memory_failure_resource'], 'host')
            measured = json.loads((R28 / 'benchmarks.json').read_text())
            self.assertEqual(chain.ramp.mib_per_env, estimate_increment(measured))
            self.assertEqual(calls[0]['count'], 6144)
            return calls

    def test_resume_remaining_updates_at_lower_count_from_saved_full_checkpoint(self):
        calls = self.run_recycle(5)
        self.assertEqual([c['iterations'] for c in calls], [64, 59])
        self.assertEqual(calls[1]['count'], 4096)
        self.assertEqual(Path(calls[1]['checkpoint']).name, 'saved-full-optimizer.pt')

    def test_recycle_on_final_update_also_lowers_next_count(self):
        self.assertEqual(len(self.run_recycle(64)), 1)


class StrictResourceMigration(unittest.TestCase):
    def test_migration_smoke_and_training_use_full_checkpoint_without_actor_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / 'prior'
            prior.mkdir()
            new_run = root / 'new'
            new_run.mkdir()
            shutil.copytree(R28 / 'source/physical_v29', prior / 'source/physical_v29',
                            ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(prior / 'source/physical_v29', new_run / 'source/physical_v29')
            shutil.copy2(R28 / 'benchmarks.json', prior / 'benchmarks.json')
            preflight = prior / '001_physics_preflight'
            preflight.mkdir()
            shutil.copy2(R28 / '001_physics_preflight/result.json', preflight / 'result.json')
            checkpoint = root / 'full-optimizer.pt'
            checkpoint.write_bytes(b'isolated continuation fixture')
            prior_state = dict(status='superseded_for_diagnostic_repair', child_pid=None,
                               checkpoint=str(checkpoint), promotions=[], stage='stand',
                               stage_speed=0., total_training_iterations=17)
            (prior / 'state.json').write_text(json.dumps(prior_state))
            chain = supervisor.Chain.__new__(supervisor.Chain)
            chain.optimization_proof = chain.curriculum_proof = ''
            chain.root = new_run
            chain.migrate_run = chain.prior_run = prior
            chain.validated_checks = chain.reward_migrate_run = chain.resume_run = None
            chain.actor_warm_start = True
            chain.initial_checkpoint = str(checkpoint)
            chain.train_envs = 8192
            chain.reserve = 300
            chain.ramp = VramRamp()
            chain.step_seconds = chain.step_count = None
            chain.chunk = 64
            chain.deadline = time.time() + 60
            chain.standing_prior = chain.standing_prior_sha256 = None
            chain.state = dict(promotions=[], total_training_iterations=0)
            calls = []

            def checked(*args, **kwargs):
                calls.append(dict(mode=args[0], actor_warm_start=chain.actor_warm_start,
                                  arguments=args, kwargs=kwargs))
                return dict(completed=True, checkpoint=str(checkpoint)), new_run

            chain.checked = checked
            chain.gates = lambda speed, candidate, label: [dict(passed=True, cases=[])]
            with patch.object(supervisor, 'STAGES', [('stand', 0.)]), \
                    patch.object(supervisor, 'HERE', prior / 'source/physical_v29/training'):
                chain.run()
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(not c['actor_warm_start'] for c in calls))
            self.assertEqual(calls[0]['arguments'][3], str(checkpoint))
            self.assertEqual(calls[0]['kwargs']['iterations'], 4)
            self.assertEqual(chain.state['checkpoint'], str(checkpoint))
            self.assertEqual(chain.state['prior_training_iterations'], 17)
            self.assertFalse(chain.actor_warm_start)
            self.assertEqual(chain.state['next_training_environment_count'], 6144)

    def test_every_runtime_byte_and_inventory_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / 'prior'
            prior.mkdir()
            old = prior / 'source/physical_v29'
            new = root / 'new/physical_v29'
            shutil.copytree(R28 / 'source/physical_v29', old, ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(old, new, ignore=shutil.ignore_patterns('usd'))
            checkpoint = root / 'saved.pt'
            checkpoint.write_bytes(b'provenance fixture only')
            (prior / 'state.json').write_text(json.dumps(dict(
                status='superseded_for_diagnostic_repair', child_pid=None, checkpoint=str(checkpoint))))
            supervisor.verify_resource_migration(prior, new)
            for name in ['training/stage_rewards.py', 'training/environment_unitree.py',
                         'training/evaluation.py', 'cad_differential.py', 'training/worker_unitree.py']:
                p = new / name
                original = p.read_bytes()
                p.write_bytes(original + b'\n# controlled mismatch\n')
                with self.subTest(name=name), self.assertRaisesRegex(AssertionError, 'Runtime source changed'):
                    supervisor.verify_resource_migration(prior, new)
                p.write_bytes(original)
            extra = new / 'extra.py'
            extra.write_text('pass\n')
            with self.assertRaisesRegex(AssertionError, 'inventory changed'):
                supervisor.verify_resource_migration(prior, new)


if __name__ == '__main__':
    unittest.main(verbosity=2)
