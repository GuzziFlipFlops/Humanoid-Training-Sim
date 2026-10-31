"""Exercise the real chain loop with simulated workers, no GPU allocation."""
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import supervisor


class Continuation(unittest.TestCase):
    def test_explicit_gate_repair_preserves_every_existing_threshold(self):
        import ast,copy
        with tempfile.TemporaryDirectory() as folder:
            tree=ast.parse((supervisor.HERE/'evaluation.py').read_text())
            legacy=copy.deepcopy(tree)
            assignment=next(n for n in legacy.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GATE_VERSION' for t in n.targets))
            assignment.value=ast.Constant('physical_landing_v2')
            gate=next(n for n in legacy.body if isinstance(n,ast.FunctionDef) and n.name=='gate_case')
            branch=next(n for n in gate.body if isinstance(n,ast.If) and ast.unparse(n.test)=='speed == 0')
            branch.orelse=branch.orelse[:-2]
            old=Path(folder)/'v2.py';old.write_text(ast.unparse(legacy))
            new=Path(folder)/'v3.py';new.write_text(ast.unparse(tree))
            self.assertTrue(supervisor.verify_gate_upgrade(old,new)['existing_gate_logic_preserved'])
            new.write_text(new.read_text().replace('0.97','0.5'))
            with self.assertRaisesRegex(AssertionError,'Existing independent gate'):
                supervisor.verify_gate_upgrade(old,new)

    def test_reward_migration_does_not_silently_allow_verifier_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);prior=base/'prior'
            supervisor.Chain(prior,300,1,64,initial_checkpoint='placeholder.pt')
            candidate=base/'candidate.pt';candidate.touch()
            (prior/'state.json').write_text(json.dumps(dict(status='superseded_for_diagnostic_repair',child_pid=None,
                checkpoint=str(candidate),stage='walk_015',total_training_iterations=0,
                promotions=[dict(stage='stand',speed=0.,checkpoint=str(candidate))])))
            evaluator=prior/'source/physical_v29/training/evaluation_steps.py'
            evaluator.write_bytes(evaluator.read_bytes()+b'\n# Changed verifier\n')
            with self.assertRaisesRegex(AssertionError,'evaluation_steps'):
                supervisor.verify_reward_migration(prior,supervisor.HERE.parent)
    def test_reward_migration_keeps_candidate_and_requalifies_prior(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);prior=base/'prior'
            supervisor.Chain(prior,300,1,64,initial_checkpoint='placeholder.pt')
            candidate=base/'learned.pt';candidate.touch()
            (prior/'state.json').write_text(json.dumps(dict(status='superseded_for_diagnostic_repair',child_pid=None,
                checkpoint=str(candidate),stage='walk_015',total_training_iterations=548,prior_training_iterations=64,
                promotions=[dict(stage='stand',speed=0.,checkpoint='trusted_stand.pt')])) )
            # A reward edit is allowed; observations/action/limits/PPO are not.
            reward=prior/'source/physical_v29/training/walking_rewards.py'
            reward.write_bytes(reward.read_bytes()+b'\n# Prior reward version\n')
            chain=supervisor.Chain(base/'next',300,1,64,initial_checkpoint=str(candidate),train_envs=8192,reward_migrate_run=str(prior))
            called=[]
            chain.checked=lambda *a,**kw:(dict(completed=True,passed=True,checkpoint='smoke.pt'),base)
            def child(mode,n,speed=0.,checkpoint='',**kwargs):
                path=base/kwargs['label'];path.mkdir()
                if kwargs['label'].startswith('benchmark'):
                    (path/'progress.jsonl').write_text(json.dumps(dict(collect_seconds=1.,learn_seconds=.1,gpu={'free_mb':2000}))+'\n')
                    return dict(completed=True),{},path
                self.assertEqual(checkpoint,str(candidate));self.assertEqual(n,8192)
                return dict(completed=True,checkpoint='new_candidate.pt',iterations=64),{},path
            def gates(speed,checkpoint,label):
                called.append((speed,checkpoint,label))
                passed=checkpoint in ['trusted_stand.pt','new_candidate.pt']
                return [dict(passed=passed,cases=[dict(passed=passed,statistics={'command_m_s':0.})])]
            chain.child=child;chain.gates=gates
            with patch.object(supervisor,'STAGES',[('stand',0.),('walk_015',.15)]):chain.run()
            self.assertEqual(called[0],(0.,'trusted_stand.pt','reward_migration_prior'))
            self.assertEqual(called[1][1],str(candidate))
            self.assertEqual(chain.state['prior_training_iterations'],612)
            self.assertEqual(chain.state['total_training_iterations'],64)
            self.assertEqual(chain.state['retention_failures'],1)
            self.assertFalse(json.loads((base/'next/reward_migration.json').read_text())['qualification_reused'])

    def test_reward_migration_rejects_action_mapping_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);prior=base/'prior'
            supervisor.Chain(prior,300,1,64,initial_checkpoint='placeholder.pt')
            candidate=base/'candidate.pt';candidate.touch()
            (prior/'state.json').write_text(json.dumps(dict(status='superseded_for_diagnostic_repair',child_pid=None,
                checkpoint=str(candidate),stage='walk_015',total_training_iterations=0,
                promotions=[dict(stage='stand',speed=0.,checkpoint=str(candidate))])))
            env=prior/'source/physical_v29/training/environment.py'
            env.write_text(env.read_text().replace("self.actions=actions.clamp(-3.,3.).clone()","self.actions=actions.clamp(-2.,2.).clone()"))
            with self.assertRaisesRegex(AssertionError,'Non-reward environment'):
                supervisor.verify_reward_migration(prior,supervisor.HERE.parent)

    def test_resource_recycling_preserves_learning_and_finishes_same_block(self):
        with tempfile.TemporaryDirectory() as folder:
            chain=supervisor.Chain(Path(folder)/'run',300,1,64,
                initial_checkpoint='trusted_stand.pt',train_envs=8192)
            training=[];walk_gates=[]
            chain.checked=lambda *a,**kw:(dict(completed=True,passed=True,checkpoint='smoke.pt'),Path(folder))
            def child(mode,n,speed=0.,checkpoint='',**kwargs):
                path=Path(folder)/kwargs['label'];path.mkdir()
                if kwargs['label'].startswith('benchmark'):
                    (path/'progress.jsonl').write_text(json.dumps(dict(collect_seconds=1.,learn_seconds=.1,gpu={'free_mb':2000}))+'\n')
                    return dict(completed=True),{},path
                self.assertEqual(n,8192)
                self.assertEqual([p['stage'] for p in chain.state['promotions']],['stand'])
                training.append((checkpoint,kwargs['iterations']))
                if len(training)==1:
                    return dict(completed=True,checkpoint='recycled.pt',iterations=24,recycled=True,
                        recycle_reason='host_ram_below_2048MiB',gpu={'ram_available_mb':1000}),{},path
                return dict(completed=True,checkpoint='finished.pt',iterations=kwargs['iterations']),{},path
            def gates(speed,checkpoint,label):
                if speed==0:return [dict(passed=True,cases=[])]
                walk_gates.append(checkpoint)
                return [dict(passed=checkpoint=='finished.pt',cases=[dict(passed=True,statistics={'command_m_s':0.})])]
            chain.child=child;chain.gates=gates
            with patch.object(supervisor,'STAGES',[('stand',0.),('walk_015',.15)]):chain.run()
            self.assertEqual(training,[('trusted_stand.pt',64),('recycled.pt',40)])
            self.assertEqual(walk_gates,['trusted_stand.pt','finished.pt'])
            self.assertEqual(chain.state['total_training_iterations'],64)
            self.assertEqual(chain.state['resource_recycles'][0]['environments'],8192)
            self.assertEqual(chain.state['status'],'all_requested_stages_passed')

    def test_resource_migration_rejects_reward_or_ppo_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            prior=Path(folder)/'prior';source=prior/'source/physical_v29'
            supervisor.Chain(prior,300,1,64,initial_checkpoint='placeholder.pt')
            checkpoint=Path(folder)/'candidate.pt';checkpoint.touch()
            (prior/'state.json').write_text(json.dumps(dict(status='superseded_for_diagnostic_repair',child_pid=None,checkpoint=str(checkpoint))))
            self.assertEqual(supervisor.verify_resource_migration(prior,supervisor.HERE.parent)['checkpoint'],str(checkpoint))
            reward=source/'training/walking_rewards.py';original=reward.read_bytes();reward.write_bytes(original+b'\n# Changed reward\n')
            with self.assertRaisesRegex(AssertionError,'walking_rewards'):supervisor.verify_resource_migration(prior,supervisor.HERE.parent)
            reward.write_bytes(original)
            worker=source/'training/worker.py';worker.write_text(worker.read_text().replace('num_steps_per_env=24','num_steps_per_env=25'))
            with self.assertRaisesRegex(AssertionError,'PPO config'):supervisor.verify_resource_migration(prior,supervisor.HERE.parent)

    def test_failed_retention_preserves_learning_but_cannot_promote(self):
        with tempfile.TemporaryDirectory() as folder:
            chain = supervisor.Chain(Path(folder)/'run', 300, 1, 64,
                initial_checkpoint='trusted_stand.pt', train_envs=8192)
            starts = []
            walk_gates = []
            def checked(*args, **kwargs):
                return dict(completed=True, passed=True, checkpoint='smoke.pt'), Path(folder)
            def child(mode, n, speed=0., checkpoint='', **kwargs):
                path = Path(folder)/kwargs['label']; path.mkdir()
                if kwargs['label'].startswith('benchmark'):
                    (path/'progress.jsonl').write_text(json.dumps(dict(collect_seconds=1.,learn_seconds=.1,gpu={'free_mb':2000}))+'\n')
                    return dict(completed=True), {}, path
                starts.append(checkpoint)
                # No walking promotion before ALL independent cases pass.
                self.assertEqual([p['stage'] for p in chain.state['promotions']], ['stand'])
                self.assertEqual(chain.state['trusted_checkpoint'], 'trusted_stand.pt')
                return dict(completed=True, checkpoint=f'candidate_{len(starts)}.pt',iterations=64), {}, path
            def gates(speed, checkpoint, label):
                if speed==0:
                    return [dict(passed=True, cases=[dict(passed=True, statistics={'command_m_s':0.})])]
                walk_gates.append(checkpoint)
                passed = len(starts)>=2
                retention = len(starts)!=1
                return [dict(passed=passed, cases=[
                    dict(passed=retention, statistics={'command_m_s':0.}),
                    dict(passed=passed, statistics={'command_m_s':speed})])]
            chain.checked = checked
            chain.child = child
            chain.gates = gates
            with patch.object(supervisor, 'STAGES', [('stand',0.),('walk_015',.15)]):
                chain.run()
            self.assertEqual(starts, ['trusted_stand.pt','candidate_1.pt'])
            self.assertEqual(walk_gates, ['trusted_stand.pt','candidate_1.pt','candidate_2.pt'])
            self.assertEqual(chain.state['checkpoint'], 'candidate_2.pt')
            self.assertEqual(chain.state['retention_failures'], 1)
            self.assertEqual(chain.state['status'], 'all_requested_stages_passed')


if __name__=='__main__': unittest.main(verbosity=2)
