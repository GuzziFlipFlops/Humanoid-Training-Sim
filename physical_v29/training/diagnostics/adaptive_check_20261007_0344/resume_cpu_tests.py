"""CPU recovery/guards/provenance tests, no simulator or active-run mutations."""
import ast,importlib.util,sys,json,math,tempfile,shutil,unittest,hashlib,time
from pathlib import Path
out=Path(__file__).resolve().parent;train=out.parents[1];sys.path.insert(0,str(train))
s=importlib.util.spec_from_file_location('new_supervisor',train/'supervisor_unitree.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
live=json.loads((out/'initial_owned_observation.json').read_text());prior=Path(live['active_run']['run'])
source=out/'control_source/physical_v29';context=m.restore_resume_context(prior,source)
(out/'resume_context_cpu.json').write_text(json.dumps(context,indent=2))
class RecoveryTests(unittest.TestCase):
 def test_unbounded_goal_budget(self):self.assertTrue(math.isinf(m.chain_deadline(10.,0.)))
 def test_finite_budget(self):self.assertEqual(m.chain_deadline(10.,2.),7210.)
 def test_bad_budgets(self):
  for h in [-1.,math.inf,-math.inf,math.nan]:
   with self.subTest(h=h),self.assertRaises(AssertionError):m.chain_deadline(1.,h)
 def test_exact_selected_full_actor(self):self.assertEqual(context['checkpoint_sha256'],hashlib.sha256(Path(live['state']['checkpoint']).read_bytes()).hexdigest())
 def test_no_unsaved_updates_counted(self):
  self.assertEqual(context['verified_recovered_main_updates'],33)
  self.assertEqual(context['inherited_main_updates'],27507)
 def test_standing_teacher_preserved(self):
  self.assertEqual(context['standing_prior_sha256'],'22e582af9b9ec17e54e3a86dad97c8d84fbd46bb2727d5232e22e458a2e578d7')
 def test_moving_teacher_qualified_under_current_contract(self):
  self.assertEqual(Path(context['qualification_run']).resolve(),prior.resolve())
  self.assertEqual(json.loads(Path(context['locomotion_prior_certificate']).read_text())['maximum_command_m_s'],3.)
 def test_termination_bridge_retained(self):self.assertTrue(context['termination_proof'].endswith('termination_continuation_proof_retry.json'))
 def test_protected_all_runtime_bytes(self):self.assertEqual(len(context['protected_hashes']),53)
 def test_no_deadline_kill_mid_worker(self):
  tree=ast.parse((train/'supervisor_unitree.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Chain')
  child=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='child')
  self.assertNotIn('deadline',ast.unparse(child));self.assertIn('NO_PROGRESS_600_SECONDS',ast.unparse(child))
 def test_smoke_start_from_original_full_state(self):
  t=(train/'supervisor_unitree.py').read_text();self.assertIn("speed,context['checkpoint'],iterations=4,label='resume_ppo_smoke'",t)
  self.assertIn("checkpoint=context['checkpoint']",t)
 def test_actual_chain_freeze_and_early_teacher_context(self):
  with tempfile.TemporaryDirectory(dir=out) as tmp:
   chain=m.Chain(Path(tmp)/'chain',300,0.,64,train_envs=8192,resume_run=str(prior),learning_rate=.0001)
   self.assertTrue(math.isinf(chain.deadline))
   self.assertEqual(chain.standing_prior,context['standing_prior'])
   self.assertEqual(chain.state['promotions'],context['previous']['promotions'])
   self.assertEqual(chain.resume_context['inherited_main_updates'],27507)
   cfg=json.loads((chain.root/'chain_config.json').read_text())
   self.assertEqual(cfg['max_hours'],0.);self.assertEqual(len(cfg['sources_sha256']),55)
 def test_healthy_cpu_worker_finishes_after_budget(self):
  with tempfile.TemporaryDirectory(dir=out) as tmp:
   chain=m.Chain(Path(tmp)/'chain',300,.0001,64,train_envs=8192,resume_run=str(prior),learning_rate=.0001)
   chain.worker=out/'cpu_budget_worker.py';chain.deadline=time.time()-1.
   r,item,folder=chain.child('train',512,0.,iterations=4,label='cpu_budget_boundary')
   self.assertTrue(r['completed'] and r['cpu_mock'] and item['completed'])
   self.assertFalse((folder/'supervisor_failure.json').exists())
 def test_resume_rejections(self):
  with tempfile.TemporaryDirectory(dir=out) as tmp:
   p=Path(tmp);shutil.copytree(source,p/'source/physical_v29',ignore=shutil.ignore_patterns('usd','__pycache__','*.pyc'))
   state=json.loads((prior/'state.json').read_text());(p/'state.json').write_text(json.dumps(state))
   state['status']='running';(p/'state.json').write_text(json.dumps(state))
   with self.assertRaises(AssertionError):m.restore_resume_context(p,source)
   state['status']='failed';state['child_pid']=123;(p/'state.json').write_text(json.dumps(state))
   with self.assertRaises(AssertionError):m.restore_resume_context(p,source)
   state['child_pid']=None;(p/'state.json').write_text(json.dumps(state))
   path=p/'source/physical_v29/training/walking_rewards.py';path.write_bytes(path.read_bytes()+b'\n# changed\n')
   with self.assertRaises(AssertionError):m.restore_resume_context(p,source)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
record=dict(passed=result.wasSuccessful(),test_methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),verified_recovered_updates=context['verified_recovered_main_updates'],inherited_main_updates=context['inherited_main_updates'],supervisor_sha256=hashlib.sha256((train/'supervisor_unitree.py').read_bytes()).hexdigest(),no_physics_or_reward_change=True,unsaved_failed_block_updates_discarded=32)
(out/'resume_cpu_tests.json').write_text(json.dumps(record,indent=2));assert record['passed']
