"""Check the added training gradient and unchanged deployed actor interface."""
import copy,unittest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from standing_retention import StandingRetention,validate_teacher


def actor(obs):
    return MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[16],
        activation='elu',obs_normalization=True,
        distribution_cfg=dict(class_name='GaussianDistribution',init_std=.3,std_type='log'))


class Retention(unittest.TestCase):
    def test_qualified_legacy_metadata_keeps_mandatory_contract_and_asset(self):
        expected=dict(asset_sha256='asset',motion_sha256='motion',actuator_sha256='actuator')
        self.assertEqual(validate_teacher(dict(contract='23_axis',asset_sha256='asset'),'23_axis',expected),
            ['actuator_sha256','motion_sha256'])
        with self.assertRaisesRegex(AssertionError,'asset mismatch'):
            validate_teacher(dict(contract='23_axis',asset_sha256='other'),'23_axis',expected)

    def test_present_teacher_hashes_cannot_be_ignored(self):
        with self.assertRaisesRegex(AssertionError,'motion_sha256 mismatch'):
            validate_teacher(dict(contract='23_axis',asset_sha256='asset',motion_sha256='changed'),
                '23_axis',dict(asset_sha256='asset',motion_sha256='motion'))

    def setUp(self):
        torch.manual_seed(7)
        raw=torch.randn(8,320)*.1;raw[:,246]=torch.tensor([0.,.15,0.,1.5,0.,.15,0.,2.5]);raw[:,247:249]=0.
        self.obs=TensorDict({'policy':raw},batch_size=[8])
        self.a=actor(self.obs);self.teacher=copy.deepcopy(self.a)
        with torch.no_grad():self.a.mlp[-1].bias.add_(.7)

    def test_gradient_equals_explicit_standing_mean_squared_error(self):
        expected=copy.deepcopy(self.a);target=self.teacher(self.obs).detach().clamp(-3.,3.)
        output=expected(self.obs);stopped=self.obs['policy'][:,246].abs()<=.08
        loss=output.square().mean()+.25*(output[stopped]-target[stopped]).square().mean()
        loss.backward()
        prior=StandingRetention(self.a,self.teacher)
        value=self.a(self.obs);value.square().mean().backward()
        for left,right in zip(expected.parameters(),self.a.parameters()):
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-7)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertGreater(prior.read_audit()['standing_prior_mse'],.4)
        prior.close()

    def test_moving_commands_receive_original_ppo_gradient(self):
        self.obs['policy'][:,246]=.15
        expected=copy.deepcopy(self.a);expected(self.obs).square().mean().backward()
        prior=StandingRetention(self.a,self.teacher)
        self.a(self.obs).square().mean().backward()
        for left,right in zip(expected.parameters(),self.a.parameters()):
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,rtol=0.,atol=0.)
        self.assertEqual(prior.read_audit()['batches'],0)
        prior.close()

    def test_frozen_teacher_does_not_change_forward_actions_or_parameters(self):
        original=copy.deepcopy(self.a);prior=StandingRetention(self.a,self.teacher)
        self.a.eval()
        torch.testing.assert_close(self.a(self.obs),original(self.obs),rtol=0.,atol=0.)
        for k,v in original.state_dict().items():torch.testing.assert_close(v,self.a.state_dict()[k],rtol=0.,atol=0.)
        self.assertEqual(prior.read_audit()['batches'],0)
        prior.close()

    def test_collection_and_stochastic_update_paths_are_supported(self):
        prior=StandingRetention(self.a,self.teacher)
        with torch.inference_mode():self.a(self.obs,stochastic_output=True)
        self.assertEqual(prior.read_audit()['batches'],0)
        samples=self.a(self.obs,stochastic_output=True)
        loss=-self.a.get_output_log_prob(samples.detach()).mean()
        loss.backward()
        self.assertEqual(prior.read_audit()['batches'],1)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        prior.close()

    def test_supervised_update_reduces_stop_error_without_sim_helper(self):
        self.obs['policy'][:,246]=0.
        prior=StandingRetention(self.a,self.teacher)
        initial=float((self.a(self.obs)-self.teacher(self.obs)).square().mean())
        optimizer=torch.optim.SGD(self.a.parameters(),lr=.03)
        for _ in range(8):
            optimizer.zero_grad();prediction=self.a(self.obs)
            (prediction.sum()*0.).backward();optimizer.step()
        final=float((self.a(self.obs)-self.teacher(self.obs)).square().mean())
        self.assertLess(final,initial)
        prior.close()

    def test_teacher_sample_cap_bounds_added_work(self):
        prior=StandingRetention(self.a,self.teacher,max_samples=2)
        self.a(self.obs).square().mean().backward()
        self.assertEqual(prior._samples,2*23)
        prior.close()

    def test_matching_prior_has_zero_extra_gradient(self):
        self.a.load_state_dict(self.teacher.state_dict());expected=copy.deepcopy(self.a)
        expected(self.obs).square().mean().backward()
        prior=StandingRetention(self.a,self.teacher)
        self.a(self.obs).square().mean().backward()
        for left,right in zip(expected.parameters(),self.a.parameters()):
            if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-7)
        # Subset versus full-batch matrix products have float32 roundoff.
        self.assertLess(prior.read_audit()['standing_prior_mse'],1e-12)
        prior.close()


if __name__=='__main__':unittest.main(verbosity=2)
