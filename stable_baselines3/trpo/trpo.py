import copy
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, update_learning_rate

SelfTRPO = TypeVar("SelfTRPO", bound="TRPO")


def _flat_grad(y: th.Tensor, params: list[th.nn.Parameter], retain_graph: bool = False, create_graph: bool = False) -> th.Tensor:
    grads = th.autograd.grad(y, params, retain_graph=retain_graph, create_graph=create_graph)
    return th.cat([grad.reshape(-1) for grad in grads])


def _get_flat_params(params: list[th.nn.Parameter]) -> th.Tensor:
    return th.cat([p.data.view(-1) for p in params])


def _set_flat_params(params: list[th.nn.Parameter], flat: th.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.data.copy_(flat[offset : offset + numel].view_as(param))
        offset += numel


class TRPO(OnPolicyAlgorithm):
    """
    Trust Region Policy Optimization (TRPO).

    Paper: https://arxiv.org/abs/1502.05477

    The implementation follows the vanilla formulation with a conjugate-gradient
    natural gradient step on the policy and standard gradient updates on the value
    function.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: Optimizer learning rate for the value function
    :param n_steps: The number of steps to run for each environment per update
    :param gamma: Discount factor
    :param gae_lambda: Factor for bias vs variance for Generalized Advantage Estimator.
    :param ent_coef: Entropy coefficient
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: Max gradient norm for critic updates
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for logging
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...)
    :param advantage_multiplier: Multiply normalized advantages by this factor
    :param normalize_advantage: Whether to normalize advantages
    :param normalize_advantage_mean: If True, subtract mean before std normalization
    :param max_kl: KL divergence constraint
    :param cg_iters: Number of conjugate gradient iterations
    :param cg_damping: Damping factor added to Fisher-vector products
    :param line_search_coef: Step reduction factor during backtracking line search
    :param line_search_max_backtracks: Max number of backtracking steps
    :param vf_iters: Number of gradient steps for the value function per update
    :param vf_batch_size: Batch size for value updates (defaults to full rollout)
    :param separate_optimizers: If True, build a dedicated optimizer for the critic
    :param _init_setup_model: Whether or not to build the network at instance creation
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        advantage_multiplier: float = 1.0,
        normalize_advantage: bool = True,
        normalize_advantage_mean: bool = True,
        max_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        line_search_coef: float = 0.8,
        line_search_max_backtracks: int = 10,
        vf_iters: int = 5,
        vf_batch_size: Optional[int] = None,
        separate_optimizers: bool = True,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.advantage_multiplier = advantage_multiplier
        self.normalize_advantage = normalize_advantage
        self.normalize_advantage_mean = normalize_advantage_mean
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_coef = line_search_coef
        self.line_search_max_backtracks = line_search_max_backtracks
        self.vf_iters = vf_iters
        self.vf_batch_size = vf_batch_size
        self.separate_optimizers = separate_optimizers

        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self._actor_params: Optional[list[th.nn.Parameter]] = None
        self._critic_params: Optional[list[th.nn.Parameter]] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # Use full rollout by default
        if self.batch_size is None:
            self.batch_size = self.n_steps * self.n_envs
        if self.vf_batch_size is None:
            self.vf_batch_size = self.batch_size

        if self.separate_optimizers:
            self._actor_params, self._critic_params = self._collect_actor_critic_params()

            critic_initial_lr = self.lr_schedule(1)
            self.critic_optimizer = self.policy.optimizer_class(
                self._critic_params, lr=critic_initial_lr, **self.policy.optimizer_kwargs
            )  # type: ignore[arg-type]

    def _collect_actor_critic_params(self) -> tuple[list[th.nn.Parameter], list[th.nn.Parameter]]:
        def _extend_unique(dst: list[th.nn.Parameter], params_iter) -> None:
            seen = {id(p) for p in dst}
            for p in params_iter:
                if id(p) not in seen:
                    dst.append(p)
                    seen.add(id(p))

        actor_params: list[th.nn.Parameter] = []
        critic_params: list[th.nn.Parameter] = []

        _extend_unique(actor_params, self.policy.mlp_extractor.policy_net.parameters())
        _extend_unique(actor_params, self.policy.action_net.parameters())
        if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, th.nn.Parameter):
            actor_params.append(self.policy.log_std)

        _extend_unique(critic_params, self.policy.mlp_extractor.value_net.parameters())
        _extend_unique(critic_params, self.policy.value_net.parameters())

        if getattr(self.policy, "share_features_extractor", True):
            _extend_unique(actor_params, self.policy.features_extractor.parameters())
        else:
            _extend_unique(actor_params, self.policy.pi_features_extractor.parameters())
            _extend_unique(critic_params, self.policy.vf_features_extractor.parameters())

        return actor_params, critic_params

    def conjugate_gradients(self, fvp_fn, b: th.Tensor, nsteps: int) -> th.Tensor:
        x = th.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = th.dot(r, r)
        for _ in range(nsteps):
            Avp = fvp_fn(p)
            alpha = rdotr / (th.dot(p, Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = th.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _get_policy_distribution(self, obs: th.Tensor) -> Any:
        return self.policy.get_distribution(obs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        assert self.batch_size is not None
        # Ensure parameter lists are ready for natural gradient update
        if self._actor_params is None or self._critic_params is None:
            self._actor_params, self._critic_params = self._collect_actor_critic_params()
            if self.critic_optimizer is None:
                self.critic_optimizer = self.policy.optimizer_class(
                    self._critic_params, lr=self.lr_schedule(1), **self.policy.optimizer_kwargs
                )  # type: ignore[arg-type]

        # One full-batch pass for the policy step
        rollout_data = next(self.rollout_buffer.get(self.batch_size))
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            if self.normalize_advantage_mean:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages / (advantages.std() + 1e-8)
        advantages = advantages * self.advantage_multiplier

        # Snapshot old policy for KL and ratio baselines
        with th.no_grad():
            old_policy = copy.deepcopy(self.policy)
            old_policy.set_training_mode(False)
            old_policy.to(self.device)
            old_dist = old_policy.get_distribution(rollout_data.observations)
            old_log_prob = rollout_data.old_log_prob

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        ratio = th.exp(log_prob - old_log_prob)
        surrogate_loss = -(advantages * ratio).mean()
        policy_grad = _flat_grad(surrogate_loss, self._actor_params)
        policy_grad_detached = policy_grad.detach()

        # Fisher-vector product using KL Hessian
        def fisher_vector_product(vec: th.Tensor) -> th.Tensor:
            new_dist = self._get_policy_distribution(rollout_data.observations)
            kl = kl_divergence(old_dist, new_dist).mean()
            # retain graph for the second backward pass when computing Hessian-vector product
            kl_grad = _flat_grad(kl, self._actor_params, retain_graph=True, create_graph=True)
            kl_v = (kl_grad * vec).sum()
            hvp = _flat_grad(kl_v, self._actor_params)
            return hvp + self.cg_damping * vec

        # Natural gradient direction for minimizing surrogate_loss
        step_dir = -self.conjugate_gradients(fisher_vector_product, policy_grad_detached, self.cg_iters)
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum()
        step_size = th.sqrt(2 * self.max_kl / (shs + 1e-8))
        full_step = step_dir * step_size
        expected_improve = -(policy_grad_detached * full_step).sum()

        old_params = _get_flat_params(self._actor_params).detach()
        old_loss = surrogate_loss.detach()

        def set_and_eval(step_frac: float) -> tuple[float, float]:
            new_params = old_params + step_frac * full_step
            _set_flat_params(self._actor_params, new_params)
            with th.no_grad():
                new_dist = self._get_policy_distribution(rollout_data.observations)
                new_log_prob = new_dist.log_prob(actions)
                new_ratio = th.exp(new_log_prob - old_log_prob)
                new_loss = -(advantages * new_ratio).mean()
                kl_val = kl_divergence(old_dist, new_dist).mean()
            return new_loss.item(), kl_val.item()

        success = False
        final_kl = 0.0
        loss_after = old_loss.item()
        for step in range(self.line_search_max_backtracks):
            step_frac = self.line_search_coef**step
            candidate_loss, candidate_kl = set_and_eval(step_frac)
            improve = old_loss.item() - candidate_loss
            expected = expected_improve.item() * step_frac
            if (improve > 0) and candidate_kl <= self.max_kl and (improve >= 0.1 * expected) and candidate_kl <= self.max_kl:
                success = True
                final_kl = candidate_kl
                loss_after = candidate_loss
                break
        if not success:
            _set_flat_params(self._actor_params, old_params)

        # Value function updates
        value_losses = []
        if self.critic_optimizer is not None:
            current_lr = self.lr_schedule(self._current_progress_remaining)
            update_learning_rate(self.critic_optimizer, current_lr)
            self.logger.record("train/learning_rate", current_lr)

            # Single critic update per rollout, mirroring A2C behaviour
            for value_batch in self.rollout_buffer.get(self.vf_batch_size):
                value_pred = self.policy.predict_values(value_batch.observations).flatten()
                value_loss = F.mse_loss(value_batch.returns, value_pred)
                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
                self.critic_optimizer.step()
                value_losses.append(value_loss.item())
                break

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logging
        self.logger.record("train/policy_gradient_loss", surrogate_loss.item())
        self.logger.record("train/policy_loss_after", loss_after)
        self.logger.record("train/approx_kl", final_kl)
        self.logger.record("train/entropy_loss", -th.mean(entropy).item() if entropy is not None else -log_prob.mean().item())
        self.logger.record("train/value_loss", np.mean(value_losses) if len(value_losses) > 0 else 0.0)
        self.logger.record("train/grad_norm_actor", policy_grad_detached.norm().item())
        self.logger.record("train/expected_improve", expected_improve.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/line_search_accepted", float(success))
        self.logger.record("train/line_search_steps", 0 if not success else step)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        if self.separate_optimizers:
            return ["policy", "policy.optimizer", "critic_optimizer"], []
        return super()._get_torch_save_params()
