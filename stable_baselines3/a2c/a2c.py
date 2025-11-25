from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, update_learning_rate

SelfA2C = TypeVar("SelfA2C", bound="A2C")


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`a2c_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param use_pullback: Whether to perform the pullback/anpg-style update instead of the vanilla one
    :param statistic: Statistic used when building the pullback metric ("logp" or "entropy")
    :param prox_h: Pullback proximal parameter ``h``
    :param cg_lambda: Conjugate gradient regularization
    :param cg_max_iter: Maximum number of conjugate gradient iterations
    :param cg_tol: Conjugate gradient tolerance
    :param fisher_ridge: Ridge term added to the covariance estimate
    :param step_clip: Optional clipping applied to the search direction length
    :param actor_learning_rate: Optional custom learning rate (or schedule) for the actor optimizer
        when ``separate_optimizers`` is True.
    :param critic_learning_rate: Optional custom learning rate (or schedule) for the critic optimizer
        when ``separate_optimizers`` is True.
    :param separate_optimizers: If True, use two optimizers to update actor and critic separately
        (hyperparameters identical). Shared feature extractor (if any) is updated once using
        the combined gradients from both losses.
    :param log_param_norms: When True, log parameter norms (Frobenius, spectral, nuclear) and rank each update
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
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        use_pullback: bool = False,
        statistic: str = "logp",
        prox_h: float = 10.0,
        cg_lambda: float = 1e-2,
        cg_max_iter: int = 10,
        cg_tol: float = 1e-10,
        fisher_ridge: float = 1e-1,
        step_clip: Optional[float] = 0.1,
        actor_learning_rate: Union[float, Schedule, None] = None,
        critic_learning_rate: Union[float, Schedule, None] = None,
        separate_optimizers: bool = False,
        log_param_norms: bool = False,
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

        self.normalize_advantage = normalize_advantage

        self.use_pullback = use_pullback
        self.statistic = statistic
        self.pb_h = float(prox_h)
        self.pb_lambda = float(cg_lambda)
        self.pb_cg_max_iter = int(cg_max_iter)
        self.pb_cg_tol = float(cg_tol)
        self.pb_c_ridge = float(fisher_ridge)
        self.pb_step_clip = step_clip
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.separate_optimizers = separate_optimizers
        self.log_param_norms = log_param_norms

        # Split-optimizer related attributes
        self.actor_optimizer: Optional[th.optim.Optimizer] = None
        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self._actor_params: Optional[list[th.nn.Parameter]] = None
        self._critic_params: Optional[list[th.nn.Parameter]] = None
        self.actor_lr_schedule: Optional[Schedule] = None
        self.critic_lr_schedule: Optional[Schedule] = None

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # When requested, build two optimizers with separated parameter groups
        if self.separate_optimizers:
            if self.actor_learning_rate is not None:
                self.actor_lr_schedule = FloatSchedule(self.actor_learning_rate)
            if self.critic_learning_rate is not None:
                self.critic_lr_schedule = FloatSchedule(self.critic_learning_rate)

            # Helpers to collect unique parameters
            def _extend_unique(dst: list[th.nn.Parameter], params_iter) -> None:
                seen = {id(p) for p in dst}
                for p in params_iter:
                    if id(p) not in seen:
                        dst.append(p)
                        seen.add(id(p))

            actor_params: list[th.nn.Parameter] = []
            critic_params: list[th.nn.Parameter] = []

            # Actor-specific modules
            _extend_unique(actor_params, self.policy.mlp_extractor.policy_net.parameters())
            _extend_unique(actor_params, self.policy.action_net.parameters())
            if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, th.nn.Parameter):
                actor_params.append(self.policy.log_std)

            # Critic-specific modules
            _extend_unique(critic_params, self.policy.mlp_extractor.value_net.parameters())
            _extend_unique(critic_params, self.policy.value_net.parameters())

            # Feature extractors: shared or separate
            if getattr(self.policy, "share_features_extractor", True):
                _extend_unique(actor_params, self.policy.features_extractor.parameters())
            else:
                _extend_unique(actor_params, self.policy.pi_features_extractor.parameters())
                _extend_unique(critic_params, self.policy.vf_features_extractor.parameters())

            # Save param lists for clipping/logging
            self._actor_params = actor_params
            self._critic_params = critic_params

            # Create optimizers mirroring policy optimizer hyperparameters
            actor_initial_lr = (
                self.actor_lr_schedule(1) if self.actor_lr_schedule is not None else self.lr_schedule(1)
            )
            critic_initial_lr = (
                self.critic_lr_schedule(1) if self.critic_lr_schedule is not None else self.lr_schedule(1)
            )
            self.actor_optimizer = self.policy.optimizer_class(self._actor_params, lr=actor_initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]
            self.critic_optimizer = self.policy.optimizer_class(self._critic_params, lr=critic_initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer (one gradient step over whole data).
        """
        self.policy.set_training_mode(True)

        # Values recorded for logging after finishing the minibatch loop
        policy_loss = None
        value_loss = None
        entropy_loss = None
        actor_grad_norm = None
        critic_grad_norm = None

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            policy_loss = -(advantages * log_prob).mean()
            value_loss = F.mse_loss(rollout_data.returns, values)

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            dnorm = None
            if self.use_pullback:
                (
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    actor_grad_norm,
                    dnorm,
                ) = self._anpg_pb_update(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.advantages,
                    rollout_data.returns,
                )
            else:
                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None

                    actor_lr = (
                        self.actor_lr_schedule(self._current_progress_remaining)  # type: ignore[operator]
                        if self.actor_lr_schedule is not None
                        else self.lr_schedule(self._current_progress_remaining)
                    )
                    critic_lr = (
                        self.critic_lr_schedule(self._current_progress_remaining)  # type: ignore[operator]
                        if self.critic_lr_schedule is not None
                        else self.lr_schedule(self._current_progress_remaining)
                    )
                    self.logger.record("train/learning_rate", actor_lr)
                    self.logger.record("train/learning_rate_actor", actor_lr)
                    self.logger.record("train/learning_rate_critic", critic_lr)
                    update_learning_rate(self.actor_optimizer, actor_lr)
                    update_learning_rate(self.critic_optimizer, critic_lr)
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    actor_loss = policy_loss + self.ent_coef * entropy_loss
                    actor_loss.backward(retain_graph=True)
                    critic_loss = self.vf_coef * value_loss
                    critic_loss.backward()

                    actor_grad_norm = th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
                    self.actor_optimizer.step()

                    critic_grad_norm = th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
                    self.critic_optimizer.step()
                else:
                    self._update_learning_rate(self.policy.optimizer)
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    actor_grad_norm = th.norm(
                        th.cat([p.grad.view(-1) for p in self.policy.parameters() if p.grad is not None])
                    )

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        if entropy_loss is not None:
            self.logger.record("train/entropy_loss", entropy_loss.item())
        if dnorm is not None:
            self.logger.record("train/dnorm", dnorm.item())
        if policy_loss is not None:
            self.logger.record("train/policy_loss", policy_loss.item())
        if value_loss is not None:
            self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        if actor_grad_norm is not None:
            self.logger.record("train/actor_grad_norm", actor_grad_norm.item())
        if critic_grad_norm is not None:
            self.logger.record("train/critic_grad_norm", critic_grad_norm.item())
        if self.log_param_norms:
            self._record_param_norms()

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Include separate optimizers in state dicts when enabled.
        """
        if self.separate_optimizers:
            return ["policy", "policy.optimizer", "actor_optimizer", "critic_optimizer"], []
        # Default behavior from parent
        return super()._get_torch_save_params()

    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _anpg_pb_update(self, observations, actions, advantages, returns):
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        adv = advantages
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(adv * log_prob).mean()
        value_loss = F.mse_loss(returns, values)
        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self._update_learning_rate(self.policy.optimizer)
        params = self._params()

        g_list = th.autograd.grad(loss, params, retain_graph=False, create_graph=False)
        g = self._flat([gi if gi is not None else th.zeros_like(p) for gi, p in zip(g_list, params)])

        J, C_inv = self._build_J_Cinv(observations, actions)

        one_over_h = 1.0 / max(self.pb_h, 1e-12)
        lam = self.pb_lambda

        Aop = self._Aop_from_j_cinv(J, C_inv, one_over_h, lam)
        delta = self._conjugate_gradient(Aop, -g, max_iter=self.pb_cg_max_iter, tol=self.pb_cg_tol)
        with th.no_grad():
            dnorm = th.norm(delta)
            if self.pb_step_clip is not None and dnorm > self.pb_step_clip:
                delta = delta / (dnorm + 1e-12) * self.pb_step_clip
            self._add_flat_(params, delta, alpha=self.actor_learning_rate)

        actor_grad_norm = th.norm(g)
        return policy_loss.detach(), value_loss.detach(), entropy_loss.detach(), actor_grad_norm, dnorm

    def _params(self):
        return [p for p in self.policy.parameters() if p.requires_grad]

    def _flat(self, tensors):
        return th.cat([t.reshape(-1) for t in tensors])

    @th.no_grad()
    def _add_flat_(self, params, delta, alpha: float = 1.0):
        off = 0
        for p in params:
            n = p.numel()
            p.add_(alpha * delta[off : off + n].view_as(p))
            off += n

    def _conjugate_gradient(self, Aop, b, max_iter: int = 50, tol: float = 1e-10):
        x = th.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = th.dot(r, r)
        for _ in range(max_iter):
            Ap = Aop(p)
            denom = th.dot(p, Ap) + 1e-12
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = th.dot(r, r)
            if th.sqrt(rs_new) < tol:
                break
            p = r + (rs_new / (rs_old + 1e-12)) * p
            rs_old = rs_new
        return x

    def _statistic_components(self, states, actions):
        if states.dim() == 1:
            states = states.unsqueeze(0)
        dist = self.policy.get_distribution(states)

        if isinstance(self.action_space, spaces.Box):
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

            # 通用：logp / entropy 分支（保留原逻辑）
            logp = dist.log_prob(actions)
            if logp.dim() == 1:
                logp = logp.unsqueeze(-1)
            if self.statistic == "logp":
                if logp.dim() > 1:
                    logp = logp.sum(dim=-1, keepdim=True)
                return logp
            if self.statistic == "entropy":
                ent = dist.entropy()
                if ent.dim() > 1:
                    ent = ent.sum(dim=-1, keepdim=True)
                else:
                    ent = ent.unsqueeze(-1)
                return ent

            if self.statistic == "score_per_dim":
                mu = dist.distribution.loc
                std = dist.distribution.scale
                var = std * std

                s_mu = (actions - mu) / (var + 1e-12)

                s_logsig = (actions - mu).pow(2) / (var + 1e-12) - 1.0

                return th.cat([s_mu, s_logsig], dim=-1)

            raise ValueError(f"statistic '{self.statistic}' not implemented for continuous SB3 policy.")

        else:
            # 离散：保留原逻辑
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            logp = dist.log_prob(actions.long()).view(-1, 1)
            if self.statistic == "logp":
                return logp
            if self.statistic == "entropy":
                ent = dist.entropy().view(-1, 1)
                return ent

            # 新增：逐类 score（one-hot - probs），返回 [B, A]
            if self.statistic == "score_per_dim":
                probs = dist.distribution.probs  # torch.distributions.Categorical
                num_actions = probs.shape[-1]
                one_hot = F.one_hot(actions.long(), num_classes=num_actions).float()
                # score: ∂/∂logits log Cat = one_hot(a) - probs
                return one_hot - probs

            raise ValueError(f"statistic '{self.statistic}' not implemented for discrete SB3 policy.")

    def _t_mean(self, states, actions):
        t = self._statistic_components(states, actions)
        return t.mean(dim=0)

    @th.no_grad()
    def _compute_C_t_full(self, states, actions, ridge: Optional[float] = None):
        t = self._statistic_components(states, actions)
        batch_size, num_stats = t.shape
        t_center = t - t.mean(dim=0, keepdim=True)
        C = (t_center.T @ t_center) / max(batch_size, 1)
        r = float(self.pb_c_ridge if ridge is None else ridge)
        C = C + r * th.eye(num_stats, device=C.device, dtype=C.dtype)
        return C

    @th.no_grad()
    def _invert_spd(self, C):
        K = C.shape[0]
        I = th.eye(K, device=C.device, dtype=C.dtype)
        try:
            L = th.linalg.cholesky(C)
            C_inv = th.cholesky_solve(I, L)
        except RuntimeError:
            C_inv = th.linalg.pinv(C)
        return C_inv

    def _build_J(self, states, actions):
        params = self._params()
        t_mean = self._t_mean(states, actions)
        K = t_mean.numel()
        D = sum(p.numel() for p in params)
        J = th.zeros(K, D, device=states.device, dtype=states.dtype)
        for k in range(K):
            retain = k != K - 1
            grad_list = th.autograd.grad(
                t_mean[k],
                params,
                retain_graph=retain,
                create_graph=False,
                allow_unused=True,
            )
            row = self._flat([g if g is not None else th.zeros_like(p) for g, p in zip(grad_list, params)])
            J[k].copy_(row)
        return J

    def _build_J_Cinv(self, states, actions):
        J = self._build_J(states, actions)
        C = self._compute_C_t_full(states, actions)
        C_inv = self._invert_spd(C)
        with th.no_grad():
            # Ensure consistent dtype to avoid float/double matmul mismatch
            J = J.to(C_inv.dtype)
        return J, C_inv

    def _Aop_from_j_cinv(self, J, C_inv, one_over_h, lam):
        def Aop(v):
            with th.no_grad():
                # Use implicit multiplication to avoid forming dense G = J^T C_inv J
                Jv = J @ v
                CJv = C_inv @ Jv
                Gv = J.transpose(0, 1) @ CJv
                return one_over_h * Gv + lam * v

        return Aop

    @th.no_grad()
    def _record_param_norms(self) -> None:
        params = [p.detach() for p in self.policy.parameters() if p.requires_grad]
        if len(params) == 0:
            return

        float_params = [p.float() for p in params]
        frobenius_sq = sum(p.pow(2).sum() for p in float_params)
        frobenius_norm = th.sqrt(frobenius_sq).item()

        spectral_norms: list[float] = []
        nuclear_norms: list[float] = []
        ranks: list[float] = []

        for p in float_params:
            if p.ndim <= 1:
                mat = p.view(1, -1)
            else:
                mat = p.view(p.shape[0], -1)
            spectral_norms.append(th.linalg.matrix_norm(mat, 2).item())
            nuclear_norms.append(th.linalg.matrix_norm(mat, "nuc").item())
            ranks.append(float(th.linalg.matrix_rank(mat).item()))

        spectral_mean = float(sum(spectral_norms) / len(spectral_norms))
        nuclear_mean = float(sum(nuclear_norms) / len(nuclear_norms))
        rank_mean = float(sum(ranks) / len(ranks))

        self.logger.record("param_norms/frobenius", frobenius_norm)
        self.logger.record("param_norms/spectral", spectral_mean)
        self.logger.record("param_norms/nuclear", nuclear_mean)
        self.logger.record("param_norms/rank", rank_mean)
