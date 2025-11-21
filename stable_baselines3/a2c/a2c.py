from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

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
        self.log_param_norms = log_param_norms

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer (one gradient step over whole data).
        """
        self.policy.set_training_mode(True)

        # Values recorded for logging after finishing the minibatch loop
        policy_loss = None
        value_loss = None
        entropy_loss = None

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

            if self.use_pullback:
                (
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    actor_grad_norm,
                ) = self._anpg_pb_update(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.advantages,
                    rollout_data.returns,
                )
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
        if policy_loss is not None:
            self.logger.record("train/policy_loss", policy_loss.item())
        if value_loss is not None:
            self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        if self.use_pullback:
            self.logger.record("train/actor_grad_norm", actor_grad_norm.item())
        if self.log_param_norms:
            self._record_param_norms()

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

        _, _, G = self._build_J_Cinv_G(observations, actions)

        one_over_h = 1.0 / max(self.pb_h, 1e-12)
        lam = self.pb_lambda

        Aop = self._Aop_from_dense_G(G, one_over_h, lam)
        delta = self._conjugate_gradient(Aop, -g, max_iter=self.pb_cg_max_iter, tol=self.pb_cg_tol)

        with th.no_grad():
            dnorm = th.norm(delta)
            if self.pb_step_clip is not None and dnorm > self.pb_step_clip:
                delta = delta / (dnorm + 1e-12) * self.pb_step_clip
            self._add_flat_(params, delta, alpha=1.0)

        actor_grad_norm = th.norm(g)
        return policy_loss.detach(), value_loss.detach(), entropy_loss.detach(), actor_grad_norm

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
            raise ValueError(f"statistic '{self.statistic}' not implemented for continuous SB3 policy.")
        else:
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            logp = dist.log_prob(actions.long()).view(-1, 1)
            if self.statistic == "logp":
                return logp
            if self.statistic == "entropy":
                ent = dist.entropy().view(-1, 1)
                return ent
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

    def _build_J_Cinv_G(self, states, actions):
        J = self._build_J(states, actions)
        C = self._compute_C_t_full(states, actions)
        C_inv = self._invert_spd(C)
        with th.no_grad():
            CJ = C_inv @ J
            G = J.transpose(0, 1) @ CJ
            G = 0.5 * (G + G.transpose(0, 1))
        return J, C_inv, G

    def _Aop_from_dense_G(self, G, one_over_h, lam):
        def Aop(v):
            with th.no_grad():
                return one_over_h * (G @ v) + lam * v

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
        spectral_max = float(max(spectral_norms))
        nuclear_mean = float(sum(nuclear_norms) / len(nuclear_norms))
        nuclear_max = float(max(nuclear_norms))
        rank_mean = float(sum(ranks) / len(ranks))
        rank_max = float(max(ranks))
        rank_sum = float(sum(ranks))

        self.logger.record("train/param_norms/frobenius", frobenius_norm)
        self.logger.record("train/param_norms/spectral_mean", spectral_mean)
        self.logger.record("train/param_norms/spectral_max", spectral_max)
        self.logger.record("train/param_norms/nuclear_mean", nuclear_mean)
        self.logger.record("train/param_norms/nuclear_max", nuclear_max)
        self.logger.record("train/param_norms/rank_mean", rank_mean)
        self.logger.record("train/param_norms/rank_max", rank_max)
        self.logger.record("train/param_norms/rank_sum", rank_sum)
