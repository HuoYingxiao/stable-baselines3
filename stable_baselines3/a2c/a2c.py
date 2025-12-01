from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, update_learning_rate

SelfA2C = TypeVar("SelfA2C", bound="A2C")


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)
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
        actor_learning_rate: Union[float, Schedule, str, None] = None,
        critic_learning_rate: Union[float, Schedule, str, None] = None,
        separate_optimizers: bool = False,
        log_param_norms: bool = False,
        fr_order: int = 2,
        # inner loop
        pb_use_inner_loop: bool = False,
        pb_inner_steps: int = 5,
        pb_inner_lr: float = 1.0,
        pb_use_kernel: bool = False,
        pb_kernel_num_anchors: int = 64,
        pb_kernel_sigma: float = 1.0,
        pb_kernel_type: str = "rbf",        # "rbf", "rq", "laplace", "matern32", "poly"
        pb_kernel_alpha: float = 1.0,       # α,rational quadratic / polynomial
        pb_kernel_c: float = 0.0,           # c, polynomial kernel
        pb_kernel_auto_sigma: bool = True,
        pb_use_momentum: bool = False,
        pb_momentum_beta: float = 0.9,
        pb_use_nesterov_predict: bool = False,
        pb_predict_iters: int = 1,
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
        self.fr_order = fr_order

        # inner loop
        self.pb_use_inner_loop = pb_use_inner_loop
        self.pb_inner_steps = int(pb_inner_steps)
        self.pb_inner_lr = float(pb_inner_lr)

        # kernal function
        self.pb_use_kernel = pb_use_kernel
        self.pb_kernel_num_anchors = int(pb_kernel_num_anchors)
        self.pb_kernel_sigma = float(pb_kernel_sigma)
        self.pb_kernel_type = pb_kernel_type.lower()
        self.pb_kernel_alpha = float(pb_kernel_alpha)
        self.pb_kernel_c = float(pb_kernel_c)
        self.pb_kernel_auto_sigma = bool(pb_kernel_auto_sigma)
        self._kernel_anchors: Optional[th.Tensor] = None  # [M, K_base]

        # momentum
        self.pb_use_momentum = pb_use_momentum
        self.pb_momentum_beta = float(pb_momentum_beta)
        self.pb_use_midpoint_predict = pb_use_nesterov_predict
        self.pb_predict_iters = int(pb_predict_iters)


        self._pb_last_direction_flat: Optional[th.Tensor] = None
        self._pb_last_step_flat: Optional[th.Tensor] = None


        # optimizers
        self.actor_optimizer: Optional[th.optim.Optimizer] = None
        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self._actor_params: Optional[list[th.nn.Parameter]] = None
        self._critic_params: Optional[list[th.nn.Parameter]] = None
        self.actor_lr_schedule: Optional[Schedule] = None
        self.critic_lr_schedule: Optional[Schedule] = None

        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.use_pullback and not self.separate_optimizers:
            raise ValueError("separate_optimizers=True。")

        if self.separate_optimizers:
            if self.actor_learning_rate is not None:
                self.actor_lr_schedule = FloatSchedule(self.actor_learning_rate)
            if self.critic_learning_rate is not None:
                self.critic_lr_schedule = FloatSchedule(self.critic_learning_rate)

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

            self._actor_params = actor_params
            self._critic_params = critic_params

            actor_initial_lr = (
                self.actor_lr_schedule(1) if self.actor_lr_schedule is not None else self.lr_schedule(1)
            )
            critic_initial_lr = (
                self.critic_lr_schedule(1) if self.critic_lr_schedule is not None else self.lr_schedule(1)
            )
            self.actor_optimizer = self.policy.optimizer_class(
                self._actor_params, lr=actor_initial_lr, **self.policy.optimizer_kwargs
            )  # type: ignore[arg-type]
            self.critic_optimizer = self.policy.optimizer_class(
                self._critic_params, lr=critic_initial_lr, **self.policy.optimizer_kwargs
            )  # type: ignore[arg-type]

    def train(self) -> None:
        self.policy.set_training_mode(True)

        policy_loss = None
        value_loss = None
        entropy_loss = None
        actor_grad_norm = None
        critic_grad_norm = None
        dnorm = None

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            if self.use_pullback:
                assert self.separate_optimizers
                assert self._actor_params is not None and self._critic_params is not None
                assert self.actor_optimizer is not None and self.critic_optimizer is not None

                # lr 调度
                lr_actor = (
                    self.actor_lr_schedule(self._current_progress_remaining)  # type: ignore[operator]
                    if self.actor_lr_schedule is not None
                    else self.lr_schedule(self._current_progress_remaining)
                )
                lr_critic = (
                    self.critic_lr_schedule(self._current_progress_remaining)  # type: ignore[operator]
                    if self.critic_lr_schedule is not None
                    else self.lr_schedule(self._current_progress_remaining)
                )

                self.logger.record("train/learning_rate_actor", lr_actor)
                self.logger.record("train/learning_rate_critic", lr_critic)
                self.logger.record("train/learning_rate", lr_actor)

                update_learning_rate(self.actor_optimizer, lr_actor)
                update_learning_rate(self.critic_optimizer, lr_critic)

                # actor：pullback 更新
                if self.pb_use_inner_loop:
                    policy_loss, entropy_loss, actor_grad_norm, dnorm = self._anpg_prox_inner_update(
                        rollout_data.observations,
                        actions,
                        rollout_data.advantages,
                    )
                else:
                    policy_loss, entropy_loss, actor_grad_norm, dnorm = self._anpg_pb_update(
                        rollout_data.observations,
                        actions,
                        rollout_data.advantages,
                    )

                # critic：普通 A2C 更新
                values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                value_loss = F.mse_loss(rollout_data.returns, values)
                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                critic_grad_norm = th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm)
                self.critic_optimizer.step()

            else:
                # 原始 A2C
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
        if self.separate_optimizers:
            return ["policy", "policy.optimizer", "actor_optimizer", "critic_optimizer"], []
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

    def _anpg_pb_update(self, observations, actions, advantages):
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        # 1. 在当前参数 θ_k 上计算基本量：loss 和梯度 g_k
        values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
        values = values.detach()

        adv = advantages
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(adv * log_prob).mean()
        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        actor_loss = policy_loss + self.ent_coef * entropy_loss

        # 选择要更新的参数列表（actor 参数）
        if self.separate_optimizers and self._actor_params is not None:
            params = self._actor_params
        else:
            params = self._params()

        # 当前点 θ_k 的梯度 g_k
        g_list = th.autograd.grad(actor_loss, params, retain_graph=False, create_graph=False)
        g_base = self._flat([gi if gi is not None else th.zeros_like(p) for gi, p in zip(g_list, params)])

        one_over_h = 1.0 / max(self.pb_h, 1e-12)
        lam = self.pb_lambda

        # 当前参数向量 θ_k
        theta_flat_base = self._get_flat(params)

        # 2. Nesterov 外推点版本
        #    用 θ_nest = θ_k + β * Δθ_{k-1} 做 BE 中的“近似 θ_{k+1}”，
        #    然后在 θ_nest 上计算几何矩阵和梯度，用来构造隐式步。
        if self.pb_use_midpoint_predict:
            # 构造外推点 θ_nest
            use_last_step = (
                self._pb_last_step_flat is not None
                and self._pb_last_step_flat.shape == theta_flat_base.shape
            )
            if use_last_step:
                beta_nest = self.pb_momentum_beta
                theta_flat_nest = theta_flat_base + beta_nest * self._pb_last_step_flat
            else:
                # 第一次或上一次未更新成功时，没有历史步长，退化为当前点
                theta_flat_nest = theta_flat_base

            # 把参数临时设到 θ_nest
            self._set_flat_(params, theta_flat_nest)

            # 在 θ_nest 上重算 loss 和梯度（近似 BE 中的 ∇f(θ_{k+1}))
            values_p, log_prob_p, entropy_p = self.policy.evaluate_actions(observations, actions)
            values_p = values_p.detach()

            adv_p = advantages
            if self.normalize_advantage:
                adv_p = (adv_p - adv_p.mean()) / (adv_p.std() + 1e-8)

            policy_loss_p = -(adv_p * log_prob_p).mean()
            if entropy_p is None:
                entropy_loss_p = -th.mean(-log_prob_p)
            else:
                entropy_loss_p = -th.mean(entropy_p)

            actor_loss_p = policy_loss_p + self.ent_coef * entropy_loss_p

            grad_list_p = th.autograd.grad(actor_loss_p, params, retain_graph=False, create_graph=False)
            g_nest = self._flat(
                [gi if gi is not None else th.zeros_like(p) for gi, p in zip(grad_list_p, params)]
            ).detach()

            # 在 θ_nest 上构造 G(θ_nest) = J^T C^{-1} J
            J_nest, C_inv_nest = self._build_J_Cinv(observations, actions, params)

            # 回滚到真实参数 θ_k，真正的更新仍然是从 θ_k 出发
            self._set_flat_(params, theta_flat_base)

            # 在 θ_nest 所对应的几何上求解隐式步：
            # (1/h) G(θ_nest) Δθ + λ Δθ = - ∇f(θ_nest)
            Aop = self._Aop_from_j_cinv(J_nest, C_inv_nest, one_over_h, lam)
            delta = self._conjugate_gradient(
                Aop,
                -g_nest,
                max_iter=self.pb_cg_max_iter,
                tol=self.pb_cg_tol,
            )

            J = J_nest
            C_inv = C_inv_nest

        else:
            # 不用预测器：直接在当前 θ_k 上算 G(θ_k) 并解一次 BE 系统
            J, C_inv = self._build_J_Cinv(observations, actions, params)
            Aop = self._Aop_from_j_cinv(J, C_inv, one_over_h, lam)
            delta = self._conjugate_gradient(
                Aop,
                -g_base,
                max_iter=self.pb_cg_max_iter,
                tol=self.pb_cg_tol,
            )

        # 3. 方向后处理（动量 / Nesterov 外的惯性）+ line search + 记录步长
        with th.no_grad():
            search_delta = delta
            if self.pb_use_momentum:
                if (
                    self._pb_last_direction_flat is not None
                    and self._pb_last_direction_flat.shape == delta.shape
                ):
                    beta = self.pb_momentum_beta
                    # 这里是“方向动量”，和上面的 Nesterov 外推是两个概念
                    search_delta = beta * self._pb_last_direction_flat + (1.0 - beta) * delta
                self._pb_last_direction_flat = search_delta.detach().clone()

            dnorm = th.norm(search_delta)

            # FR 二次型：Δθ^T G Δθ
            Jdelta = J @ search_delta
            Cinv_Jdelta = C_inv @ Jdelta
            fr_quad = th.dot(Jdelta, Cinv_Jdelta)

            base_lr = (
                self.actor_lr_schedule(self._current_progress_remaining)  # type: ignore[operator]
                if self.actor_lr_schedule is not None
                else self.lr_schedule(self._current_progress_remaining)
            )

            # L_old 仍然用当前 θ_k 上的 actor loss
            L_old = actor_loss.detach()
            max_fr_radius = self.pb_step_clip if self.pb_step_clip is not None else None
            self._last_actor_grad_flat = g_base.detach()

            # line search 的下降性检查仍然用当前点的梯度 g_base
            L_new, used_lr = self._pb_line_search(
                params=params,
                delta=search_delta,
                g_flat=g_base.detach(),
                observations=observations,
                actions=actions,
                advantages=adv,
                base_lr=base_lr,
                fr_quad=fr_quad.detach(),
                L_old=L_old,
                max_fr_radius=max_fr_radius,
                backtrack_factor=0.5,
                max_backtracks=10,
                armijo_coef=0.8,
            )
            actor_grad_norm = th.norm(g_base)

        return policy_loss.detach(), entropy_loss.detach(), actor_grad_norm, dnorm


    def _anpg_prox_inner_update(self, observations, actions, advantages):
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        assert self._actor_params is not None
        assert self.actor_optimizer is not None

        adv = advantages
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        with th.no_grad():
            t_old = self._statistic_components(observations, actions)
            feats_old = self._fr_features(t_old)
            mu_old = feats_old.mean(dim=0)

            batch_size, num_stats = feats_old.shape
            t_center = feats_old - mu_old.unsqueeze(0)
            C = (t_center.T @ t_center) / max(batch_size, 1)
            r = float(self.pb_c_ridge)
            C = C + r * th.eye(num_stats, device=C.device, dtype=C.dtype)
            C_inv = self._invert_spd(C)

        mu_old = mu_old.detach()
        C_inv = C_inv.detach()

        one_over_h = 1.0 / max(self.pb_h, 1e-12)

        last_policy_loss = None
        last_entropy_loss = None

        for _ in range(self.pb_inner_steps):
            self.actor_optimizer.zero_grad()

            values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
            values = values.detach()

            policy_loss = -(adv * log_prob).mean()
            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            t_cur = self._statistic_components(observations, actions)
            feats_cur = self._fr_features(t_cur)
            mu_cur = feats_cur.mean(dim=0)

            diff = mu_cur - mu_old
            quad = diff @ (C_inv @ diff)

            prox_loss = 0.5 * one_over_h * quad

            total_loss = policy_loss + self.ent_coef * entropy_loss + prox_loss

            total_loss.backward()
            if self.pb_inner_lr != 1.0:
                for p in self._actor_params:
                    if p.grad is not None:
                        p.grad.mul_(self.pb_inner_lr)

            th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm)
            self.actor_optimizer.step()

            last_policy_loss = policy_loss
            last_entropy_loss = entropy_loss

        with th.no_grad():
            grads = []
            for p in self._actor_params:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            if len(grads) > 0:
                actor_grad_norm = th.norm(th.cat(grads))
            else:
                actor_grad_norm = th.tensor(0.0, device=observations.device)

            t_cur = self._statistic_components(observations, actions)
            feats_cur = self._fr_features(t_cur)
            mu_cur = feats_cur.mean(dim=0)
            diff = mu_cur - mu_old
            fr_quad = diff @ (C_inv @ diff)
            dnorm = th.sqrt(fr_quad + 1e-12)

        assert last_policy_loss is not None and last_entropy_loss is not None
        return last_policy_loss.detach(), last_entropy_loss.detach(), actor_grad_norm, dnorm

    def _params(self):
        return [p for p in self.policy.parameters() if p.requires_grad]

    def _flat(self, tensors):
        return th.cat([t.reshape(-1) for t in tensors])

    def _get_flat(self, params):
        return th.cat([p.data.view(-1) for p in params])

    @th.no_grad()
    def _set_flat_(self, params, flat):
        off = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[off:off + n].view_as(p))
            off += n


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

    @th.no_grad()
    def _pb_line_search(
        self,
        params,
        delta: th.Tensor,
        g_flat: th.Tensor,
        observations,
        actions,
        advantages,
        base_lr: float,
        fr_quad: th.Tensor,
        L_old: th.Tensor,
        max_fr_radius: Optional[float] = None,
        backtrack_factor: float = 0.5,
        max_backtracks: int = 8,
        armijo_coef: float = 0.8,
    ):

        g_dot_delta = th.dot(g_flat, delta)
        if g_dot_delta >= 0:
            # 更新方向太怪，直接拒绝这次更新
            theta_flat_old = self._get_flat(params)
            self._set_flat_(params, theta_flat_old)
            return L_old, 0.0

        # 预期下降（基于一阶近似）
        expected_improve_full = -g_dot_delta * base_lr  # 对应 base_lr 的预期下降
        if expected_improve_full.item() <= 0:
            theta_flat_old = self._get_flat(params)
            self._set_flat_(params, theta_flat_old)
            return L_old, 0.0

        theta_flat_old = self._get_flat(params)

        lr = base_lr
        best_lr = 0.0
        L_best = L_old
        accepted = False

        for _ in range(max_backtracks + 1):
            if max_fr_radius is not None:
                fr_radius = 0.5 * (lr * lr) * fr_quad 
                if fr_radius.item() > max_fr_radius:
                    lr *= backtrack_factor
                    continue

            self._set_flat_(params, theta_flat_old)
            self._add_flat_(params, delta, alpha=lr)

            values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
            values = values.detach()

            adv = advantages
            if self.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            policy_loss_new = -(adv * log_prob).mean()
            if entropy is None:
                entropy_loss_new = -th.mean(-log_prob)
            else:
                entropy_loss_new = -th.mean(entropy)

            L_new = policy_loss_new + self.ent_coef * entropy_loss_new

            actual_improve = (L_old - L_new).item()

            # 对应当前 lr 的预期下降（线性缩放）
            expected_improve_lr = (lr / base_lr) * expected_improve_full.item()

            # Armijo 条件：实际下降要至少达到预期下降的 armijo_coef 倍
            if actual_improve > 0.0 and actual_improve >= armijo_coef * expected_improve_lr:
                accepted = True
                L_best = L_new
                best_lr = lr
                break

            # 否则缩小步长
            lr *= backtrack_factor

        if not accepted:
            # 不更新，回滚参数，并清空上一轮步长（中点预测下轮退化到当前点）
            self._set_flat_(params, theta_flat_old)
            self._pb_last_step_flat = None
            return L_old, 0.0

        # 确保参数已经在 best_lr 对应的位置
        if best_lr != lr:
            self._set_flat_(params, theta_flat_old)
            self._add_flat_(params, delta, alpha=best_lr)

        # 记录真实的参数步长 Δθ_k，用于下一轮中点预测
        theta_flat_new = self._get_flat(params)
        self._pb_last_step_flat = (theta_flat_new - theta_flat_old).detach().clone()

        return L_best, best_lr


    def _statistic_components(self, states, actions):
        if states.dim() == 1:
            states = states.unsqueeze(0)
        dist = self.policy.get_distribution(states)

        if isinstance(self.action_space, spaces.Box):
            # 连续动作：Gaussian policy
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

            if self.statistic == "action_mean":
                mu_a = dist.distribution.loc  
                if mu_a.dim() == 1:
                    mu_a = mu_a.unsqueeze(0)

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
            # 离散动作：Categorical policy
            if actions.dim() > 1:
                actions = actions.squeeze(-1)

            if self.statistic == "action_mean":
                probs = dist.distribution.probs  # shape [B, num_actions]
                return probs

            # ===== 原有分支 =====
            logp = dist.log_prob(actions.long()).view(-1, 1)

            if self.statistic == "logp":
                return logp

            if self.statistic == "entropy":
                ent = dist.entropy().view(-1, 1)
                return ent

            if self.statistic == "score_per_dim":
                probs = dist.distribution.probs
                num_actions = probs.shape[-1]
                one_hot = F.one_hot(actions.long(), num_classes=num_actions).float()
                return one_hot - probs

            raise ValueError(f"statistic '{self.statistic}' not implemented for discrete SB3 policy.")

    def _t_mean(self, states, actions):
        t = self._statistic_components(states, actions)
        feats = self._fr_features(t)
        return feats.mean(dim=0)

    def _fr_features(self, t: th.Tensor) -> th.Tensor:
        """
        t: [B, K_base]
        返回 pullback 用的特征 φ(s,a)。
        当 pb_use_kernel=True 时使用核特征；否则使用原来的 FR 多项式特征。
        """
        # ===== NEW: 核化分支 =====
        if self.pb_use_kernel:
            # 懒初始化 kernel anchors：用当前 batch 的 t 选一部分
            if self._kernel_anchors is None:
                with th.no_grad():
                    B, K_base = t.shape
                    M = min(self.pb_kernel_num_anchors, B)
                    idx = th.randperm(B, device=t.device)[:M]
                    self._kernel_anchors = t[idx].detach().clone()  # [M, K_base]

            anchors = self._kernel_anchors  # [M, K_base]
            B, K_base = t.shape
            M = anchors.shape[0]

            # pairwise 距离
            diff = t.unsqueeze(1) - anchors.unsqueeze(0)  # [B, M, K_base]
            sq_dist = (diff * diff).sum(dim=-1)           # [B, M]
            dist = th.sqrt(sq_dist + 1e-12)               # [B, M]

            # 自适应带宽：σ = σ0 * mean_distance（或固定 σ）
            if self.pb_kernel_auto_sigma:
                # mean squared distance 或 mean distance 都可以，这里用 mean distance
                mean_dist = dist.mean().detach()
                sigma = self.pb_kernel_sigma * (mean_dist + 1e-12)
            else:
                sigma = self.pb_kernel_sigma
            sigma2 = sigma * sigma + 1e-12

            ktype = self.pb_kernel_type

            if ktype == "rbf":
                # Gaussian / RBF: k(x,x') = exp(-||x-x'||^2 / (2σ^2))
                phi = th.exp(-0.5 * sq_dist / sigma2)     # [B, M]

            elif ktype == "rq":
                # Rational quadratic:
                # k(x,x') = (1 + ||x-x'||^2 / (2 α σ^2))^{-α}
                alpha = self.pb_kernel_alpha
                denom = 1.0 + sq_dist / (2.0 * alpha * sigma2)
                phi = denom.pow(-alpha)

            elif ktype == "laplace":
                # Laplacian: k(x,x') = exp(-||x-x'|| / λ)
                lam = sigma
                phi = th.exp(-dist / (lam + 1e-12))

            elif ktype in ("matern32", "matern_3_2"):
                # Matérn ν = 3/2:
                # k(r) = (1 + sqrt(3) r / ℓ) * exp(-sqrt(3) r / ℓ)
                ell = sigma
                coeff = th.sqrt(th.tensor(3.0, device=t.device, dtype=t.dtype))
                r_scaled = coeff * dist / (ell + 1e-12)
                phi = (1.0 + r_scaled) * th.exp(-r_scaled)

            elif ktype == "poly":
                # Polynomial: k(x,x') = (x^T x' + c)^p
                # 注意这里是对 feature t 做内积，而不是用距离
                alpha = self.pb_kernel_alpha  # 用作 degree p
                c = self.pb_kernel_c
                inner = (t.unsqueeze(1) * anchors.unsqueeze(0)).sum(dim=-1)  # [B, M]
                # 为了数值稳定，避免负数的非整数幂，这里只建议 alpha 是正整数
                phi = (inner + c).clamp(min=0.0).pow(alpha)

            else:
                raise ValueError(f"Unknown pb_kernel_type '{self.pb_kernel_type}'")

            # 只用一阶核特征
            if self.fr_order <= 1:
                return phi

            # 可选：在核特征上再加 FR 二阶项（注意维度会是 M + M^2）
            phi_outer = phi.unsqueeze(2) * phi.unsqueeze(1)  # [B, M, M]
            phi_quad = 0.5 * phi_outer.reshape(B, M * M)
            return th.cat([phi, phi_quad], dim=1)

        # ===== 原始 FR 多项式特征 =====
        if self.fr_order <= 1:
            return t
        B, K = t.shape
        t_outer = t.unsqueeze(2) * t.unsqueeze(1)
        t_quad = 0.5 * t_outer.reshape(B, K * K)
        return th.cat([t, t_quad], dim=1)

    @th.no_grad()
    def _compute_C_t_full(self, states, actions, ridge: Optional[float] = None):
        t = self._statistic_components(states, actions)
        feats = self._fr_features(t)
        batch_size, num_stats = feats.shape
        t_center = feats - feats.mean(dim=0, keepdim=True)
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

    def _build_J(self, states, actions, params):
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

    def _build_J_Cinv(self, states, actions, params):
        J = self._build_J(states, actions, params)
        C = self._compute_C_t_full(states, actions)
        C_inv = self._invert_spd(C)
        with th.no_grad():
            J = J.to(C_inv.dtype)
        return J, C_inv

    def _Aop_from_j_cinv(self, J, C_inv, one_over_h, lam):
        def Aop(v):
            with th.no_grad():
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
