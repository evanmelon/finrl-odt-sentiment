"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import os
from gymnasium.envs.registration import register

from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
# import gym
import gymnasium as gym
# import d4rl
import torch
import numpy as np
import pandas as pd

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS

MAX_EPISODE_LEN = 1000


train = pd.read_csv('./evan/train_data.csv')
train = train.set_index(train.columns[0])
train.index.names = ['']
trade = pd.read_csv('./evan/trade_data.csv')
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']
stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# 訓練用 env：使用 train 資料
def make_stock_trading_env_train(**kwargs):
    return StockTradingEnv(
        df=train.copy(),
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs,
        **kwargs,
    )

# 測試用 env：使用 test 資料
def make_stock_trading_env_test(**kwargs):
    return StockTradingEnv(
        df=trade.copy(),
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs,
        **kwargs,
    )
# 定義工廠函數
def make_stock_trading_env(**kwargs):
    return StockTradingEnv(
        df=trade.copy(),
        turbulence_threshold=70,
        risk_indicator_col='vix',
        **env_kwargs,
        **kwargs,
    )

# 註冊環境
register(
    id='trajectories_train_sac-v0',
    entry_point=make_stock_trading_env_train,
)
register(
    id='StockTradingTrain-v0',
    entry_point=make_stock_trading_env_train,
)

register(
    id='StockTradingTest-v0',
    entry_point=make_stock_trading_env_test,
)


class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

        # sentiment_lookup
        with open('./sentiment_data/daily_sentiment_list.pkl', 'rb') as f:
            daily_sentiment_data = pickle.load(f)

        self.sentiment_lookup = {
            pd.to_datetime(row[0]).strftime('%Y-%m-%d'): np.array(row[1])
            for row in daily_sentiment_data.values
        }

    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def get_env_builder(self, seed, env_name, target_goal=None):
        def make_env_fn():
            # import d4rl

            env = gym.make(env_name)
            # env.seed(seed)
            # if hasattr(env.env, "wrapped_env"):
            #     env.env.wrapped_env.seed(seed)
            # elif hasattr(env.env, "seed"):
            #     env.env.seed(seed)
            # else:
            #     pass
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            if target_goal:
                env.set_target_goal(target_goal)
                print(f"Set the target goal to be {env.target_goal}")
            return env

        return make_env_fn

    def _augment_trajectories(
        self,
        online_envs,
        # eval_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                collect_asset_memory=False,
                sentiment_lookup=self.sentiment_lookup,
                # stock_dim=eval_envs.envs[0].stock_dim,
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns, stage="pretrain")
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

    def save_asset_memories_csv(self, asset_memories, save_path="./asset_values.csv"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        max_len = max(len(mem) for mem in asset_memories)
        df = pd.DataFrame({
            f"env_{i}": mem + [np.nan] * (max_len - len(mem))
            for i, mem in enumerate(asset_memories)
        })
        df.to_csv(save_path, index=False)

    def save_metrics_csv(self, metrics: dict, save_path="./metrics.csv"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame([metrics])
        df.to_csv(save_path, index=False)

    def evaluate(self, eval_fns, stage="pretrain"):
        def compute_financial_metrics(asset_memories, risk_free_rate=0.0):
            cumulative_returns = []
            sharpes = []
            max_drawdowns = []

            for mem in asset_memories:
                mem = np.array(mem)
                daily_returns = np.diff(mem) / mem[:-1]

                # 累積報酬
                cumulative_returns.append((mem[-1] / mem[0]) - 1)

                # 夏普
                excess = daily_returns - risk_free_rate
                sharpe = np.mean(excess) / (np.std(excess) + 1e-8) if len(excess) > 1 else 0.0
                sharpes.append(sharpe * np.sqrt(252))  # 年化

                # 最大回撤
                peak = np.maximum.accumulate(mem)
                drawdown = (peak - mem) / peak
                max_drawdowns.append(np.max(drawdown))

            return {
                "evaluation/financial_cumulative_return": np.mean(cumulative_returns),
                "evaluation/financial_sharpe": np.mean(sharpes),
                "evaluation/financial_max_drawdown": np.mean(max_drawdowns),
            }
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o, asset_memories = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        # asset_memories = outputs[f"evaluation/asset_memory_gm"]

        metrics = compute_financial_metrics(asset_memories)
        outputs.update(metrics)

        save_dir = self.logger.log_path
        asset_path = os.path.join(save_dir, f"asset_values_{stage}.csv")
        metrics_path = os.path.join(save_dir, f"metrics_{stage}.csv")
        self.save_asset_memories_csv(asset_memories, asset_path)
        self.save_metrics_csv(metrics, metrics_path)
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                # eval_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                # def compute_financial_metrics(asset_memories, risk_free_rate=0.0):
                #     cumulative_returns = []
                #     sharpes = []
                #     max_drawdowns = []
                #
                #     for mem in asset_memories:
                #         mem = np.array(mem)
                #         daily_returns = np.diff(mem) / mem[:-1]
                #
                #         # 累積報酬
                #         cumulative_returns.append((mem[-1] / mem[0]) - 1)
                #
                #         # 夏普
                #         excess = daily_returns - risk_free_rate
                #         sharpe = np.mean(excess) / (np.std(excess) + 1e-8) if len(excess) > 1 else 0.0
                #         sharpes.append(sharpe * np.sqrt(252))  # 年化
                #
                #         # 最大回撤
                #         peak = np.maximum.accumulate(mem)
                #         drawdown = (peak - mem) / peak
                #         max_drawdowns.append(np.max(drawdown))
                #
                #     return {
                #         "evaluation/financial_cumulative_return": np.mean(cumulative_returns),
                #         "evaluation/financial_sharpe": np.mean(sharpes),
                #         "evaluation/financial_max_drawdown": np.mean(max_drawdowns),
                #     }
                eval_outputs, eval_reward = self.evaluate(eval_fns, stage=f"online_{self.online_iter}")
                outputs.update(eval_outputs)
                # # ✅ 額外建立 DummyVecEnv 跑一次 rollout，拿 asset_memory
                # dummy_eval_envs = DummyVecEnv([
                #     self.get_env_builder(i + 1000, env_name="StockTradingTest-v0")  # seed 避免重複
                #     for i in range(self.variant["num_online_rollouts"])
                # ])
                # dummy_eval_fns = [
                #     create_vec_eval_episodes_fn(
                #         vec_env=dummy_eval_envs,
                #         eval_rtg=self.variant["eval_rtg"],
                #         state_dim=self.state_dim,
                #         act_dim=self.act_dim,
                #         state_mean=self.state_mean,
                #         state_std=self.state_std,
                #         device=self.device,
                #         use_mean=True,
                #         reward_scale=self.reward_scale,
                #     )
                # ]
                # eval_out, asset_memories = dummy_eval_fns[0](self.model)
                # metrics = compute_financial_metrics(asset_memories)
                # outputs.update(metrics)
                # dummy_eval_envs.close()
                # save_dir = self.logger.log_path
                # self.save_asset_memories_csv(asset_memories, os.path.join(save_dir, f"asset_memory_online_{self.online_iter}.csv"))
                # self.save_metrics_csv(metrics, os.path.join(save_dir, f"metrics_online_{self.online_iter}.csv"))

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

            self.online_iter += 1

    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        # import d4rl
        get_env_builder = self.get_env_builder

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
            sentiments,
            sentiment_weight=0.1,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)  # shape: (B, T)
            entropy = a_hat_dist.entropy().mean()

            # 將 [pos, neg, neu] 壓成 scalar，加權情緒評分
            # sentiments: shape (B, T, tick, 3)
            weights = torch.tensor([1.0, -1.0, 0.0], dtype=sentiments.dtype, device=sentiments.device)  # shape: (3,)
            sentiment_score_per_tick = torch.sum(sentiments * weights, dim=-1)  # shape: (B, T, tick)
            sentiment_score = sentiment_score_per_tick.mean(dim=-1)  # shape: (B, T)
            sentiment_score = sentiment_score.to(log_likelihood.device)


            # 產生 mask，過濾掉 padding
            sentiment_mask = attention_mask > 0

            # 損失項：鼓勵在正向情緒高時，行動機率高（log_prob 大）
            sentiment_loss = -(log_likelihood * sentiment_score)[sentiment_mask].mean()

            # 原始 loss: log_likelihood + entropy 正則
            ll_term = log_likelihood[sentiment_mask].mean()
            loss = -(ll_term + entropy_reg * entropy) + sentiment_weight * sentiment_loss

            return loss, -ll_term, entropy


        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = DummyVecEnv(
            [
                get_env_builder(i, env_name="StockTradingTest-v0", target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name="StockTradingTrain-v0", target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
