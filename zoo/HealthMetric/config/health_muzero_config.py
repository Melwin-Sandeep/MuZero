from easydict import EasyDict

env_id = 'health_metrics_optimization'  # New environment ID
memory_length = 30

max_env_step = int(10e6)

# Basic configuration settings
seed = 0
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
reanalyze_ratio = 0
td_steps = 5

policy_entropy_loss_weight = 1e-4
threshold_training_steps_for_final_temperature = int(5e5)
eps_greedy_exploration_in_collect = False

health_metrics_muzero_config = dict(
    exp_name=f'health_metrics_optimization/ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed{seed}',
    env=dict(
        stop_value=100,  # Hypothetical stop value for an episode where health is optimally improved
        env_id=env_id,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False,),
        # New configuration for health metrics (blood pressure, pulse, temperature)
        health_metric_ranges=dict(
            bp_systolic=(80, 160),
            bp_diastolic=(50, 100),
            pulse=(40, 120),
            temperature=(35.0, 39.0),
        ),
    ),
    policy=dict(
        model=dict(
            observation_shape=4,  # Adjusted for bp_systolic, bp_diastolic, pulse, temperature
            action_space_size=3,  # Example: Improve BP, Increase Pulse, Decrease Temperature
            model_type='mlp',
            latent_state_dim=128,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        td_steps=td_steps,
        cuda=True,
        env_type='health_metrics',
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        learning_rate=0.003,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

health_metrics_muzero_config = EasyDict(health_metrics_muzero_config)
main_config = health_metrics_muzero_config

health_metrics_muzero_create_config = dict(
    env=dict(
        type='health_metrics_lightzero',
        import_names=['zoo.health_metrics.envs.health_metrics_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
health_metrics_muzero_create_config = EasyDict(health_metrics_muzero_create_config)
create_config = health_metrics_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)
