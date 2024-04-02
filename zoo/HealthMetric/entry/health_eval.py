from zoo.health_metrics.config.health_metrics_muzero_config import main_config, create_config
from lzero.entry import eval_muzero
import numpy as np

if __name__ == "__main__":
    # Path to the pretrained model's checkpoint file, if available
    # model_path = './health_metrics_ckpt/ckpt_best.pth.tar'
    model_path = None

    # Initialize a list with seeds for the experiment, can add more seeds for comprehensive evaluation
    seeds = [0, 123, 456]

    # Set the number of episodes to run for each seed
    num_episodes_each_seed = 10

    # Adjust the configuration for the health metrics optimization environment
    main_config.env.env_id = 'health_metrics_optimization'
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = num_episodes_each_seed

    # Enable saving of replay, specify the path to save the replay gif for visualizing the agent's decisions
    # main_config.env.save_replay_gif = True
    # main_config.env.replay_path_gif = './health_metrics_video'

    # Initialize lists to store the mean and total returns for each seed
    returns_mean_seeds = []
    returns_seeds = []

    # Evaluate the model's performance for each seed and store the results
    for seed in seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],  # Updated configuration parameters for the health metrics environment
            seed=seed,  # The seed for the random number generator
            num_episodes_each_seed=num_episodes_each_seed,  # The number of episodes to run for this seed
            print_seed_details=True,  # Toggle to print detailed information for each seed
            model_path=model_path  # The path to the trained model, if available
        )
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    # Convert the lists of returns to numpy arrays for statistical analysis
    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    # Print the evaluation results
    print("=" * 20)
    print(f"We evaluated {len(seeds)} seeds with {num_episodes_each_seed} episodes each.")
    print(f"For seeds {seeds}, the mean returns are {returns_mean_seeds}, and the returns per episode are {returns_seeds}.")
    print("Across all seeds, the mean reward is:", returns_mean_seeds.mean())
    print("=" * 20)
