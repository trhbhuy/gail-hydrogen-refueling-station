import os
import argparse
import numpy as np
import logging

import stable_baselines3
import stable_baselines3 as sb3
from imitation.policies.serialize import load_stable_baselines_model
from solver.platform.test_env import HydrogenEnv
from utils.test_util import cal_metric, load_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """
    Parse command-line arguments for the testing script.
    """
    parser = argparse.ArgumentParser(description='Arguments for testing the trained model.')
    parser.add_argument('--num_test_scenarios', type=int, default=91, help='Number of test scenarios')
    parser.add_argument('--data_path', type=str, default='data/processed/ObjVal.csv', help='Path to the test dataset')
    parser.add_argument('--adversarial_algo', type=str, choices=['gail', 'airl'], default='gail', help='Adversarial algorithm used during training')
    parser.add_argument('--gen_algo', type=str, choices=['ppo', 'ddpg', 'trpo'], default='ppo', help='Generator algorithm used during training')
    parser.add_argument('--learning_rate', type=float, default=0.00015, help='Learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps for training')
    parser.add_argument('--verbose', action='store_true', default=True, help='Whether to print detailed logs during testing')
    parser.add_argument('--is_cuda', action='store_true', help='Use CUDA for testing if available')

    return parser.parse_args()

def load_model(args, env):
    """
    Initialize and load the model with pretrained weights.
    """
    algo_cls = getattr(stable_baselines3, args.gen_algo.upper(), None)

    if algo_cls is None:
        raise ValueError(f"Algorithm '{args.gen_algo}' is not supported.")

    model_name = f'{args.adversarial_algo}_{args.gen_algo}_lr{args.learning_rate}_bs{args.batch_size}_{args.n_steps}n_steps'
    pretrained_path = os.path.join(BASE_DIR, 'logs', model_name, 'checkpoints', 'final', 'gen_policy')

    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

    model = load_stable_baselines_model(algo_cls, pretrained_path, env)

    if args.verbose:
        logging.info(f"Loaded {args.gen_algo.upper()} model from {pretrained_path}")

    return model

def inference(args, model, env):
    """
    Perform inference using the provided model in the specified environment.
    """
    # Containers for aggregated rewards and episode information
    aggregated_rewards = []
    episode_info = []

    num_scenarios = args.num_test_scenarios

    # Evaluate the model for each day
    for scenario_idx in range(num_scenarios):
        state, info = env.reset(scenario_idx)
        total_reward = 0

        while True:
            # Predict the action using the model
            action, _ = model.predict(state, deterministic=True)

            # Step the environment with the predicted action
            next_state, reward, terminated, _, info = env.step(action)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            episode_info.append(info)
            
            # Break if the episode has terminated
            if terminated:
                break
        
        aggregated_rewards.append(total_reward)

    return np.array(aggregated_rewards), np.array(episode_info)

def evaluate(args, model, env, best_rewards):
    """
    Evaluate the models and calculate metrics based on predictions vs actual rewards.
    """
    # Perform inference with the model
    predicted_rewards, inference_info = inference(args, model, env)
    
    # Calculate evaluation metrics (e.g., MAE, MAPE) based on the true values and predictions
    metrics = cal_metric(best_rewards, predicted_rewards)
    
    # Print the evaluation results
    logging.info(f"Overall MAE: {metrics['overall_mae']:.4f}, Overall MAPE: {metrics['overall_mape']:.4f}%")
    
    return metrics, inference_info

def test(args):
    """
    Test the model by evaluating its predictions against the best rewards.
    """
    # Load the actual rewards (ground truth) from the dataset
    best_rewards = load_dataset(args)

    # Set up the environment
    env = HydrogenEnv(is_train=False)

    # Load the pre-trained model with the specified configuration
    model = load_model(args, env)

    # Evaluate the model's predictions against the actual rewards
    metrics, inference_info = evaluate(args, model, env, best_rewards)

    return metrics, inference_info

#python3 src/test_gail.py --num_test_scenarios 90 --adversarial_algo gail --gen_algo ppo --learning_rate 0.00015 --batch_size 8 --n_steps 2048
if __name__ == '__main__':
    args = parse_args()
    test(args)