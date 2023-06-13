import torch
import gymnasium as gym
import torch.nn.functional as F
from gymnasium.utils.save_video import save_video
from ActorCritic import ActorCritic, MLP


def demo(env, policy, model_type, episode = 0):
    '''Helper function to save a gif of the agent running through the environment'''
    policy.eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state, info = env.reset()
    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
        
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)
                
        action = torch.argmax(action_prob, dim = -1)
        state, reward, terminated, truncated, info = env.step(action.item())

        if terminated or truncated:
            done = True

        episode_reward += reward

    save_video(env.render(), 
               f'demo_videos/{model_type}/{episode}', 
               fps = env.metadata["render_fps"])
        
    return episode_reward


def main():
    env = gym.make('LunarLander-v2', render_mode = 'rgb_array_list')

    INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = env.observation_space.shape[0], 128, env.action_space.n

    agent = ActorCritic(actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM), 
                        critic = MLP(INPUT_DIM, HIDDEN_DIM, 1))
    
    for model in ['ppo_base.pt', 'ppo_partial.pt', 'ppo_trained.pt']:
        agent.load_state_dict(torch.load(model))

        num_demos = 5
        for i in range(num_demos):
            demo(env, agent, model[:model.find('.')], i)

if __name__ == '__main__':
    main()