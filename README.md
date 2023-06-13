This repo is a quick storage area for a demo of PPO on the LunarLander provided through Gymnassium. 

---

The ***.pt** files are PyTorch models that have already been trained, the **results.pn**g shows how the training & testing rewards changed during training of the agent.

---

**ppo.py** will train the agents on the LunarLander-v2 environment, saving the reward graph when training completes. The **ppo_base.pt** is created just after the model is initialized, so in effect it knows nothing. While training, if the agent reaches a average reward of 110+ over 25 episodes then it saves the current model to **ppo_partial.pt**. If the model is able to achieve 220+ reward over 25 episodes then it saves the current model to **ppo_trained.pt** and exits training.

---

**demo.py** will create a *demo_videos* directory, under which it will create 3 other directories, matching the naming scheme of the model's checkpoints. Each of these subdirectories will have 5 demos using that agent, saved as .mp4 files in their own folders. 

---

It appears that gymnassium requires [this](https://github.com/pygame/pygame/issues/3260#issuecomment-1288123732) fix in order to run, perhaps in general but maybe just for WSL.