from track_env import TrackEnv
from stable_baselines3 import PPO

# Hyperparameters
learning_rate = 0.000758
clip_range = 0.317926
ent_coef = 0.009135
n_steps = 2048

# Create environment and model
env = TrackEnv(render_mode=None)  # No rendering for training
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=n_steps,
    learning_rate=learning_rate,
    clip_range=clip_range,
    ent_coef=ent_coef
)

# Training Phase
print("Starting training...")
model.learn(total_timesteps=100000)
print("Training complete!")

# Testing Phase with Rendering
print("Starting testing phase with rendering...")
test_env = TrackEnv(render_mode="human")
obs = test_env.reset()

for step in range(100):  # Number of steps for testing
    action, _ = model.predict(obs)
    obs, reward, done, _ = test_env.step(action)
    test_env.render()

    print(f"Step: {step + 1} | Action: {action} | Reward: {reward:.2f} | Position: {obs}")

    if done:
        print(f"Step: {step + 1} | AI went out of bounds or finished. Resetting...")
        obs = test_env.reset()

test_env.close()
