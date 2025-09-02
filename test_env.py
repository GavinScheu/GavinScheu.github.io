from track_env import TrackEnv

# Initialize the environment
env = TrackEnv()

# Reset the environment and get the initial observation
obs = env.reset()

# Run a test loop with random actions
for _ in range(10):  # 10 steps
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, info = env.step(action)  # Perform the action
    env.render()  # Print ball position
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
    if done:
        obs = env.reset()
