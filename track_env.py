import gym
import numpy as np
import pygame
from gym import spaces

class TrackEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(TrackEnv, self).__init__()
        
        # Observation space: [x_position, y_position]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([600, 400]), dtype=np.float32)
        
        # Action space: [up, down, left, right]
        self.action_space = spaces.Discrete(4)
        
        self.position = np.array([50.0, 200.0])  # Start at the left side of the track
        self.track_center = 200.0  # Straight line center (y-coordinate)
        self.track_width = 50.0  # Boundary of the track
        self.finish_line = 550.0  # Finish line (x-coordinate)

        self.render_mode = render_mode
        self.window_size = (600, 400)

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Straight Line Track")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.position = np.array([50.0, 200.0])
        if self.render_mode == "human":
            self._render_frame()
        return self.position

    def step(self, action):
        # Action effects
        if action == 0:  # Move up
            self.position[1] -= 5
        elif action == 1:  # Move down
            self.position[1] += 5
        elif action == 2:  # Move left
            self.position[0] -= 5
        elif action == 3:  # Move right
            self.position[0] += 5
        
        # Calculate rewards
        reward = 0.0
        if self.track_center - self.track_width <= self.position[1] <= self.track_center + self.track_width:
            reward += 1.0  # Reward for staying within the track
        else:
            reward -= 10.0  # Penalty for going out of bounds

        # Bonus for moving forward
        reward += 0.1 * (self.position[0] - 50.0)  # Positive reward for moving right

        # Check for termination
        done = self.position[0] >= self.finish_line or not (0 <= self.position[1] <= 400)

        if self.render_mode == "human":
            self._render_frame()

        # Information for debugging
        info = {"position": self.position, "reward": reward}

        return self.position, reward, done, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # Clear screen
        self.screen.fill((0, 0, 0))

        # Draw track boundaries
        pygame.draw.line(self.screen, (0, 0, 255), (0, self.track_center - self.track_width), (600, self.track_center - self.track_width), 2)
        pygame.draw.line(self.screen, (0, 0, 255), (0, self.track_center + self.track_width), (600, self.track_center + self.track_width), 2)

        # Draw finish line
        pygame.draw.line(self.screen, (255, 0, 0), (self.finish_line, 0), (self.finish_line, 400), 3)

        # Draw agent
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.position[0]), int(self.position[1])), 5)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
