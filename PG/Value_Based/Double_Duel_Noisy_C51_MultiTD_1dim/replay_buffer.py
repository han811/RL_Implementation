
import numpy as np

class ReplayBuffer:

    def __init__(self, 
                 buffer_size: int, 
                 input_dim: tuple, 
                 batch_size: int):

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.save_count, self.current_size = 0, 0

        self.state_buffer = np.ones((buffer_size, input_dim), dtype=np.float32)
        self.next_state_buffer = np.ones((buffer_size, input_dim), dtype=np.float32) 
        self.action_buffer = np.ones(buffer_size, dtype=np.uint8) 
        self.reward_buffer = np.ones(buffer_size, dtype=np.float32) 
        self.done_buffer = np.ones(buffer_size, dtype=np.uint8) 

    def __len__(self):
        return self.current_size

    def store(self, 
              state: np.ndarray, 
              action: int, 
              reward: float, 
              next_state: np.ndarray, 
              done: int):

        self.state_buffer[self.save_count] = state
        self.action_buffer[self.save_count] = action
        self.reward_buffer[self.save_count] = reward
        self.next_state_buffer[self.save_count] = next_state
        self.done_buffer[self.save_count] = done
        
        self.save_count = (self.save_count + 1) % self.buffer_size
        self.current_size = min(self.current_size+1, self.buffer_size)

    def batch_load(self):
        indices = np.random.randint(self.current_size, size=self.batch_size)
        return dict(
                states=self.state_buffer[indices], 
                actions=self.action_buffer[indices],
                rewards=self.reward_buffer[indices],
                next_states=self.next_state_buffer[indices], 
                dones=self.done_buffer[indices]) 
    