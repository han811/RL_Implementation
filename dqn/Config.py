#########################
### han811 - 20.12.28 ###
#########################

config = dict()

# name of gym env
# Breakout-v0 : 25% previous action
# Breakout-v4 : 0% previous action
# Deterministic : frameskip = 4
# NoFrameskip-v4 : no frameskip
# BreakoutDeterministic-v4 : automatic frameskip
config['env_name'] = 'BreakoutDeterministic-v4'

# memory size
config['memory_size'] = 100000

# image size
config['height'] = 84
config['width'] = 84
config['state_size'] = (config['height'], config['width'])

# action size
config['action_size'] = 3

# reward clip
config['reward_clip'] = True

# episode
config['episode_num'] = 1000000

# discount factore
config['discount_factor'] = 0.99

# k - frameskip
config['k'] = 4

# batch size
config['mini_batch_size'] = 32

# history length
config['history_length'] = 4

# target model update frequency
config['target_model_update_freq'] = 400

# learning rate
config['learning_rate'] = 0.0001

# start learning step
config['start_learning_step'] = 25000

