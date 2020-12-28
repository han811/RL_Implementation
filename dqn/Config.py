#########################
### han811 - 20.12.28 ###
#########################

config = dict()

# name of gym env
# Breakout-v0 : 25% previous action
# Breakout-v4 : 0% previous action
# Deterministic : frameskip = 4
# NoFrameskip-v4 : no frameskip
config['env_name'] = 'BreakoutDeterministic-v4'

# memory size
config['memory_size'] = 1000000

# image size
config['height'] = 84
config['width'] = 84

# reward clip
config['reward_clip'] = True

# episode
config['episode_num'] = 1000000

# discount factore
config['discount_factor'] = 0.99

# k
config['k'] = 4

# batch size
config['batch_size'] = 32

