
import gym
import rware

layout = """
.......
..xx...
..x.x..
.x...x.
..x.x..
...x...
.g...g.
"""
env=rware.Warehouse(9,1,1,1,2,0,3,5,7,rware.RewardType.GLOBAL)
episodes=100
for episode in range(1,episodes+1): 
    state=env.reset()
    done =False
    score=0
    while not done:
        env.render("human")
        actions = env.action_space.sample()
        n_state,reward,done,info=env.step(actions)
        done=False
        #score+=reward
   
env.action_space.sample()

while True:
    print()