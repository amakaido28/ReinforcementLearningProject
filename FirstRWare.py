import gym
import numpy as np
import tensorflow as tf
import rware
#from robotic_warehouse.gym_envs import RwareSmallEnv

from actor_critic import Actor, Critic

# Definire la funzione di loss per l'aggiornamento della politica di actor
def actor_loss(probabilities, actions, advantages):
    neg_log_prob = tf.reduce_sum(-tf.math.log(probabilities)*tf.one_hot(actions, action_size), axis=1)
    loss = tf.reduce_mean(neg_log_prob * advantages)
    return loss

# Definire la funzione di loss per l'aggiornamento della funzione di valore di critic
def critic_loss(values, returns):
    loss = tf.reduce_mean(tf.square(values - returns))
    return loss

# Definire l'ambiente rware e le costanti per il numero di episodi e di passi massimi
#env = RwareSmallEnv()
env=rware.Warehouse(9,1,5,3,2,1,3,5,7,rware.RewardType.GLOBAL)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

num_episodes = 2000
max_steps_per_episode = 500

# Inizializzare la rete neurale per la politica di actor e la funzione di valore di critic
actor = Actor(state_size, action_size)
critic = Critic(state_size)

# Definire l'ottimizzatore per l'aggiornamento dei modelli
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Definiamo il ciclo di apprendimento
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        for step in range(max_steps_per_episode):
            # Ottenere l'azione dalla politica di actor
            action_probabilities = actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            action = np.random.choice(range(action_size), p=np.squeeze(action_probabilities))
            
            # Eseguire l'azione nell'ambiente e ottenere la ricompensa e il nuovo stato
            next_state, reward, done, _ = env.step(action)
            
            # Aggiungere la ricompensa all'episodio e calcolare l'errore di predizione della funzione di valore di critic
            value = critic(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
            next_value = critic(tf.convert_to_tensor(next_state[None, :], dtype=tf.float32))
            td_error = reward + 0.99*next_value - value
            
            # Calcolare la loss per l'aggiornamento della politica di actor e della funzione di valore di critic
            log_prob = tf.math.log(action_probabilities[0, action])
            entropy = -tf.reduce_sum(action_probabilities * tf.math.log(action_probabilities))
            actor_loss_value = -log_prob * td_error - 0.01 * entropy
            critic_loss_value = critic_loss(value, reward + 0.99*next_value)
            
            # Aggiungere la loss all'episodio
            episode_reward += reward
            
            # Aggiornare i gradienti per la politica di actor e la funzione di valore di critic
            actor_grads = actor_tape.gradient(actor_loss_value, actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss_value, critic.trainable_variables)
            
            # Aggiornare i modelli usando gli optimizer
            optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            state = next_state

            if done:
                break
        print("Episode:", episode+1, "Reward:", episode_reward)

# Chiudiamo l'ambiente
env.close()









#import gym
#import rware

#layout = """
#.......
#..xx...
#..x.x..
#.x...x.
#..x.x..
#...x...
#.g...g.
#"""
#env=rware.Warehouse(9,1,5,3,2,1,3,5,7,rware.RewardType.GLOBAL)
#episodes=100000
#for episode in range(1,episodes+1): 
#    state=env.reset()
#    done =False
#    score=0
#    while not done:
#        env.render("human")
#        actions = env.action_space.sample()
#        n_state,reward,done,info=env.step(actions)
#        done=False
#        #score+=reward
    
#env.action_space.sample()

#while True:
#    print()