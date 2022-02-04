from model import get_actor, get_critic 
import tensorflow as tf 

def initialise_globals(): 
    global actor_model, critic_model, target_actor, target_critic, critic_optimizer, actor_optimizer, gamma
    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())


    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001
    gamma = 0.99

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)