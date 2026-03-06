import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import flax.serialization as ser
import argparse
import optax
import copy
from pathlib import Path
from jax import checkpoint as remat

from learning_opacity import init, compute_visual_states, to_binary, q_values, apply_action, load_checkpoint, check_collision

plt.rcParams['font.size'] = '16'

# discounting for Q-Learning
discount_factor = 1
# number of agents in a simulation
N = 2
# length of time of simulation
T = 250
# number of training simulations
simulations = 10
#batch size for loss calculation
batch_size = 128
# number of passes through the dataset
num_epochs = 32
#minimum Temperature
T_min = 0.05
# starting Temperature
T_0 = 20
# how quickly the simulation cools off
decay = 0.1
# Number of Sensors
n_s = 50
#fixed parameters for the turning angle
delta_theta = 0.2
#fixed parameter for the velocity
v0 = 10.0
#fixed parameter for the change in speed
dv = 2.0
#fixed parameter for the tree depth
tau = 4
#fixed parameter for sensor activation
opacity_threshold = 0.5
#fixed parameter for time step
dt = 1
# fixed parameter for number of frames
memory_length = 3
# fixed parameter for total number of actions
num_actions = 5
# the bounds of the sensors
sensor_bounds = jnp.array([2*jnp.pi/n_s * i for i in range(n_s+1)])  # length N+1
# the possible velocities
velocity = jnp.array([v0,v0,v0,v0+dv,v0-dv])
# the possible re-orientations
orientation = jnp.array([delta_theta, 0.0, -delta_theta, 0.0, 0.0])

def plot_trajectory(x,y, circle = False, temp = 0):
    fig, ax = plt.subplots()
    ax.plot(x,y, '-')
    if circle:
        for t in range(len(x)):
            for j in range(len(x[0])):
                circle = plt.Circle((x[t,j], y[t,j]), 1.0)
                ax.add_patch(circle)
    plt.gca().set_aspect('equal')

    plt.title(r"$T = $" + str(temp))
    plt.savefig(str(temp) + ".png")
    plt.close()

def produce_frames(xs, ys, os, vs, N, tMax):
    xvs = [vs[i] * jnp.cos(o) for i,o in enumerate(os)]
    yvs = [vs[i] * jnp.sin(o) for i,o in enumerate(os)]

    for t in range(tMax):
        fig, ax = plt.subplots()

        ax.quiver(xs[t,:], ys[t,:], xvs[t], yvs[t], color = 'k')

        for j in range(N):
            circle = plt.Circle((xs[t, j], ys[t, j]), 1.0,
                                color='blue', fill=False, linewidth=1.5)
            ax.add_patch(circle)

        # plt.plot(xs[:t+1], ys[:t+1], '--')

        plt.gca().set_aspect('equal')

#         plt.axis('off')
        plt.xlim(-100,200)
        plt.ylim(-50,250)

        plt.savefig("VideoPhotos/img" + "%03d" % t + ".jpg")

        plt.close()

def boltzman_selection(q_values, key, temp):

    def exponentiate(q_value):
        return jnp.exp(q_value/temp)

    proportions = jax.vmap(exponentiate)(q_values)
    probabilities = proportions/jnp.sum(proportions)

    action = jax.random.choice(key, a = jnp.arange(0,num_actions), p = probabilities)

    return action

def run_NN(key, N, T, QNN_params, temp):
    """
    Runs one simulation of N agents for T Time
    QNN: The network used for the heuristic
    """
    # set up the agents
    x0, y0, o0, v0 = init(key, N)

    init_frames = jnp.zeros((N, memory_length, n_s), dtype=jnp.int32)
    carry0 = (key, x0, y0, o0, v0, init_frames, QNN_params)

    def step(carry, t):
        key, x, y, o, v, prev_frames, params = carry

        key, subkey = jax.random.split(key)

        # get current visual states
        vis_state_integers = jnp.array(compute_visual_states(x, y, o))   # (N,)
        curr_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(vis_state_integers)

        opacity = jnp.sum(curr_vis_states)/ (N * n_s)

        def update_frames(prev_frames, curr_vis_states):
            retained_frames = prev_frames[:,1:,:]
            return jnp.concatenate([retained_frames, curr_vis_states[:,None,:]], axis = 1)

        curr_frames = update_frames(prev_frames, curr_vis_states)

#         jax.debug.print("{}", curr_frames.shape)

        # 2) Q-values for each agent
        agents_q_values = q_values(QNN_params, curr_frames)

        agent_keys = jax.random.split(subkey, N)   # N keys

        def select_action(qv, k):

            return boltzman_selection(qv, k, temp)

        actions = jax.vmap(select_action)(agents_q_values, agent_keys)    # (N,)

        x1, y1, o1, v1 = apply_action(x, y, o, actions)


        next_vis_state_integers = jnp.array(compute_visual_states(x1, y1, o1))   # (N,)
        next_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(next_vis_state_integers)


        rewards = -4*((jnp.sum(next_vis_states, axis=1)/n_s - 0.5) ** 2) + 1

        # update positions

        def collision_fn(i):
            x_roll, y_roll = jnp.roll(x, -i), jnp.roll(y, -i)
            x1_roll, y1_roll = jnp.roll(x1, -i), jnp.roll(y1, -i)

            return check_collision(x_roll[0],y_roll[0], x1_roll[0],y1_roll[0], jnp.stack([x_roll[1:],y_roll[1:]], axis=-1), jnp.stack([x1_roll[1:], y1_roll[1:]], axis=-1))

        collisions = jax.vmap(collision_fn)(jnp.arange(N))

        rewards = jnp.mean(jnp.where(collisions == 1, -1.0, rewards))

        transition = (x1, y1, o1, v1, opacity, rewards)

        new_carry = (key, x1, y1, o1, v1, curr_frames, params)

        return new_carry, transition

    carry_final, transitions = jax.lax.scan(step, carry0,jnp.arange(T),)

    return transitions   # see shapes below


def calculate_order(orientations, T = 1):
    x = 0
    y = 0
    N = len(orientations[-1])

    for i in range(N):
        for j in range(T):
            x += jnp.cos(orientations[-j-1][i])
            y += jnp.sin(orientations[-j-1][i])

    order = jnp.sqrt(x*x + y*y)/(N*T)
    return order

def calculate_COM(xs, ys, T = 1):
    x_COM = 0
    y_COM = 0
    N = len(xs[-1])

    for i in range(N):
        for j in range(T):
            x_COM += xs[-j-1][i]
            y_COM += ys[-j-1][i]

    return x_COM/(N*T), y_COM/(N*T)

def COM_distance(xs,ys, T = 1):
    x_COM, y_COM = calculate_COM(xs, ys, T)
    return jnp.sqrt(x_COM ** 2 + y_COM ** 2)


def evaluate_NN(key, sims, params):
    orders = 0
    COM_dist = 0
    visual_state = jnp.zeros((1, n_s))
    for i in range(sims):
        key, subkey = jax.random.split(key)

        data = run_NN(subkey, N, T, params, T_min)

        COM_dist += COM_distance(data[0], data[1], 10)

        orders += calculate_order(data[2], 10)

    return orders/sims, COM_dist/sims, q_zeros
#
# def plot_opacity(key, simulations, save_dir):
#
#     for cycle in range(simulations):


def plot_Q0(simulations, save_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    q_zeros = []
    q_ones = []

    dummy_zeros = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
    dummy_ones = jnp.ones((1, memory_length, n_s), dtype=jnp.float32)

    for cycle in range(simulations):

        if cycle % 10 == 0:
            print ("Starting Cycle " + str(cycle))

        save_dir = Path(save_dir)

        chk = load_checkpoint(save_dir, cycle)

        online_params = chk["online_params"]
        target_params = chk["target_params"]
        agents_q_values_0 = q_values(online_params, dummy_zeros)
        agents_q_values_1 = q_values(online_params, dummy_ones)

        q_zeros.append(jnp.mean(agents_q_values_0))
        q_ones.append(jnp.mean(agents_q_values_1))

    # plt.plot(cycles_ticks, orders)
    # plt.ylim(0,1)
    # plt.ylabel(r"$\phi$")
    # plt.xlabel("Episodes")
    # plt.savefig("Order")
    # plt.close()
    #
    # plt.plot(cycles_ticks, COMs)
    # plt.ylabel("|| COM ||")
    # plt.xlabel("Episodes")
    # plt.savefig("COMs")
    # plt.close()

    ax.plot(q_zeros, label = r"$s = ({\bf 0}, {\bf 0}, {\bf 0})$")
    ax.plot(q_ones, label = r"$s = ({\bf 1}, {\bf 1}, {\bf 1})$")

    plt.ylabel(r"$\langle \mathcal{Q} \rangle$")
    plt.xlabel("Episodes")
    plt.xlim(0, simulations)
    plt.ylim(0, 20)
    plt.legend()
    plt.savefig("V_Zero.png", bbox_inches = "tight")

    plt.close()

    # plt.plot(cycles_ticks, ave_loss, label = "Average")
    # plt.plot(cycles_ticks, med_loss, label = "Median")
    # plt.ylabel(r"Average Loss")
    # plt.xlabel("Episodes")
    # plt.legend()
    # plt.savefig("Average Loss")
    # plt.close()

    print ("Complete")

save_dir = "Download/N50_gamma0.9_mem3"




n = 50
tMax = 250
repeats = 4
key = jax.random.PRNGKey(0)

simulations = 1000
# plot_Q0(simulations, save_dir)



def plot_average_reward(cycle, save_dir, repeats):
    save_dir = Path(save_dir)

    chk = load_checkpoint(save_dir, cycle)
    fig, ax = plt.subplots(figsize=(12, 6))

    online_params = chk["online_params"]
    target_params = chk["target_params"]

    temps = 3
    data = np.zeros((3, tMax, repeats))
    Ts = np.linspace(0,0.01, temps)

    for i in range(repeats):
        key = jax.random.PRNGKey(i)

        for j in range(temps):
            sim = run_NN(key, n, tMax, target_params,Ts[j])

            print ("Sim Complete")

            # plot_Q0(key, 1000, save_dir)
            #
            # plot_trajectory(data[0], data[1], temp = T)

            data[j, :, i] = sim[5]
    data = np.mean(data, axis = 2)

    for t in range(temps):
        plt.plot(data[t,:], label = r"$T = $" + str(Ts[t]))
    plt.ylabel(r"$\langle r^t \rangle$")
    plt.xlabel(r"$t$ (time steps)")
    plt.ylim(0,1)
    plt.xlim(0, tMax)
    plt.legend()
    plt.savefig("AverageReward.png", bbox_inches = "tight")
    plt.show()

# plot_average_reward(999, save_dir, 10)



def video_frames(cycle, save_dir, temp):
    key = jax.random.PRNGKey(0)

    chk = load_checkpoint(save_dir, cycle)
    online_params = chk["online_params"]
    target_params = chk["target_params"]
    data = run_NN(key, n, tMax, target_params, temp)

    produce_frames(data[0], data[1], data[2], data[3], n, tMax)

video_frames(999, save_dir, 0.0)
