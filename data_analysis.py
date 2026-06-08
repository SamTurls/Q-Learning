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
from sklearn.cluster import DBSCAN

from learning_opacity_sym import init, compute_visual_states, to_binary, q_values, apply_action, load_checkpoint

from learning_opacity import q_values as asym_q_values

plt.rcParams['font.size'] = '16'

# discounting for Q-Learning
discount_factor = 1
# number of agents in a simulation
N = 2
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
n_s = 40
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
memory_length = 2
# fixed parameter for total number of actions
num_actions = 5
# the bounds of the sensors
sensor_bounds = jnp.array([2*jnp.pi/n_s * i for i in range(n_s+1)])  # length N+1
# the possible velocities
velocity = jnp.array([v0,v0,v0,v0+dv,v0-dv])
# the possible re-orientations
orientation = jnp.array([delta_theta, 0.0, -delta_theta, 0.0, 0.0])

def load_simulations(save_dir):
    results = np.load(save_dir + "/sims.npy")
    return results

def save_simulations(cycle, save_dir, sym, repeats = 500, temp = 0):
    save_path = Path(save_dir)
    chk = load_checkpoint(save_path, cycle)
    online_params = chk["online_params"]
    target_params = chk["target_params"]

    results = []

    for r in range(repeats):
        print (r/ repeats)
        #set the key up
        key = jax.random.PRNGKey(r)

        # run a simulation with T = 0
        sim = run_NN(key, N, tMax, target_params, temp, sym)

        results.append(sim)

    results_array = np.array(results)
    np.save(save_dir + "/sims.npy", results_array)

def dbscan(sim):
    xs, ys, os, vs = sim[0], sim[1], sim[2], sim[3]

    number_of_groups = np.zeros(tMax)
    size_of_largest_group = np.zeros(tMax)
    order_of_largest_group = np.zeros(tMax)

    for t in range(tMax):
        distance_matrix = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i <= j:
                    continue
                physical = np.sqrt( (xs[t, i]-xs[t, j]) ** 2 + (ys[t, i]-ys[t, j]) ** 2)/ (np.pi * np.sqrt(N))

                orientation = abs(os[t, i]-os[t, j])%(2*np.pi)
                orientation = min(orientation, 2 * np.pi - orientation)/(2 * delta_theta)

                distance_matrix[i,j] = distance_matrix[j,i] = (physical + orientation)/2

        # use dbscan to find clusters
        db = DBSCAN(eps=1, min_samples=1, metric='precomputed').fit(distance_matrix)

        unique, counts = np.unique(db.labels_, return_counts=True)

        number_of_groups[t] = len(unique)

        size_of_largest_group[t] = max(counts)

        index = np.argmax(counts)

        thetas = [[os[t, i] for i in range(N) if db.labels_[i] == index]]

        order_of_largest_group[t] = calculate_order(thetas)[0]

    return number_of_groups, size_of_largest_group, order_of_largest_group

def save_dbscan_data(cycle, save_dirs, syms, repeats = 1):

    for i, save_dir in enumerate(save_dirs):

        save_path = Path(save_dir)

        simulations = load_simulations(save_dir)

        results = []

        number_of_groups = np.zeros((tMax, repeats))
        size_of_largest_group = np.zeros((tMax, repeats))
        order_of_largest_group = np.zeros((tMax, repeats))

        for r in range(repeats):
            print (r/repeats * 100, "%")
            sim = simulations[r]

            db = dbscan(sim)

            number_of_groups[:, r] = db[0]
            size_of_largest_group[:, r] = db[1]
            order_of_largest_group[:, r] = db[2]

        results.append(number_of_groups)
        results.append(size_of_largest_group)
        results.append(order_of_largest_group)
        results_array = np.array(results)

        np.save(save_dir + "/dbscandata.npy", results_array)

def load_dbscan_data(save_dir):
    results = np.load(save_dir + "/dbscandata.npy")
    return results

def plot_dbscan_graphs(save_dirs, labels):

    average_number_of_groups = np.zeros((len(save_dirs), tMax))
    average_size_of_largest_group = np.zeros((len(save_dirs), tMax))
    average_order_of_largest_group = np.zeros((len(save_dirs), tMax))

    for i, save_dir in enumerate(save_dirs):

        data = load_dbscan_data(save_dir)

        number_of_groups = data[0]
        size_of_largest_group = data[1]
        order_of_largest_group = data[2]

        average_number_of_groups[i, :] = np.mean(number_of_groups, axis=1)
        average_size_of_largest_group[i, :] = np.mean(size_of_largest_group, axis=1)
        average_order_of_largest_group[i, :] = np.mean(order_of_largest_group, axis=1)

    """ Average Number of Groups """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        plt.plot(average_number_of_groups[i,:],  label = labels[i])

    plt.ylim(0,N)
    plt.xlim(0, tMax)
    plt.legend()
    plt.ylabel("Number of Clusters")
    plt.xlabel(r"$t$ (time steps)")
    plt.vlines(250, 0, N, 'k', linestyles = "dashed")

    plt.savefig("AverageNumberofClusters.eps", bbox_inches = "tight")
    plt.close()

    """ Average Size of Clusters """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        plt.plot(N/average_number_of_groups[i,:],  label = labels[i])
    plt.xlim(0, tMax)
    plt.ylim(1, 6)
    plt.legend()
    plt.ylabel("Average Size of Clusters")
    plt.xlabel(r"$t$ (time steps)")
    plt.vlines(250, 1, 6, 'k', linestyles = "dashed")
    plt.hlines(2, 0, tMax, 'k', linestyles = "dotted")

    plt.savefig("AverageSizeofClusters.eps", bbox_inches = "tight")
    plt.close()

    """ Average Size of Largest Group"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        plt.plot(average_size_of_largest_group[i,:],  label = labels[i])

    plt.ylim(0,N)
    plt.xlim(0, tMax)
    plt.legend()
    plt.ylabel("Size of Largest Cluster")
    plt.xlabel(r"$t$ (time steps)")
    plt.vlines(250, 0, N, 'k', linestyles = "dashed")

    plt.savefig("AverageSizeofLargestClusters.eps", bbox_inches = "tight")
    plt.close()

    """ Average Order of Largest Group"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        plt.plot(average_order_of_largest_group[i,:],   label = labels[i])

    plt.ylim(0.8,1)
    plt.xlim(0, tMax)
    plt.legend()
    plt.ylabel(r"$\phi_C$")
    plt.xlabel(r"$t$ (time steps)")
    plt.vlines(250, 0, 1, 'k', linestyles = "dashed")

    plt.savefig("AverageOrderofLargestCluster.eps", bbox_inches = "tight")
    plt.close()

def calculate_order(thetas):

    cos_mean = np.mean(np.cos(thetas), axis=1)  # shape (T,)
    sin_mean = np.mean(np.sin(thetas), axis=1)

    phis = np.sqrt(cos_mean**2 + sin_mean**2)   # shape (T,)

    return phis


def plot_average_order(save_dirs, labels, repeats = 1):

    fig, ax = plt.subplots(figsize=(12, 6))


    average_orders = np.zeros((len(save_dirs), tMax))

    for i, save_dir in enumerate(save_dirs):
        save_path = Path(save_dir)

        results = load_simulations(save_dir)

        # where to store the distances
        orders = np.zeros((tMax, repeats))

        for r in range(repeats):

            sim = results[r]
            thetas = np.array(sim[2])  # shape (N, T), in radians

            orders[:, r] = calculate_order(thetas)


        average_orders = np.mean(orders, axis=1)

        plt.plot(average_orders, label = labels[i])

    plt.ylabel(r"$\phi$")
    plt.xlabel(r"$t$ (time steps)")
    plt.legend()
    plt.xlim(0, tMax)
    plt.ylim(0, 1)
    plt.vlines(250, 0, 1, 'k', linestyles = "dashed")

    plt.savefig("Order.eps", bbox_inches = "tight")
    plt.close()

def plot_average_distance_to_COM(save_dirs, labels, repeats = 1):

    fig, ax = plt.subplots(figsize=(12, 6))


    distances = np.zeros((len(save_dirs), tMax))

    for i, save_dir in enumerate(save_dirs):
        save_path = Path(save_dir)

        results = load_simulations(save_dir)

        # where to store the distances
        average_distances = np.zeros((tMax, repeats))
        average_x = np.zeros((tMax, repeats))
        average_y = np.zeros((tMax, repeats))

        for r in range(repeats):

            sim = results[r]

            xs = np.array(sim[0])
            ys = np.array(sim[1])

            average_x[:, r] = np.mean(sim[0], axis = 1)

            average_y[:, r] = np.mean(sim[1], axis = 1)

            for t in range(tMax):
                xs[t, :] -= average_x[t, r]
                ys[t, :] -= average_y[t, r]

            average_distances[:, r] = np.mean(np.sqrt(xs**2 + ys**2), axis = 1)

        average_distances = np.mean(average_distances, axis = 1)

        distances[i, :] = average_distances

        m = (average_distances[tMax - 1] - average_distances[tMax - 101])/100
        c = average_distances[tMax - 1] - m * (tMax - 1)
        y = np.arange(0,tMax,1) * m + c
        plt.plot(average_distances, label = labels[i])

    plt.ylabel(r"$\langle | {\bf x} - {\bf x}_\mathrm{COM} |^t \rangle $")
    plt.xlabel(r"$t$ (time steps)")
    plt.legend()
    plt.vlines(250, 0, np.max(distances[:,:]), 'k', linestyles = "dashed")

    plt.xlim(0, tMax)
    plt.ylim(0, np.max(distances[:,:]))
    plt.savefig("Average Distance to COM.eps", bbox_inches = "tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        differences = np.zeros(tMax)
        for t in range(1, tMax):
            differences[t] = (distances[i,t] - distances[i,t-1])/v0
        plt.plot(differences,   label = labels[i])

    plt.ylim(-1,1)
    plt.xlim(0,tMax)
    plt.xlabel(r"$t$ (time steps)")
    plt.ylabel("Radial Alignment to COM")
    plt.legend()
    plt.vlines(250, -1, 1, 'k', linestyles = "dashed")


    plt.savefig("Radial Alignment to COM.eps", bbox_inches = "tight")

def plot_average_reward(save_dirs, labels, repeats = 1):

    average_rewards = np.zeros((len(save_dirs), tMax))
    rewards_top = np.zeros((len(save_dirs), tMax))
    rewards_bottom = np.zeros((len(save_dirs), tMax))

    average_opacity = np.zeros((len(save_dirs), tMax))
    opacity_std = np.zeros((len(save_dirs), tMax))

    for i, save_dir in enumerate(save_dirs):
        save_path = Path(save_dir)

        results = load_simulations(save_dir)

        # where to store the distances
        reward_data = np.zeros((tMax, repeats))
        opacity_data = np.zeros((tMax, repeats))

        for r in range(repeats):

            sim = results[r]

            rewards = np.array(sim[5])
            reward_data[:, r] = np.mean(rewards, axis = 1)

            opacity = np.array(sim[4])
            opacity_data[ :, r] = np.mean(opacity, axis = 1)


        average_rewards[i] = np.mean(reward_data, axis = 1)
        rewards_top[i] = np.percentile(reward_data, q = 95, axis = 1)
        rewards_bottom[i] = np.percentile(reward_data, q = 5, axis = 1)

        average_opacity[i] = np.mean(opacity_data, axis = 1)
        opacity_std[i] = np.std(opacity_data, axis = 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        plt.plot(average_rewards[i], label = labels[i])
        # plt.fill_between(np.arange(0,tMax), rewards_bottom[i], rewards_top[i],  alpha = 0.5)

    plt.ylabel(r"$\langle r^t \rangle$")
    plt.xlabel(r"$t$ (time steps)")
    plt.ylim(0,1)
    plt.xlim(0, tMax)
    plt.legend()
    plt.vlines(250, 0, 1, 'k', linestyles = "dashed")
    plt.savefig("AverageReward.png", bbox_inches = "tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, save_dir in enumerate(save_dirs):
        plt.plot(average_opacity[i], label = labels[i])

    plt.ylabel(r"$\langle \Theta^t \rangle$")
    plt.xlabel(r"$t$ (time steps)")
    plt.ylim(0,1)
    plt.xlim(0, tMax)
    plt.legend()
    plt.vlines(250, 0, 1, 'k', linestyles = "dashed")

    plt.savefig("AverageOpacity.eps", bbox_inches = "tight")



def produce_frames(xs, ys, os, vs, N, tMax, centered = False):
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
        if centered:
            plt.xlim(xs[t,0]-50,xs[t,0]+50)
            plt.ylim(ys[t,0]-50,ys[t,0]+50)

        plt.savefig("VideoPhotos/img" + "%03d" % t + ".jpg")

        plt.close()

def boltzman_selection(q_values, key, temp):
    q_values = jnp.asarray(q_values)
    # jax.debug.print("{}", q_values)

    def greedy_with_random_tie_break(key):
        max_q = jnp.max(q_values)
        is_max = (q_values == max_q)
        probs = is_max / jnp.sum(is_max)#
        # jax.debug.print("{}", probs)
        return jax.random.choice(key, a=q_values.shape[0], p=probs)

    def softmax_sample(key):
        logits = q_values / temp
        return jax.random.categorical(key, logits)

    return jax.lax.cond(
        temp == 0,
        greedy_with_random_tie_break,
        softmax_sample,
        key
    )

def run_NN(key, N, T, QNN_params, temp, sym):
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

        opacity = jnp.sum(curr_vis_states, axis = 1)/ (n_s)

        def update_frames(prev_frames, curr_vis_states):
            retained_frames = prev_frames[:,1:,:]
            return jnp.concatenate([retained_frames, curr_vis_states[:,None,:]], axis = 1)

        curr_frames = update_frames(prev_frames, curr_vis_states)

        # 2) Q-values for each agent
        if sym:
            agents_q_values = q_values(QNN_params, curr_frames)
        else:
            agents_q_values = asym_q_values(QNN_params, curr_frames)


        agent_keys = jax.random.split(subkey, N)   # N keys

        def select_action(qv, k):

            return boltzman_selection(qv, k, temp)

        # jax.debug.print("{}", agents_q_values[0])

        actions = jax.vmap(select_action)(agents_q_values, agent_keys)    # (N,)

        x1, y1, o1, v1 = apply_action(x, y, o, actions)

        next_vis_state_integers = jnp.array(compute_visual_states(x1, y1, o1))   # (N,)
        next_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(next_vis_state_integers)

        rewards = 1 -4*((jnp.sum(next_vis_states, axis=1)/n_s - 0.5) ** 2)

        transition = (x1, y1, o1, v1, opacity, rewards)

        new_carry = (key, x1, y1, o1, v1, curr_frames, params)

        return new_carry, transition

    carry_final, transitions = jax.lax.scan(step, carry0,jnp.arange(T),)

    return transitions   # see shapes below

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

def plot_Q0(simulations, save_dir, mod = 10):
    fig, ax = plt.subplots(figsize=(12, 6))

    q_zeros = np.zeros((simulations//mod, 5))
    q_ones = np.zeros((simulations//mod, 5))

    dummy_zeros = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
    dummy_ones = jnp.ones((1, memory_length, n_s), dtype=jnp.float32)


    for cycle in range(simulations):

        if cycle % mod == 0:
            print ("Starting Cycle " + str(cycle))

            save_dir = Path(save_dir)

            chk = load_checkpoint(save_dir, cycle)

            online_params = chk["online_params"]
            target_params = chk["target_params"]
            agents_q_values_0 = asym_q_values(online_params, dummy_zeros)
            agents_q_values_1 = asym_q_values(online_params, dummy_ones)
            q_zeros[cycle//mod, :] = agents_q_values_0[0]
            q_ones[cycle//mod, :] = agents_q_values_1[0]

    for i in [0,2]:
        ax.plot(np.arange(0, simulations, mod), q_zeros[:,i], label = str(i))

    # ax.plot(q_ones, label = r"$s = ({\bf 1}, {\bf 1}, {\bf 1})$")

    plt.ylabel(r"$\langle \mathcal{Q} \rangle$")
    plt.xlabel("Episodes")

    plt.savefig("V_Zeros.png", bbox_inches = "tight")
    plt.close()
    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(5):
        ax.plot(np.arange(0, simulations, mod), q_ones[:,i], label = str(i))

    # ax.plot(q_ones, label = r"$s = ({\bf 1}, {\bf 1}, {\bf 1})$")

    plt.ylabel(r"$\langle \mathcal{Q} \rangle$")
    plt.xlabel("Episodes")
    plt.legend()

    plt.savefig("V_Ones.png", bbox_inches = "tight")

    plt.close()

    print ("Complete")

# plot_Q0(simulations, save_dir)


def asymmetry(params):
    zeros_state = jnp.ones((1, memory_length, n_s), dtype=jnp.float32)
    l = asym_q_values(params, zeros_state)[0]
    r = asym_q_values(params, zeros_state[..., ::-1])[0]
    s = abs(l[0]-r[2]) + abs(l[1]-r[1]) + abs(l[2]-r[0]) + abs(l[3]-r[3]) + abs(l[4]-r[4])
    return s/5

def best_action(params):
    zeros_state = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
    l = asym_q_values(params, zeros_state)[0]
    print (l)
    if np.argmax(l) == 0:
        return -1
    elif np.argmax(l) == 2:
        return 1
    else:
        return 0

def plot_best_action(simulations):
    fig, ax = plt.subplots(figsize=(12, 6))
    a_s = []
    for cycle in range(0, simulations):
        a = 0
        chk = load_checkpoint(save_dir, cycle)
        online_params = chk["online_params"]
        a_s.append(best_action(online_params))

    ax.plot(a_s, 'k.')
    ax.set_yticks([-1,0,1], ["Left", "Neutral", "Right"])
    plt.show()

def plot_asymmetry_average(simulations, smooth = 10):
    fig, ax = plt.subplots(figsize=(12, 6))
    a_s = []
    for cycle in range(simulations):
        print (cycle)
        a = 0
        chk = load_checkpoint(save_dir, cycle)
        online_params = chk["online_params"]
        a_s.append(asymmetry(online_params))

    sim_average = [np.mean(a_s[i-smooth:i+smooth + 1]) for i in range(smooth, simulations-smooth -1)]
    plt.semilogy(a_s, alpha = 0.5)
    plt.semilogy(jnp.arange(smooth, simulations-smooth -1), sim_average)
    plt.xlabel("Episodes")
    plt.ylabel(r"$\langle A^*(s^t) \rangle$")
    plt.savefig("Asymmetry.png", bbox_inches = "tight")

def video_frames(cycle, save_dir, temp, sym = False, centered = False, sim = 1):

    simulations = load_simulations(save_dir)
    data = simulations[sim]

    produce_frames(data[0], data[1], data[2], data[3], N, tMax, centered )

N = 50
tMax = 1000
repeats = 500
key = jax.random.PRNGKey(0)

simulations = 2000

# save_dir = "N50M1"
# memory_length = 2
# video_frames(9995, save_dir, 1, True, True)
#
# save_dir = "N50M2_sym"
# memory_length = 1
# video_frames(9995, save_dir, 1, sym = True, centered = True)


# save_simulations(9995, "N50M2_sym", True, repeats = repeats)
# save_simulations(9995, "N50M2_asym+sym", True, repeats = repeats)
memory_length = 1
# save_simulations(9995, "N50M1_asym+sym", True, repeats = repeats)
# # save_simulations(9995, "N50M1_sym", True, repeats = repeats)
# save_dbscan_data(9995, ["N50M1_asym+sym"], syms = [True], repeats = repeats)
# save_dbscan_data(9995, ["N50M1_sym"], syms = [True], repeats = repeats)

# memory_length = 2

# save_simulations(9995, "N50M2_sym_utilitarian", True, repeats = repeats)
# save_simulations(9995, "N50M2_asym+sym", True, repeats = repeats)
#
#
# save_dbscan_data(9995, ["N50M2_sym_utilitarian"], syms = [True], repeats = repeats)
# save_dbscan_data(9995, ["N50M2_asym+sym"], syms = [True], repeats = repeats)

# save_simulations(9995, "N50M2", True, repeats = repeats)
#
# save_dbscan_data(9995, ["N50M2_asym+sym"], syms = [True], repeats = repeats)


#
# plot_average_distance_to_COM(["N50M1", "N50M2"], labels = ["M1", "M2"], repeats = repeats)
plot_average_reward(["N50M2", "N50M2_asym+sym", "N50M1", "N50M1_asym+sym"  ], labels = ["M = 2", "M = 2 + Sym", "M = 1", "M = 1 + Sym"], repeats = 500)
# plot_average_order( 9995, ["N50M1", "N50M1_sym", "N50M2_asym+sym"], labels = ["Asym NN", "Sym NN", "Asym NN w/ Sym"], repeats = repeats)
# plot_dbscan_graphs(["N50M1", "N50M1_sym", "N50M1_asym+sym"],["Asym NN", "Sym NN", "Asym NN w/ Sym"])

# dummy_zeros = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
# dummy_ones = jnp.ones((1, memory_length, n_s), dtype=jnp.float32)
#
#
#
# """ FSM """
#
# save_dir = "N10M2_FSM"
# N = 50
# tMax = 250
# save_path = Path(save_dir)
# chk = load_checkpoint(save_path, 999)
# online_params = chk["online_params"]
# target_params = chk["target_params"]
# dummy_zeros = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
# dummy_ones = jnp.ones((1, memory_length, n_s), dtype=jnp.float32)
# print (q_values(target_params, dummy_ones))
# print (q_values(target_params, dummy_zeros))
#
# key = jax.random.PRNGKey(0)
# data = run_NN(key, N, tMax, target_params, 0, True)
# produce_frames(data[0], data[1], data[2], data[3], N, tMax, False)


def plot_losses_graph(folders):

    cmap = plt.get_cmap("tab10")
    colors = {folder: cmap(i) for i, folder in enumerate(folders)}

    # Individual plots
    for folder in folders:

        color = colors[folder]

        loss_dir = Path(folder) / "Losses"
        files = sorted(loss_dir.glob("losses_cycle_*.npy"))

        losses = [np.mean(np.load(f)) for f in files]

        window = 50
        moving_avg = np.convolve(
            losses,
            np.ones(window) / window,
            mode="valid"
        )

        x = np.arange(len(losses)) * 5

        plt.figure(figsize=(8, 5))

        plt.plot(
            x,
            losses,
            alpha=0.3,
            color=color,
            label="Cycle Loss"
        )

        plt.plot(
            x[window-1:],
            moving_avg,
            linewidth=2,
            color=color,
            label="Moving Average"
        )
        plt.ylabel("Average Loss")
        plt.yscale("log")
        plt.xlim(0, 10000)
        plt.legend()
        plt.xlabel("Simulations")

        plt.savefig(Path(folder) / "Losses Graph.png", bbox_inches = "tight")
        plt.close()

    # Combined plot
    plt.figure(figsize=(8, 5))

    for folder in folders:

        color = colors[folder]

        loss_dir = Path(folder) / "Losses"
        files = sorted(loss_dir.glob("losses_cycle_*.npy"))

        losses = [np.mean(np.load(f)) for f in files]

        window = 50
        moving_avg = np.convolve(
            losses,
            np.ones(window) / window,
            mode="valid"
        )

        x = np.arange(len(losses)) * 5

        plt.plot(
            x,
            losses,
            alpha=0.15,
            color=color
        )

        plt.plot(
            x[window-1:],
            moving_avg,
            linewidth=2,
            color=color,
            label=folder[-2:]
        )

    plt.yscale("log")
    plt.xlim(0, 10000)
    plt.xlabel("Simulations")
    plt.ylabel("Average Loss")

    plt.legend()
    plt.savefig(f"Losses Graph {folders}.png", bbox_inches = "tight")
    plt.close()

# plot_losses_graph(["N50M1", "N50M2"])
