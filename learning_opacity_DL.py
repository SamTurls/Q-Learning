import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.serialization as ser
import argparse
import optax
import copy
from pathlib import Path
from jax import checkpoint as remat
from jax import tree_util

import time


key = jax.random.PRNGKey(0)

# discounting for Q-Learning
discount_factor = 0.5
# number of agents in a simulation
N = 2
# length of time of simulation
T = 256
# number of training simulations
simulations = 2000
#batch size for loss calculation
batch_size = 128
# number of passes through the dataset
num_epochs = 32
#minimum Temperature
T_min = 0.00
# starting Temperature
T_0 = 10
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
#fixed parameter for sensor activation
opacity_threshold = 0.5
#fixed parameter for time step
dt = 1
# fixed parameter for number of frames
memory_length = 5
# fixed parameter for total number of actions
num_actions = 5
# the bounds of the sensors
sensor_bounds = jnp.array([2*jnp.pi/n_s * i for i in range(n_s+1)])  # length N+1
# the possible velocities
velocity = jnp.array([v0,v0,v0,v0+dv,v0-dv])
# the possible re-orientations
orientation = jnp.array([delta_theta, 0.0, -delta_theta, 0.0, 0.0])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--simulations", type=int, required = True)

    p.add_argument("--memory_length", type=int, required = True)

    # research parameters
    p.add_argument("--discount_factor", type=float, default=0.9)
    p.add_argument("--N", type=int, default=10)

    return p.parse_args()

# RCNN for taking a visual state
class RingConvEncoder(nn.Module):
    features: int = 32
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x):
        """
        x: (B, n_s, 1)
        returns: (B, n_s, features)
        """

        pad = self.kernel_size // 2

        # --- Conv layer 1 (circular) ---
        x = jnp.concatenate(
            [x[:, -pad:, :], x, x[:, :pad, :]],
            axis=1
        )
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding="VALID"
        )(x)

        x = nn.relu(x)

        # --- Conv layer 2 (circular) ---
        x = jnp.concatenate(
            [x[:, -pad:, :], x, x[:, :pad, :]],
            axis=1
        )
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding="VALID"
        )(x)
        x = nn.relu(x)

        return x

def encode_frames(encoder, frames):
    """
    frames: (B, T, n_s)
    returns: (B(atch), T(ime), n_s, F(rames))
    """
    frames = frames[..., None]  # add channel dim

    # map encoder over time axis
    return jax.vmap(encoder, in_axes=1, out_axes=1)(frames)

class QNetwork(nn.Module):
    num_actions: int
    features: int = 32
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, frames):
        """
        frames: (B, T, n_s)
        returns: (B, num_actions)
        """
        encoder = RingConvEncoder(features=self.features)

        h = encode_frames(encoder, frames)   # (B, T, n_s, F)

        # concatenate encoded frames
        h = h.reshape(h.shape[0], -1)        # (B, T * n_s * F)

        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)

        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)

        return nn.Dense(self.num_actions)(h)

def call_networks(QNNs, frames):

    return jax.vmap(q_values_single, in_axes=(0,0))(QNNs, frames)

def q_values_single(params, frame):
    return q_values_batch(params, frame[None, ...])[0]

def q_values_batch(params, frame):
    """
    params: network parameters
    frames: array of shape (batch, m, n_s)
    returns: Q-values of shape (batch, num_actions)
    """

    def swap_actions(x):
        # swap columns 0 and 2
        x = x.at[:, [0, 2]].set(x[:, [2, 0]])
        return x

    mirror = frame[..., ::-1]
    a = q_model.apply(params, frame)
    b = q_model.apply(params, mirror)
    b = swap_actions(b)

    return 0.5 * (a + b)

def compute_td_target(target_params, rewards, next_frames, gamma=discount_factor):
    """
    target_params: parameters of the target network
    rewards: shape (batch,)
    next_states: shape (batch, n_s)
    dones: shape (batch,) with 1 if episode ended, else 0
    gamma: discount factor
    """
    # Q-values from the target network for the next states
    q_next = q_values_batch(target_params, next_frames)   # shape: (batch, 5)

    # max_a Q(s', a')
    max_next_q = jnp.max(q_next, axis=1)           # shape: (batch,)

    # TD target:
    #   r + gamma * max_next_q * (1 - done)
    target = rewards + gamma * max_next_q

    return target

@jax.jit

def train_step(online_params, target_params, opt_state,
               curr_frames, actions, rewards, next_frames, gamma):
    """
    Performs one gradient update on the online Q-network.
    """

    # Define loss function inside so JAX can compute gradients
    def loss_fn(params):
        # 1. Q-values from online network for current states

        q_all = q_values_batch(params, curr_frames)    # shape: (batch, 5)

        # 2. Select Q-value for the action taken
        # actions is shape (batch,), so we gather the correct column
        q_pred = jnp.take_along_axis(
            q_all, actions[:, None], axis=1
        ).squeeze()                          # shape: (batch,)

        # 3. Compute TD target using the target network
        td_target = compute_td_target(
            target_params, rewards, next_frames, gamma
        )

        # 4. MSE loss
        loss = jnp.mean((q_pred - td_target) ** 2)

        return loss

    # 5. Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(online_params)

    # 6. Apply optimizer update
    updates, opt_state = optimizer.update(grads, opt_state, online_params)
    new_online_params = optax.apply_updates(online_params, updates)

    # 7. Return everything updated
    return new_online_params, opt_state, loss

def make_minibatches(states, actions, rewards, next_states, batch_size, rng):

    B = states.shape[0]

    # Shuffle all samples in the same way
    perm = jax.random.permutation(rng, B)

    states = states[perm]
    actions = actions[perm]
    rewards = rewards[perm]
    next_states = next_states[perm]

    # Keep only a whole number of full minibatches
    num_batches = B // batch_size
    usable = num_batches * batch_size

    states = states[:usable]
    actions = actions[:usable]
    rewards = rewards[:usable]
    next_states = next_states[:usable]

    # Reshape into minibatches
    batch_states = states.reshape(num_batches, batch_size, *states.shape[1:])
    batch_actions = actions.reshape(num_batches, batch_size)
    batch_rewards = rewards.reshape(num_batches, batch_size)
    batch_next_states = next_states.reshape(num_batches, batch_size, *next_states.shape[1:])

    return batch_states, batch_actions, batch_rewards, batch_next_states


def train_one_epoch(
    online_params,
    target_params,
    opt_state,
    states,
    actions,
    rewards,
    next_states,
    batch_size,
    epoch_rng,
    gamma,
):
    # Build all minibatches for this epoch
    batch_states, batch_actions, batch_rewards, batch_next_states = make_minibatches(
        states, actions, rewards, next_states, batch_size, epoch_rng
    )

    def one_batch(carry, batch):

        online_params, opt_state = carry
        b_states, b_actions, b_rewards, b_next_states = batch

        online_params, opt_state, loss = train_step(
            online_params,
            target_params,
            opt_state,
            b_states,
            b_actions,
            b_rewards,
            b_next_states,
            gamma,
        )

        return (online_params, opt_state), loss

    # Scan over all minibatches
    (online_params, opt_state), losses = jax.lax.scan(
        one_batch,
        (online_params, opt_state),
        (batch_states, batch_actions, batch_rewards, batch_next_states),
    )

    last_loss = losses[-1]

    return online_params, opt_state, losses, last_loss


def train_for_epochs(
    rng,
    online_params,
    target_params,
    opt_state,
    states,
    actions,
    rewards,
    next_states,
    gamma,
    batch_size=80,
    num_epochs=10,
):

    def one_epoch(carry, _):
        rng, online_params, opt_state = carry

        # Fresh key for this epoch
        rng, epoch_rng = jax.random.split(rng)

        online_params, opt_state, losses, last_loss = train_one_epoch(
            online_params,
            target_params,
            opt_state,
            states,
            actions,
            rewards,
            next_states,
            batch_size,
            epoch_rng,
            gamma,
        )
        # Debug print is safe here
        # jax.debug.print("epoch loss = {}", last_loss)

        new_carry = (rng, online_params, opt_state)

        # Return both the final epoch loss and the whole minibatch-loss trace
        return new_carry, (last_loss, losses)

    (rng, online_params, opt_state), (epoch_losses, batch_losses) = jax.lax.scan(
        one_epoch,
        (rng, online_params, opt_state),
        xs=None,
        length=num_epochs,
    )


    return online_params, opt_state, epoch_losses

def train_all_agents(
    training_key,
    online_networks,
    target_networks,
    opt_states,
    states,
    actions,
    rewards,
    next_states,
    gamma,
    batch_size,
    num_epochs,
):
    # Number of agents
    N = states.shape[0]
    # One RNG key per agent
    subkeys = jax.random.split(training_key, N)

    def train_one_agent(
        key,
        online_params,
        target_params,
        opt_state,
        agent_states,
        agent_actions,
        agent_rewards,
        agent_next_states,
    ):
        """
        Train one agent's network on that agent's dataset.
        """
        return train_for_epochs(
            key,
            online_params,
            target_params,
            opt_state,
            agent_states,
            agent_actions,
            agent_rewards,
            agent_next_states,
            gamma,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

    # Map over the leading agent axis
    new_online_networks, new_opt_states, epoch_losses = jax.vmap(train_one_agent,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )(
        subkeys,
        online_networks,
        target_networks,
        opt_states,
        states,
        actions,
        rewards,
        next_states,
    )

    return new_online_networks, new_opt_states, epoch_losses

def update_target_network(online_params):
    """
    Hard update: make target network equal to the new online network.
    """
    return copy.deepcopy(online_params)

def init(key, N = 1):
    # Split RNG for reproducibility
    key1, key2, key3, key4 = jax.random.split(key, num = 4)

    size = jax.random.uniform(key4)

    # Positions: uniform in [0,1) scaled by N
    x_positions = jax.random.uniform(key1, (N,)) * N * (1 + 4 * size)

    y_positions = jax.random.uniform(key2, (N,)) * N * (1 + 4 * size)

    # Orientations: normal with mean 0, std=1, scaled by delta_theta
    orientations = jax.random.normal(key3, (N,)) * 3 * delta_theta

    velocities = jnp.full((N,), v0)

    return x_positions, y_positions, orientations, velocities

def apply_action(x, y, o, action):

    o_next = o + orientation[action]

    x_next = x + velocity[action] * jnp.cos(o_next) * dt
    y_next = y + velocity[action] * jnp.sin(o_next) * dt
    v_next = velocity[action]

    return x_next, y_next, o_next, v_next

def rotate_vectors(vectors, theta):
    # rotation matrix for clockwise rotation by theta

#     jax.debug.print("{}", vectors)

    rot = jnp.array([
        [jnp.cos(theta), jnp.sin(theta)],
        [-jnp.sin(theta),jnp.cos(theta)]
    ])
    return vectors @ rot.T

def binary_array_to_number(binary_array):

    powers_of_two = 2 ** jnp.arange(n_s-1, -1, -1)
    return jnp.sum(binary_array * powers_of_two)

def _split_generator():

    def fun(arc):
        return jax.lax.cond(
            arc[1] > 2 * jnp.pi,
            lambda: [
                jnp.array([arc[0], 2*jnp.pi]),
                jnp.array([jnp.zeros_like(2*jnp.pi), arc[1]%(2*jnp.pi)]),
            ],
            lambda: [jnp.array(arc), jnp.array([2*jnp.pi,2*jnp.pi])],
        )

    return fun

_split = jax.vmap(_split_generator(), in_axes=0)

def unionise_projection(arcs):

    def scan_fn(carry, idx):
        i, mergearcs = carry
        a = arcs[idx]

        # check overlap: current arc start <= previous merged arc end
        overlap = a[0] <= mergearcs[i, 1]

        def merge_fn(_):
            # merge current arc into previous
            new_mergearcs = mergearcs.at[i, 1].set(jnp.maximum(mergearcs[i, 1], a[1]))
            return i, new_mergearcs

        def next_fn(_):
            # move to next merged arc
            new_mergearcs = mergearcs.at[i + 1].set(a)
            return i + 1, new_mergearcs

        new_carry = jax.lax.cond(overlap, merge_fn, next_fn, operand=None)
        return new_carry, None

    # initialize merged arcs array
    arcs = arcs[jnp.argsort(arcs[:, 0])]  # sort on end angle
    mergearcs = jnp.zeros_like(arcs)
    mergearcs = mergearcs.at[0].set(arcs[0])
    init_carry = (0, mergearcs)

    # scan over remaining arcs
    (i, mergearcs), _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(1, arcs.shape[0]))
    return i, mergearcs

def sensor_fill(sensor_range, intervals):
    """
    Compute how much of a single sensor range is covered by intervals.
    """
    s0, s1 = sensor_range
    l, r = intervals[:,0], intervals[:,1]
    overlaps = jnp.maximum(0.0, jnp.minimum(r, s1) - jnp.maximum(l, s0))
    return jnp.sum(overlaps)

def fill_sensors(sensor_bounds, merged_intervals):
    sensor_ranges = jnp.stack([sensor_bounds[:-1], sensor_bounds[1:]], axis=1)
    fill_per_sensor = jax.vmap(sensor_fill, in_axes=(0,None))(sensor_ranges, merged_intervals)
    sensor_sizes = sensor_bounds[1:] - sensor_bounds[:-1]
    fill_fraction = fill_per_sensor / sensor_sizes
    return jnp.where(fill_fraction > opacity_threshold , 1, 0)

def boltzman_selection(q_values, key, temp):
    q_values = jnp.asarray(q_values)
    # jax.debug.print("{}", q_values)

    def greedy_with_random_tie_break(key):
        max_q = jnp.max(q_values)
        is_max = (q_values == max_q)
        probs = is_max / jnp.sum(is_max)#
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

def run_sim(key, N, T, QNNs, temp):
    """
    Runs one simulation of N agents for T Time
    QNN: The network used for the heuristic
    """
    # set up the agents
    x0, y0, o0, v0 = init(key, N)

    init_frames = jnp.zeros((N, memory_length, n_s), dtype=jnp.int32)
    carry0 = (key, x0, y0, o0, v0, init_frames, QNNs)

    def step(carry, t):
        key, x, y, o, v, prev_frames, params = carry

        key, subkey = jax.random.split(key)

        # get current visual states
        vis_state_integers = jnp.array(compute_visual_states(x, y, o))   # (N,)
        curr_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(vis_state_integers)

        def update_frames(prev_frames, curr_vis_states):
            retained_frames = prev_frames[:,1:,:]
            return jnp.concatenate([retained_frames, curr_vis_states[:,None,:]], axis = 1)

        curr_frames = update_frames(prev_frames, curr_vis_states)

        # 2) Q-values for each agent
        agents_q_values = call_networks(QNNs, curr_frames)

        agent_keys = jax.random.split(subkey, N)   # N keys

        def select_action(qv, k):

            return boltzman_selection(qv, k, temp)

        actions = jax.vmap(select_action)(agents_q_values, agent_keys)    # (N,)

        # actions = jnp.zeros(N)

        x1, y1, o1, v1 = apply_action(x, y, o, actions)

        next_vis_state_integers = jnp.array(compute_visual_states(x1, y1, o1))   # (N,)
        next_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(next_vis_state_integers)

        rewards = 1 - 4 * ((jnp.sum(next_vis_states, axis=1)/n_s - 0.5) ** 2)

        transition = (curr_frames, actions, rewards)

        new_carry = (key, x1, y1, o1, v1, curr_frames, QNNs)

        return new_carry, transition

    carry_final, transitions = jax.lax.scan(step, carry0,jnp.arange(T),)

    return transitions   # see shapes below

def compute_visual_states(x, y, o):
    current_positions = jnp.stack([x, y], axis=-1)    # (N, 2)

    def agent_fn(i):
#         #roll agents so this agent is first
        x_roll, y_roll, o_roll = jnp.roll(x, -i), jnp.roll(y, -i), jnp.roll(o, -i)
        current_roll = jnp.roll(current_positions, -i, axis =0)

        vs = get_visual_state(x_roll[0], y_roll[0], o_roll[0], current_roll[1:])


        return vs

    return jax.vmap(agent_fn)(jnp.arange(x.shape[0]))


def get_visual_state(x, y, o, assumed_positions):

    rotated = rotate_vectors(assumed_positions - jnp.array([x,y]), o)

    dists = jnp.sqrt(jnp.sum(rotated**2, axis=1))

    alphas = jnp.arctan2(rotated[:,1], rotated[:,0])

    delta_alpha = jnp.arcsin(1/dists)

    delta_alpha = jnp.nan_to_num(delta_alpha, nan= jnp.pi)

    intervals = merged_intervals(alphas, delta_alpha)

    sensor = fill_sensors(sensor_bounds, intervals)

    return binary_array_to_number(sensor)

def merged_intervals(alphas, delta_alpha):

    left  = (alphas - delta_alpha) % (2 * jnp.pi)
    right = left + 2 * delta_alpha

    arcs = jnp.stack([left, right], axis=1)

    arcs = jnp.concatenate(_split(arcs))

    i, merged_arcs = unionise_projection(arcs)

    return merged_arcs

def to_binary(n, width):
    # Make exponents: [width-1, ..., 1, 0]
    exponents = jnp.arange(width - 1, -1, -1)
    # Compute: (n >> k) & 1 for each bit
    return (n >> exponents) & 1

def mirror_state(state):
    mirror_state = state[::-1]
    return mirror_state

def mirror_actions(actions):
    # actions: (...,) int
    # 0 <-> 2, everything else unchanged
    return jnp.where(actions == 0, 2,
           jnp.where(actions == 2, 0, actions))

def get_input_state(prev_vis_state, curr_vis_state):
    return jnp.concatenate([prev_vis_state, curr_vis_state])


def save_checkpoint(save_dir, online_params, target_params, opt_state, cycle):
    data = {
        "cycle": cycle,
        "online_params": online_params,
        "target_params": target_params,
        "opt_state": opt_state,
    }
    bytes_data = ser.to_bytes(data)

    ckpt_path = Path(save_dir) / f"cycle_{cycle:06d}.ckpt"
    with open(ckpt_path, "wb") as f:
        f.write(bytes_data)

    print(f"Saved checkpoint at {ckpt_path}")

def load_checkpoint(save_dir, cycle):
    ckpt_path = Path(save_dir) / f"cycle_{cycle:06d}.ckpt"
    with open(ckpt_path, "rb") as f:
        bytes_data = f.read()
    return ser.from_bytes(None, bytes_data)

def prepare_data(data, memory_length):
    """
    Prepare training data while preserving the agent axis.

    Inputs
    ------
    data[0] : frames   with shape (T, N, M, S)
        T = number of timesteps
        N = number of agents
        M = memory length stored in each frame
        S = number of sensors / state features

    data[1] : actions  with shape (T, N)
    data[2] : rewards  with shape (T, N)

    Returns
    -------
    states_out :      (N, 2L, M, S)
    actions_out :     (N, 2L)
    rewards_out :     (N, 2L)
    next_states_out : (N, 2L, M, S)

    where
    -----
    L = T - 1 - memory_length

    Interpretation
    --------------
    states_out[i] contains all training state samples for agent i.
    actions_out[i] contains the matching actions for agent i.
    rewards_out[i] contains the matching rewards for agent i.
    next_states_out[i] contains the matching next states for agent i.

    The factor of 2 comes from mirror augmentation.
    """

    # Unpack the simulation outputs
    frames, actions, rewards = data[0], data[1], data[2]

    # Total number of timesteps in the rollout
    T = frames.shape[0]

    # Valid training indices:
    # j runs from memory_length up to T-2 inclusive
    # because we use both frames[j] and frames[j+1]
    j = jnp.arange(memory_length, T - 1)

    # Gather the state, action, reward, and next-state data over valid times
    # states, next_states: (L, N, M, S)
    # act, rew:            (L, N)
    states = frames[j]
    next_states = frames[j + 1]
    act = actions[j]
    rew = rewards[j]

    # Move the agent axis to the front so that outputs are grouped by agent
    # (L, N, M, S) -> (N, L, M, S)
    # (L, N)       -> (N, L)
    states = jnp.swapaxes(states, 0, 1)
    next_states = jnp.swapaxes(next_states, 0, 1)
    act = jnp.swapaxes(act, 0, 1)
    rew = jnp.swapaxes(rew, 0, 1)

    # Mirror augmentation:
    # reverse the sensor axis in each state
    # states_m, next_states_m still have shape (N, L, M, S)
    states_m = states[..., ::-1]
    next_states_m = next_states[..., ::-1]

    # Mirror the actions so they remain consistent with the mirrored states
    # act_m has shape (N, L)
    act_m = mirror_actions(act)

    # Concatenate original and mirrored samples along the sample axis
    # This preserves the agent axis N
    #
    # Final shapes:
    # states_out, next_states_out: (N, 2L, M, S)
    # actions_out, rewards_out:    (N, 2L)
    states_out = jnp.concatenate([states, states_m], axis=1)
    next_states_out = jnp.concatenate([next_states, next_states_m], axis=1)
    actions_out = jnp.concatenate([act, act_m], axis=1)
    rewards_out = jnp.concatenate([rew, rew], axis=1)

    return states_out, actions_out, rewards_out, next_states_out


def train(key, online_networks, target_networks, opt_states, start, end, simulations, save_dir, T_0, T_min, decay, gamma):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for cycle in range(start, end):

        temp = T_min + (T_0-T_min) * jnp.exp(-decay * cycle/simulations)

        key, sim_key, training_key = jax.random.split(key, 3)

        print(f"\n=== Training Cycle {cycle+1} ===")
        print(f"\n=== Temperature = {temp} ===")

        data = run_sim(sim_key, N, T, online_networks, temp)

        # TODO Update prepare data so that the outputs are split by agent
        states, actions, rewards, next_states = prepare_data(data, memory_length)


        training_key, train_key = jax.random.split(training_key)

        online_networks, opt_states, losses = train_all_agents(
            train_key,
            online_networks,
            target_networks,
            opt_states,
            states,
            actions,
            rewards,
            next_states,
            gamma,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        # save losses

        target_networks = update_target_network(online_networks)
        if cycle % 5 == 0:
            t1 = time.time()
            tpc = (t1-t0)/(cycle+1)

            print ("Time remaining:", round(tpc * (simulations - cycle)/60), " minutes")
            print ("Average Loss", jnp.mean(losses))
            jnp.save(save_dir / f"losses_cycle_{cycle:06d}.npy", jnp.array(losses))
            # update the target network
            # save the network
            save_checkpoint(save_dir, online_networks, target_networks, opt_states, cycle)

    return key, online_params, target_params, opt_state

q_model = QNetwork(num_actions=num_actions)
# pick the optimizer
optimizer = optax.adamw(
    learning_rate= 3e-4,
    weight_decay=1e-5
)

def main():
    global key, online_params, target_params, opt_state, discount_factor, N, memory_length, T_0, T_min, decay, simulations

    args = parse_args()

    simulations = args.simulations

    discount_factor = args.discount_factor
    N = args.N
    memory_length = args.memory_length

    T_0 = 10
    T_min = 0
    decay = 10

    online_networks = []
    target_networks = []
    opt_states = []

    # initialize N networks

    dummy_frames = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
    for i in range(N):
        key, subkey = jax.random.split(key)
        online_params = q_model.init(subkey, dummy_frames)
        target_params = update_target_network(online_params)
        opt_state = optimizer.init(online_params)
        online_networks.append(online_params)
        target_networks.append(target_params)
        opt_states.append(opt_state)

    online_networks = tree_util.tree_map(lambda *xs: jnp.stack(xs), *online_networks)
    target_networks = tree_util.tree_map(lambda *xs: jnp.stack(xs), *target_networks)
    opt_states = tree_util.tree_map(lambda *xs: jnp.stack(xs), *opt_states)
    # dummy_frames = jnp.ones((N, memory_length, n_s), dtype=jnp.float32)
    #
    # agents_q_values = call_networks(online_networks, dummy_frames)
    #
    # jax.debug.print("{}", agents_q_values)

    key, online_networks, target_networks, opt_state = train(
        key, online_networks, target_networks, opt_states, args.start, args.end, simulations, args.save_dir, T_0, T_min, decay, discount_factor)

if __name__ == "__main__":
    main()
