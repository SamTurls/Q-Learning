import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.serialization as ser
import argparse
import optax
import copy
from pathlib import Path
from jax import checkpoint as remat


key = jax.random.PRNGKey(0)

# discounting for Q-Learning
discount_factor = 0.5
# number of agents in a simulation
N = 2
# length of time of simulation
T = 250
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
    p.add_argument("--N", type=int, default=50)

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

def q_values(params, frames):
    """
    params: network parameters
    frames: array of shape (batch, m, n_s)
    returns: Q-values of shape (batch, num_actions)
    """

    a = q_model.apply(params, frames)

    return a

def compute_td_target(target_params, rewards, next_frames, gamma=discount_factor):
    """
    target_params: parameters of the target network
    rewards: shape (batch,)
    next_states: shape (batch, n_s)
    dones: shape (batch,) with 1 if episode ended, else 0
    gamma: discount factor
    """
    # Q-values from the target network for the next states
    q_next = q_values(target_params, next_frames)   # shape: (batch, 5)

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

        q_all = q_values(params, curr_frames)    # shape: (batch, 5)

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

def iterate_minibatches(states, actions, rewards, next_states, batch_size, rng):
    """
    Simple generator that yields mini-batches from the full dataset.
    All inputs are jnp.arrays with the same length N in axis 0.
    """
    N = states.shape[0]

    # Create a random permutation of indices to shuffle the data
    perm = jax.random.permutation(rng, N)
    perm = jnp.array(perm)

    # Apply the permutation to shuffle all arrays in the same way
    states_shuffled      = states[perm]
    actions_shuffled     = actions[perm]
    rewards_shuffled     = rewards[perm]
    next_states_shuffled = next_states[perm]


    # Step through the shuffled data in chunks of batch_size
    for start in range(0, N, batch_size):
        end = start + batch_size
        if end > N:
            break  # drop last incomplete batch for simplicity

        yield (
            states_shuffled[start:end],
            actions_shuffled[start:end],
            rewards_shuffled[start:end],
            next_states_shuffled[start:end],

        )


def train_one_epoch(online_params, target_params, opt_state,
                    states, actions, rewards, next_states,
                    batch_size, epoch_rng, gamma ):
    """
    Trains online_params for one full epoch over the entire offline dataset.
    Returns updated (online_params, opt_state) and the final loss observed.
    """

    last_loss = 0.0
    # Generate minibatches using the epoch-specific RNG
    for (batch_states, batch_actions, batch_rewards, batch_next_states) in         iterate_minibatches(states, actions, rewards, next_states, batch_size, epoch_rng):

        # Perform one update on this minibatch
        online_params, opt_state, loss = train_step(
            online_params,
            target_params,
            opt_state,
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            gamma
        )
        last_loss = loss  # store last loss (optional)

    return online_params, opt_state, last_loss

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
    num_epochs=10):
    """
    Train the online network for num_epochs over the offline dataset.
    Returns: updated online_params, opt_state
    """
    losses = []

    for epoch in range(num_epochs):
        # Create a unique RNG key for each epoch
        rng, epoch_rng = jax.random.split(rng)

        # Train one epoch
        online_params, opt_state, last_loss = train_one_epoch(
            online_params,
            target_params,
            opt_state,
            states,
            actions,
            rewards,
            next_states,
            batch_size,
            epoch_rng,
            gamma
        )

        losses.append(float(last_loss))

    return online_params, opt_state, losses

def update_target_network(online_params):
    """
    Hard update: make target network equal to the new online network.
    """
    return copy.deepcopy(online_params)


# In[22]:


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

def run_sim(key, N, T, QNN_params, temp):
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


        def update_frames(prev_frames, curr_vis_states):
            retained_frames = prev_frames[:,1:,:]
            return jnp.concatenate([retained_frames, curr_vis_states[:,None,:]], axis = 1)

        curr_frames = update_frames(prev_frames, curr_vis_states)

        # 2) Q-values for each agent
        agents_q_values = q_values(QNN_params, curr_frames)

        agent_keys = jax.random.split(subkey, N)   # N keys

        def select_action(qv, k):

            return boltzman_selection(qv, k, temp)

        actions = jax.vmap(select_action)(agents_q_values, agent_keys)    # (N,)

        x1, y1, o1, v1 = apply_action(x, y, o, actions)

        next_vis_state_integers = jnp.array(compute_visual_states(x1, y1, o1))   # (N,)
        next_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(next_vis_state_integers)

        rewards = 1 - 4 * ((jnp.sum(next_vis_states, axis=1)/n_s - 0.5) ** 2)

        transition = (curr_frames, actions, rewards)

        new_carry = (key, x1, y1, o1, v1, curr_frames, params)

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
    Vectorized replacement for your Python-loop prepare_data.

    data is the output of run_sim():
      data[0] = curr_frames over time: (T, N, M, S) where M=memory_length, S=n_s
      data[1] = actions:              (T, N)
      data[2] = rewards:              (T, N)

    Returns:
      states:      (B, M, S)
      actions:     (B,)
      rewards:     (B,)
      next_states: (B, M, S)
    where B = 2 * N * (T - 1 - memory_length)
    """
    frames, actions, rewards = data[0], data[1], data[2]

    T = frames.shape[0]
    # time indices j = memory_length .. T-2 inclusive
    j = jnp.arange(memory_length, T - 1)

    # Gather along time
    states      = frames[j]       # (L, N, M, S)
    next_states = frames[j + 1]   # (L, N, M, S)
    act         = actions[j]      # (L, N)
    rew         = rewards[j]      # (L, N)

    # Flatten (time, agent) -> batch
    L, N = act.shape
    states      = states.reshape(L * N, *states.shape[2:])       # (L*N, M, S)
    next_states = next_states.reshape(L * N, *next_states.shape[2:])
    act         = act.reshape(L * N)
    rew         = rew.reshape(L * N)

    # Mirror augmentation: reverse sensors axis (last axis) for every frame
    states_m      = states[..., ::-1]
    next_states_m = next_states[..., ::-1]
    act_m         = mirror_actions(act)

    # Duplicate dataset with mirrored samples
    states_out      = jnp.concatenate([states, states_m], axis=0)
    next_states_out = jnp.concatenate([next_states, next_states_m], axis=0)
    actions_out     = jnp.concatenate([act, act_m], axis=0)
    rewards_out     = jnp.concatenate([rew, rew], axis=0)

    return states_out, actions_out, rewards_out, next_states_out


# name the directory we are saving networks and losses to
# Initialize online network parameters
q_model = QNetwork(num_actions=num_actions)
# pick the optimizer
optimizer = optax.adamw(
    learning_rate= 3e-4,
    weight_decay=1e-5
)

# finish = 15

# chk = load_checkpoint("CheckpointsN=" + str(N) + "/cycle_" + str(finish) + ".ckpt")
# online_params = chk["online_params"]
# target_params = chk["target_params"]
# opt_state = optimizer.init(online_params)

def train(key, online_params, target_params, opt_state, start, end, simulations, save_dir, T_0, T_min, decay, gamma):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for cycle in range(start, end):

        temp = T_min + (T_0-T_min) * jnp.exp(-decay * cycle/simulations)

        key, sim_key, training_key = jax.random.split(key, 3)
        data = run_sim(sim_key, N, T, online_params, temp)

        states, actions, rewards, next_states = prepare_data(data, memory_length)

        online_params, opt_state, losses = train_for_epochs(
            training_key,
            online_params,
            target_params,
            opt_state,
            states,
            actions,
            rewards,
            next_states,
            gamma,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        # save losses
        target_params = update_target_network(online_params)
        if cycle % 5 == 0:
            t1 = time.time()
            tpc = (t1-t0)/(cycle+1)
            print(f"\n=== Training Cycle {cycle+1} ===")
            print(f"\n=== Temperature = {temp} ===")
            print(f"Average Loss {jnp.mean(jnp.array(losses))}")
            print ("Time remaining:", round(tpc * (simulations - cycle)/60,2))

            jnp.save(save_dir / f"losses_cycle_{cycle:06d}.npy", jnp.array(losses))
            # update the target network
            # save the network
            save_checkpoint(save_dir, online_params, target_params, opt_state, cycle)

    return key, online_params, target_params, opt_state

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

    if args.start == 0:
        dummy_frames = jnp.zeros((1, memory_length, n_s), dtype=jnp.float32)
        online_params = q_model.init(key, dummy_frames)
        target_params = update_target_network(online_params)
        opt_state = optimizer.init(online_params)

    else:
        save_dir = Path(args.save_dir)

        chk = load_checkpoint(save_dir, args.start -1)

        online_params = chk["online_params"]
        target_params = chk["target_params"]
        opt_state = optimizer.init(online_params)

    key, online_params, target_params, opt_state = train(
        key, online_params, target_params, opt_state, args.start, args.end, simulations, args.save_dir, T_0, T_min, decay, discount_factor)

if __name__ == "__main__":
    main()
