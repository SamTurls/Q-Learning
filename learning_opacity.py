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
discount_factor = 0.9
# number of agents in a simulation
N = 10
# length of time of simulation
T = 250
# number of training simulations
simulations = 50
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)

    # research parameters
    p.add_argument("--discount_factor", type=float, default=1.0)
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--memory_length", type=int, default=3)
    p.add_argument("--simulations", type=int, default=50)

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

# temporal convolution for time
class TemporalConv(nn.Module):
    features: int = 32
    kernel_size: int = memory_length

    @nn.compact
    def __call__(self, x):
        """
        x: (B, T, n_s, F)

        returns: (B, n_s, F')
        """
        # Step 1: Reorder axes so that time is grouped per angle
        # We want each angular position to have its own time sequence.
        #
        x = jnp.transpose(x, (0, 2, 1, 3))
        # Shape is now: (B, A, T, F)


        # Step 2: Treat each angle as independent
        # Conv1D expects input of shape: (batch, length, channels)
        # Here:
        #   length   = time dimension (T)
        #   channels = feature dimension (F)
        #
        # So we collapse (B, A) into a single batch dimension.
        # Each (b, a) pair is now one independent time sequence.

        B, A, T, F = x.shape
        x = x.reshape(B * A, T, F)
        # Shape is now: (B * A, T, F)


        # ---------------------------------------------------------
        # Step 3: Apply temporal convolution over time
        # ---------------------------------------------------------
        # This convolution slides over the time axis only.
        #
        # padding="VALID" means:
        #   - no fake past or future
        #   - output time length shrinks

        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding="VALID"
        )(x)
        x = nn.relu(x)
        # Shape is now: (B * A, T_out, features)


        # Step 4: Collapse the remaining time dimension
        # ---------------------------------------------------------
        # Because VALID padding shrinks time,
        # we now summarize the temporal window by taking the
        # most recent output (the rightmost time index).

        x = x[:, -1, :]
        # Shape is now: (B * A, features)


        # Step 5: Restore the original batch and angle structure
        x = x.reshape(B, A, self.features)
        # Final shape: (B, A, features)

        return x

class QNetwork(nn.Module):
    num_actions: int
    # size of feature vector of a visual state
    features: int = 32
    # size of smoothing averages
    kernel_size: int = 5
    # size of last layer
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, frames):
        """
        prev, curr: (B, T, n_s)
        returns: (B, num_actions)
        """
        encoder = RingConvEncoder(features=self.features)
        temporal = TemporalConv(features=self.features)

        # pass each frame into a feature encoder NN
        h = encode_frames(encoder, frames)      # (B, T, n_s, F)

        # pass the encoded frames into a CNN for temporal features
        h = temporal(h)                          # (B, n_s, F')

        #reshape ready to put into final layer
        h = h.reshape(h.shape[0], -1)            # flatten

        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)

        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)

        return nn.Dense(self.num_actions)(h)

def q_values(params, frames):
    """
    params: network parameters (online or target)
    states: array of shape (batch, n_s)
    returns: Q-values of shape (batch, num_actions)
    """
    return q_model.apply(params, frames)

def compute_td_target(target_params, rewards, next_frames, terminals, gamma=discount_factor):
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
    target = rewards + gamma * max_next_q * (1 - terminals)

    return target

@jax.jit

def train_step(online_params, target_params, opt_state,
               curr_frames, actions, rewards, next_frames, terminals):
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
            target_params, rewards, next_frames, terminals
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

def iterate_minibatches(states, actions, rewards, next_states, terminals, batch_size, rng):
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
    terminals = terminals[perm]

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
            terminals[start:end]
        )


def train_one_epoch(online_params, target_params, opt_state,
                    states, actions, rewards, next_states, terminals,
                    batch_size, epoch_rng):
    """
    Trains online_params for one full epoch over the entire offline dataset.
    Returns updated (online_params, opt_state) and the final loss observed.
    """

    last_loss = 0.0
    # Generate minibatches using the epoch-specific RNG
    for (batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals) in         iterate_minibatches(states, actions, rewards, next_states, terminals, batch_size, epoch_rng):

        # Perform one update on this minibatch
        online_params, opt_state, loss = train_step(
            online_params,
            target_params,
            opt_state,
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_terminals
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
    terminals,
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
            terminals,
            batch_size,
            epoch_rng
        )

        losses.append(float(last_loss))

        print(f"Epoch {epoch+1}/{num_epochs}, loss = {float(last_loss):.6f}")

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
    x_positions = jax.random.uniform(key1, (N,)) * N * (2 * size + 1)

    y_positions = jax.random.uniform(key2, (N,)) * N * (2 * size + 1)

    # Orientations: normal with mean 0, std=1, scaled by delta_theta
    orientations = jax.random.normal(key3, (N,)) * 2 * delta_theta

    velocities = jnp.full((N,), v0)

    return x_positions, y_positions, orientations, velocities

def apply_action(x, y, o, action):

    o_next = o + orientation[action]

    x_next = x + velocity[action] * jnp.cos(o_next) * dt
    y_next = y + velocity[action] * jnp.sin(o_next) * dt
    v_next = velocity[action]

    return x_next, y_next, o_next, v_next

def apply_action_expected(x, y, o, prob):
    """
    x, y, o: (N,)
    prob: (N, num_actions)
    """

    # Expected orientation change and velocity
    delta_o = jnp.sum(prob * orientation, axis=1)      # (N,)
    v_next  = jnp.sum(prob * velocity, axis=1)         # (N,)

    o_next = o + delta_o

    x_next = x + v_next * jnp.cos(o_next) * dt
    y_next = y + v_next * jnp.sin(o_next) * dt

    return x_next, y_next, o_next, v_next


def simulate_path(path, x, y, o, velocity, orientation, QNN_params, prev_frames, temp):

    # x, y, o contains everyone positions etc with the i agent first in the list

    def step_fn(carry, depth):

        x, y, o, prev_frames, collision_flag = carry

        # work out the Q-NN actions for all agents other than agent i, who is following the actions from path

        # get the current visual states for all agents
        vis_state_integers = jnp.array(compute_visual_states(x, y, o))   # (N,)
        curr_vis_states = jax.vmap(lambda n: to_binary(n, n_s))(vis_state_integers)

        def update_frames(prev_frames, curr_vis_states):
            retained_frames = prev_frames[:,1:,:]
            return jnp.concatenate([retained_frames, curr_vis_states[:,None,:]], axis = 1)

        curr_frames = update_frames(prev_frames, curr_vis_states)

        # Q-values for each agent apart from 0
        agents_q_values = q_values(QNN_params, curr_frames[1:])

        def get_probabilities(q_values):

            def exponentiate(q_value):
                return jnp.exp(q_value/temp)

            proportions = jax.vmap(exponentiate)(q_values)
            probabilities = proportions/jnp.sum(proportions)

            return probabilities

        agents_probabilites = jax.vmap(get_probabilities)(agents_q_values)

        agent_i_action = path[depth]

        def action_to_prob(a, num_actions):
            prob = jnp.zeros(num_actions)
            prob = prob.at[a].set(1.0)
            return prob

        agent_i_prob = action_to_prob(agent_i_action, num_actions)

        probs = jnp.concatenate([agent_i_prob[None,:], agents_probabilites], axis = 0)

#         x.debug.print("{}", actions)

        # 4) environment dynamics
        x_next, y_next, o_next, v_next = apply_action_expected(
            x, y, o, probs)  # each (N,)

        # 5) positions at this next step
        positions = jnp.stack([x, y], axis=-1)
        positions_next = jnp.stack([x_next, y_next], axis=-1)    # (N, 2)

        # Compute whether a collision occurs at this step
        collision_now = check_collision(x[0],y[0], x_next[0], y_next[0], positions[1:], positions_next[1:])

        # Update cumulative collision flag (once collision, always collision)
        collision_flag_next = jnp.maximum(collision_flag, collision_now)

        vs = get_visual_state(x_next[0], y_next[0], o_next[0], positions_next[1:])

        new_carry = (x_next, y_next, o_next, curr_frames, collision_flag_next)
        metrics = (vs, collision_flag_next)
        return new_carry, metrics

    init_carry = (x, y, o, prev_frames, 0)

    step_fn = remat(step_fn)

    _, metrics = jax.lax.scan(step_fn, init_carry, jnp.arange(tau))


    vs_trajectory, collision_trajectory = metrics

    return vs_trajectory, collision_trajectory[-1]

def check_collision(x_old, y_old, x_new, y_new, assumed_positions_old, assumed_positions_new, threshold=2.0):

    """
    Calculate whether there is a collision between actions
    """
    relx_new = assumed_positions_new[:, 0] - x_new
    rely_new = assumed_positions_new[:, 1] - y_new

    relx_old = assumed_positions_old[:, 0] - x_old
    rely_old = assumed_positions_old[:, 1] - y_old

    dx, dy   = (relx_old - relx_new), (rely_old - rely_new)  # segment direction toward "old"

    dd   = dx * dx + dy * dy

    r0d  = relx_new * dx + rely_new * dy

    eps  = 1e-12
    safe = dd > eps
    t_star = jnp.where(safe, -r0d / dd, 0.0)  # dummy where degenerate, will be masked out

    strictly_between = (t_star >= 0.0) & (t_star < 1.0)

    cx = relx_new + t_star * dx
    cy = rely_new + t_star * dy
    c2 = cx * cx + cy * cy
    interior_close = strictly_between & (c2 <= threshold * threshold)

    return jnp.any(interior_close)

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

simulate_paths_vmap = jax.vmap(simulate_path, in_axes=(0, None, None, None, None, None, None, None,None))

def argmax_random_tie(q_values, key):
    """
    q_values: shape (..., A)
    key: PRNGKey
    returns: indices of shape (...) with random tie-breaking over argmax
    """
    # 1. max per row / vector
    max_q = jnp.max(q_values, axis=-1, keepdims=True)

    # 2. mask of max positions
    is_max = (q_values == max_q)

    # 3. random noise for every entry
    noise = jax.random.uniform(key, shape=q_values.shape)

    # 4. keep noise only on max entries, zero elsewhere
    #    (non-max entries are exactly 0; max entries are in (0,1))
    scores = jnp.where(is_max, noise, 0.0)

    # 5. argmax over the (possibly batched) last dimension
    return jnp.argmax(scores, axis=-1)

def boltzman_selection(q_values, key, temp):

    def exponentiate(q_value):
        return jnp.exp(q_value/temp)

    proportions = jax.vmap(exponentiate)(q_values)
    probabilities = proportions/jnp.sum(proportions)

    action = jax.random.choice(key, a = jnp.arange(0,num_actions), p = probabilities)

    return action

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

        rewards = jnp.where(collisions == 1, -1.0, rewards)

        transition = (curr_frames, actions, rewards, collisions)

        new_carry = (key, x1, y1, o1, v1, curr_frames, params)

        return new_carry, transition

    carry_final, transitions = jax.lax.scan(step, carry0,jnp.arange(T),)

    return transitions   # see shapes below

def entropy(vs_trajectory, collision_flags, sentinel=-1):
    """
    Global frequency of visual states over ALL paths (and all tau steps),
    excluding collided paths.

    vs_trajectory: (P, tau) int32/int64
    collision_flags: (P,) 0/1 or bool; 1 means collided

    Returns
    -------
    values: (M,) int     sorted unique visual-state IDs, padded with sentinel
    counts: (M,) int32   counts aligned with values, padded with 0
    num_unique: () int32 how many entries in (values, counts) are valid
    total: () int32      total number of counted samples (non-collided entries)
    where M = P * tau
    """
    P, tau = vs_trajectory.shape

    # Mask collided paths out entirely
    keep = (collision_flags == 0)                # (P,)
    masked = jnp.where(keep[:, None], vs_trajectory, sentinel)  # (P, tau)

    flat = masked.reshape(-1)                    # (M,)
    M = flat.size

    # Sort so equal values are contiguous
    s = jnp.sort(flat)                           # (M,)

    # Run starts (True at i=0 and where value changes)
    is_start = jnp.concatenate([jnp.array([True]), s[1:] != s[:-1]])  # (M,)

    # Run id for each element: 0,0,0,1,1,2,...
    run_id = jnp.cumsum(is_start) - 1            # (M,), in [0, M-1]

    # Count occurrences per run (static length M)
    counts = jnp.zeros((M,), dtype=jnp.int32).at[run_id].add(1)

    # Store the run value once per run (static length M)
    run_vals = jnp.full((M,), sentinel, dtype=s.dtype).at[run_id].set(s)

    # Exclude sentinel runs
    valid = (run_vals != sentinel) & (counts > 0)

    values = jnp.where(valid, run_vals, sentinel)
    counts = jnp.where(valid, counts, 0)

    num_unique = jnp.sum(valid).astype(jnp.int32)
    total = jnp.sum(counts).astype(jnp.int32)

    total_f = jnp.maximum(total.astype(jnp.float32), 1.0)

    p = counts.astype(jnp.float32) / total_f
    nz = counts > 0
    return -jnp.sum(jnp.where(nz, p * jnp.log(p), 0.0))

    return values, counts, num_unique, total

def possible_paths(first_action, num_actions, tau):
    """
    first_action : scalar int, in [0, num_actions)
    num_actions  : total available discrete actions
    tau          : length of each path

    Returns
    -------
    paths : (num_paths, tau) int32
        All possible action sequences of length `tau`,
        with paths[:, 0] == first_action.
    """
    if tau == 1:
        # Only one step: just the already chosen action
        return jnp.array([[first_action]], dtype=jnp.int32)

    # Shape for the remaining tau-1 steps: (num_actions, num_actions, ..., num_actions)
    # length of this tuple is tau-1
    grid_shape = (num_actions,) * (tau - 1)

    # jnp.indices(grid_shape) has shape (tau-1, num_actions, ..., num_actions)
    # Each "row" along axis 0 represents the values along one time index.
    grid = jnp.indices(grid_shape)  # (tau-1, num_actions, ..., num_actions)

    # Flatten all the combination dimensions into one axis
    # grid.reshape(tau-1, -1) has shape (tau-1, num_paths)
    rest_actions = grid.reshape(tau - 1, -1).transpose(1, 0)  # (num_paths, tau-1)

    num_paths = rest_actions.shape[0]

    # Column of the fixed first_action, shape (num_paths, 1)
    first_col = jnp.full((num_paths, 1), first_action, dtype=jnp.int32)

    # Concatenate to get full paths: (num_paths, tau)
    paths = jnp.concatenate([first_col, rest_actions], axis=1)

    return paths

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
      data[3] = collisions/terminals: (T, N)  (you treat this as terminals)

    Returns:
      states:      (B, M, S)
      actions:     (B,)
      rewards:     (B,)
      next_states: (B, M, S)
      terminals:   (B,)
    where B = 2 * N * (T - 1 - memory_length)
    """
    frames, actions, rewards, terminals = data[0], data[1], data[2], data[3]

    T = frames.shape[0]
    # time indices j = memory_length .. T-2 inclusive
    j = jnp.arange(memory_length, T - 1)

    # Gather along time
    states      = frames[j]       # (L, N, M, S)
    next_states = frames[j + 1]   # (L, N, M, S)
    act         = actions[j]      # (L, N)
    rew         = rewards[j]      # (L, N)
    term        = terminals[j]    # (L, N)

    # Flatten (time, agent) -> batch
    L, N = act.shape
    states      = states.reshape(L * N, *states.shape[2:])       # (L*N, M, S)
    next_states = next_states.reshape(L * N, *next_states.shape[2:])
    act         = act.reshape(L * N)
    rew         = rew.reshape(L * N)
    term        = term.reshape(L * N)

    # Mirror augmentation: reverse sensors axis (last axis) for every frame
    states_m      = states[..., ::-1]
    next_states_m = next_states[..., ::-1]
    act_m         = mirror_actions(act)

    # Duplicate dataset with mirrored samples
    states_out      = jnp.concatenate([states, states_m], axis=0)
    next_states_out = jnp.concatenate([next_states, next_states_m], axis=0)
    actions_out     = jnp.concatenate([act, act_m], axis=0)
    rewards_out     = jnp.concatenate([rew, rew], axis=0)
    terminals_out   = jnp.concatenate([term, term], axis=0)

    return states_out, actions_out, rewards_out, next_states_out, terminals_out


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

def train(key, online_params, target_params, opt_state, start, end, simulations, save_dir, T_0, T_min, decay):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for cycle in range(start, end):

        temp = T_min + (T_0-T_min) * jnp.exp(-cycle/(simulations*decay))

        print(f"\n=== Training Cycle {cycle+1} ===")
        print(f"\n=== Temperature = {temp} ===")

        key, sim_key, training_key = jax.random.split(key, 3)
        data = run_sim(sim_key, N, T, online_params, temp)

        states, actions, rewards, next_states, terminals = prepare_data(data, memory_length)

        online_params, opt_state, losses = train_for_epochs(
            training_key,
            online_params,
            target_params,
            opt_state,
            states,
            actions,
            rewards,
            next_states,
            terminals,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        # save losses
        jnp.save(save_dir / f"losses_cycle_{cycle:06d}.npy", jnp.array(losses))
        # update the target network
        target_params = update_target_network(online_params)
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

    T_0 = 1/(1-discount_factor)
    T_min = 40/((n_s * n_s)) * T_0

    decay = 1/ (2*jnp.log((T_0/T_min) -1))

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
        key, online_params, target_params, opt_state, args.start, args.end, simulations, args.save_dir, T_0, T_min, decay)

if __name__ == "__main__":
    main()
