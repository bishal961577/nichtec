"""Neural baseline on the same CPU: Zaremba et al. (2014) 'small' LSTM (2 x 200, no dropout),
trained with truncated backprop through time. Used to measure how much CPU time a neural
network needs to reach a given perplexity."""
import time

import jax
import jax.numpy as jnp
import numpy as np


def init(V, H=200, L=2, seed=0, scale=0.1):
    k = jax.random.split(jax.random.PRNGKey(seed), 2 * L + 2)
    u = lambda key, shape: jax.random.uniform(key, shape, jnp.float32, -scale, scale)
    p = {"emb": u(k[0], (V, H)), "out_w": u(k[1], (H, V)), "out_b": jnp.zeros(V)}
    for l in range(L):
        p[f"w{l}"] = u(k[2 + 2 * l], (2 * H, 4 * H))
        p[f"b{l}"] = jnp.zeros(4 * H)
    return p


def forward(p, x, state, L=2):
    """x: (B, T) token ids; state: tuple of (h, c) per layer. Returns logits (B, T, V), new state."""
    h_seq = p["emb"][x]  # (B, T, H)
    new_state = []
    for l in range(L):
        W, b = p[f"w{l}"], p[f"b{l}"]

        def cell(carry, xt):
            h, c = carry
            z = jnp.concatenate([xt, h], -1) @ W + b
            i, f, g, o = jnp.split(z, 4, -1)
            c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
            h = jax.nn.sigmoid(o) * jnp.tanh(c)
            return (h, c), h

        (h, c), hs = jax.lax.scan(cell, state[l], jnp.swapaxes(h_seq, 0, 1))
        h_seq = jnp.swapaxes(hs, 0, 1)
        new_state.append((h, c))
    return h_seq @ p["out_w"] + p["out_b"], tuple(new_state)


def loss_fn(p, x, y, state):
    logits, st = forward(p, x, state)
    lp = jax.nn.log_softmax(logits)
    return -jnp.take_along_axis(lp, y[..., None], -1).mean(), st


def _train_loss(p, x, y, state):
    # Zaremba et al. sum the loss over the unrolled time steps and average over the batch
    l, st = loss_fn(p, x, y, state)
    return l * x.shape[1], (l, st)


@jax.jit
def train_step(p, x, y, state, lr):
    (_, (l, st)), g = jax.value_and_grad(_train_loss, has_aux=True)(p, x, y, state)
    norm = jnp.sqrt(sum(jnp.sum(v * v) for v in jax.tree_util.tree_leaves(g)))
    scale = jnp.minimum(1.0, 5.0 / (norm + 1e-6))  # gradient clipping at 5, as in Zaremba et al.
    p = jax.tree_util.tree_map(lambda a, b: a - lr * scale * b, p, g)
    st = jax.tree_util.tree_map(jax.lax.stop_gradient, st)
    return p, st, l


@jax.jit
def eval_step(p, x, y, state):
    return loss_fn(p, x, y, state)


def batchify(ids, B):
    n = len(ids) // B
    return ids[: n * B].reshape(B, n)


def zeros_state(B, H=200, L=2):
    return tuple((jnp.zeros((B, H)), jnp.zeros((B, H))) for _ in range(L))


def evaluate(p, ids, T=35):
    data = batchify(ids, 1)
    st = zeros_state(1)
    tot, n = 0.0, 0
    for i in range(0, data.shape[1] - 1, T):
        x = data[:, i : i + T]
        y = data[:, i + 1 : i + 1 + T]
        x = x[:, : y.shape[1]]
        l, st = eval_step(p, jnp.array(x), jnp.array(y), st)
        tot += float(l) * y.shape[1]
        n += y.shape[1]
    return float(np.exp(tot / n))


def train(tr, va, V, epochs=13, B=20, T=20, lr=1.0, decay_after=4, decay=0.5, log=print, time_budget=None):
    p = init(V)
    data = batchify(tr, B)
    t0 = time.time()
    curve = []
    for ep in range(epochs):
        st = zeros_state(B)
        cur_lr = lr * (decay ** max(0, ep + 1 - decay_after))
        for i in range(0, data.shape[1] - 1 - T, T):
            p, st, l = train_step(p, jnp.array(data[:, i : i + T]), jnp.array(data[:, i + 1 : i + 1 + T]), st, cur_lr)
        vppl = evaluate(p, va)
        curve.append((ep + 1, time.time() - t0, vppl))
        log(f"epoch {ep + 1} cpu_seconds {time.time() - t0:.0f} valid_ppl {vppl:.1f}")
        if time_budget and time.time() - t0 > time_budget:
            break
    return p, curve


@jax.jit
def _step_logp(p, x, y, state):
    logits, st = forward(p, x, state)
    lp = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(lp, y[..., None], -1)[..., 0], st


def token_probs(p, ids, T=35):
    """P(ids[t] | ids[:t]) for t >= 1 (position 0 has no context and is given probability 1/V)."""
    data = batchify(ids, 1)
    st = zeros_state(1)
    out = []
    for i in range(0, data.shape[1] - 1, T):
        y = data[:, i + 1 : i + 1 + T]
        x = data[:, i : i + y.shape[1]]
        lp, st = _step_logp(p, jnp.array(x), jnp.array(y), st)
        out.append(np.exp(np.asarray(lp[0])))
    return np.concatenate(out)
