import numpy as onp

import jax
import jax.numpy as jnp
from jax.random import split
from jax import jit, grad, value_and_grad, random, lax
from jax.experimental import optix

import matplotlib.pyplot as plt


def flatten(array, dims=None):
    shape = array.shape
    if dims is None:
        return array.reshape(-1)
    elif isinstance(dims, tuple):
        assert (0 <= dims[0] < len(shape)) and (0 <= dims[1] < len(shape))
        final_shape = (
            *shape[: dims[0]],
            onp.prod(shape[dims[0] : dims[1] + 1]),
            *shape[dims[1] + 1 :],
        )
        return array.reshape(final_shape)
    else:
        assert 0 <= dims < len(shape)
        final_shape = (onp.prod(shape[: dims + 1]), *shape[dims + 1 :])
        return array.reshape(final_shape)


@jit
def xe_loss(logits, targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)


@jit
def mean_xe_loss(logits, targets):
    return -jnp.take_along_axis(
        jax.nn.log_softmax(logits), targets[..., None], axis=-1
    ).mean()


@jit
def xe_and_acc(logits, targets):
    acc = (logits.argmax(1) == targets).astype(jnp.float32)
    return xe_loss(logits, targets), acc


def make_inner_loop_fn(loss_acc_fn, opt_update_fn):
    def inner_loop(rln_params, pln_params, x_spt, y_spt, opt_state):
        for i, (_x, _y) in enumerate(zip(x_spt, y_spt)):
            (loss, acc), grads = value_and_grad(loss_acc_fn, 1, has_aux=True)(
                rln_params, pln_params, _x, _y
            )
            if i == 0:
                initial_loss = loss
                initial_acc = acc
            updates, opt_state = opt_update_fn(grads, opt_state, pln_params)
            pln_params = optix.apply_updates(pln_params, updates)
        return (
            pln_params,
            {
                "initial_loss": initial_loss,
                "initial_acc": initial_acc,
                "final_loss": loss,
                "final_acc": acc,
                "opt_state": opt_state,
            },
        )

    return inner_loop
