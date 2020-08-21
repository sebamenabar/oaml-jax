from tqdm import tqdm
import configargparse

import pickle
import numpy as onp

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import ops, nn, jit, grad, value_and_grad, lax, vmap, random, numpy as jnp

from jax.experimental import stax
from jax.experimental import optix
from jax.experimental import optimizers

import haiku as hk

from lib import xe_and_acc, flatten
from miniimagenet.models import MiniImagenetCNNMaker


mean = onp.array([120.04989, 114.6117, 102.67341])
std = onp.array([72.4941, 70.19553, 74.02528])


# def loss_acc_state_fn(slow_params, fast_params, state, is_training, inputs, targets):
#     # params = hk.data_structures.merge(slow_params, fast_params)
#     params = {**slow_params, **fast_params}
#     logprobs, state = MiniImagenetCNN.apply(params, state, None, inputs, is_training)
#     loss, acc = xe_and_acc(logprobs, targets)
#     return loss.mean(), (acc.mean(), state)


def make_inner_loop(loss_acc_state_fn, opt_update_fn):
    def inner_loop(
        slow_params, fast_params, state, x_spt, y_spt, opt_state, num_steps, is_training
    ):
        for i in range(num_steps):
            (loss, (acc, state)), grads = value_and_grad(
                loss_acc_state_fn, 1, has_aux=True,
            )(slow_params, fast_params, state, is_training, x_spt, y_spt)
            if i == 0:
                initial_loss = loss
                initial_acc = acc
            updates, opt_state = opt_update_fn(grads, opt_state, fast_params)

            fast_params = optix.apply_updates(fast_params, updates)
        final_loss, (final_acc, _) = loss_acc_state_fn(
            slow_params, fast_params, state, False, x_spt, y_spt
        )
        return (
            fast_params,
            state,
            {
                "initial_loss": initial_loss,
                "initial_acc": initial_acc,
                "final_loss": final_loss,
                "final_acc": final_acc,
                "opt_state": opt_state,
            },
        )

    return inner_loop


def make_outer_loss_fn(loss_acc_state_fn, inner_loop_fn):
    def outer_loss_fn(
        slow_params, fast_params, state, inner_opt_state, x_spt, y_spt, x_qry, y_qry, num_steps,
    ):
        initial_loss, (initial_acc, _) = loss_acc_state_fn(
            slow_params, fast_params, state, False, x_qry, y_qry
        )
        # inner_opt_state = inner_opt_init(fast_params)
        fast_params, state, inner_info = inner_loop_fn(
            slow_params,
            fast_params,
            state,
            x_spt,
            y_spt,
            inner_opt_state,
            num_steps,
            True,
        )
        final_loss, (final_acc, state) = loss_acc_state_fn(
            slow_params, fast_params, state, True, x_qry, y_qry
        )
        return (
            final_loss,
            (
                state,
                {
                    "inner": inner_info,
                    "outer": {
                        "initial_loss": initial_loss,
                        "initial_acc": initial_acc,
                        "final_loss": final_loss,
                        "final_acc": final_acc,
                    },
                },
            ),
        )

    return outer_loss_fn


if __name__ == "__main__":

    default_platform = "cpu"
    jax.config.update("jax_platform_name", default_platform)

    gpu = jax.devices("gpu")[0]
    cpu = jax.devices("cpu")[0]

    rng = random.PRNGKey(0)

    with open(
        "/workspace1/samenabar/data/mini-imagenet/mini-imagenet-cache-train.pkl", "rb"
    ) as f:
        train_data = pickle.load(f)

    images = jax.device_put(train_data["image_data"], cpu)
    labels = jax.device_put(onp.tile(onp.arange(64).reshape(64, 1), 600), cpu)

    mean = jax.device_put(mean, gpu)
    std = jax.device_put(std, gpu)

    images = images.reshape(64, 600, 84, 84, 3)
    labels = labels.reshape(64, 600)

    MiniImagenetCNN = hk.transform_with_state(
        lambda x, is_training: MiniImagenetCNNMaker(output_size=20,)(x, is_training)
    )

    inner_opt = optix.chain(optix.sgd(1e-2))
    inner_loop_fn = make_inner_loop(loss_acc_state_fn, inner_opt.update)

    params, state = MiniImagenetCNN.init(rng, jnp.zeros((2, 84, 84, 3)), True)
    params = jax.tree_map(lambda x: jax.device_put(x, gpu), params)
    state = jax.tree_map(lambda x: jax.device_put(x, gpu), state)

    # slow_params = params
    # fast_params = params["mini_imagenet_cnn/linear"]
    # predicate = lambda m, n, v: m == 'mini_imagenet_cnn/linear'
    predicate = lambda m, n, v: True
    # fast_params, slow_params = hk.data_structures.partition(predicate, params)
    inner_opt_state = inner_opt.init(fast_params)

    outer_opt_init, outer_opt_update, outer_get_params = optimizers.adam(step_size=1e-2)
    outer_opt_state = outer_opt_init(params)
    num_inner_steps = 1

    way = 5
    shot = 1

    x_spt, true_labels = sample_classes_and_images(
        rng, images, labels, way, shot * 2, platform=gpu
    )
    mapped_labels = jax.device_put(jnp.arange(way), gpu)[:, None].repeat(shot * 2, 1)
    x_qry = x_spt[:, shot:].reshape(way * shot, 84, 84, 3)
    y_qry = mapped_labels[:, shot:].reshape(way * shot)
    x_spt = x_spt[:, :shot].reshape(way * shot, 84, 84, 3)
    y_spt = mapped_labels[:, :shot].reshape(way * shot)

    def wrapper(params, *args, **kwargs):
        fast_params, slow_params = hk.data_structures.partition(predicate, params)
        return outer_loop_fn(slow_params, fast_params, *args, **kwargs)

    def step(rng, i, opt_state, state):

        x_spt, true_labels = sample_classes_and_images(
            rng, images, labels, way, shot * 2, platform=gpu
        )
        mapped_labels = jax.device_put(
            jnp.arange(way)[:, None].repeat(shot * 2, 1), gpu
        )
        x_qry = x_spt[:, shot:].reshape(way * shot, 84, 84, 3)
        y_qry = mapped_labels[:, shot:].reshape(way * shot)
        x_spt = x_spt[:, :shot].reshape(way * shot, 84, 84, 3)
        y_spt = mapped_labels[:, :shot].reshape(way * shot)

        outer_loop_fn = make_outer_loss_fn(
            loss_acc_state_fn, inner_opt.init, inner_loop_fn
        )

        params = outer_get_params(opt_state)
        # slow_params = params
        # fast_params = params["mini_imagenet_cnn/linear"]

        # (outer_loss, (state, info)), grads = value_and_grad(outer_loop_fn, has_aux=True)(params, fast_params, state, x_spt, y_spt, x_qry, y_qry, num_inner_steps)
        (outer_loss, (state, info)), grads = value_and_grad(wrapper, has_aux=True)(
            params, state, x_spt, y_spt, x_qry, y_qry, num_inner_steps
        )

        opt_state = outer_opt_update(i, grads, opt_state)

        return opt_state, state, info

    num_outer_steps = 1000
    pbar = tqdm(range(num_outer_steps))
    # inner_opt_state = inner_opt_init(fast_params)
    inner_loop_fn = jit(inner_loop_fn, static_argnums=(6, 7))
    for i in pbar:
        fast_params, state, inner_info = inner_loop_fn(
            slow_params,
            fast_params,
            state,
            x_spt,
            y_spt,
            inner_opt_state,
            num_inner_steps,
            True,
        )
        inner_opt_state = inner_info["opt_state"]

        if i % 50 == 0:
            pbar.set_postfix(
                #  loss=f"{info['outer']['final_loss']:.3f}",
                iia=f"{inner_info['initial_acc']:.3f}",
                iil=f"{inner_info['initial_loss']:.3f}",
                fia=f"{inner_info['final_acc']:.3f}",
                fil=f"{inner_info['final_loss']:.3f}",
                # ioa=f"{info['outer']['initial_acc']:.3f}",
                # iol=f"{info['outer']['initial_loss']:.3f}",
                # foa=f"{info['outer']['final_acc']:.3f}",
                # fil=f"{info['inner']['final_loss']:.3f}",
            )

