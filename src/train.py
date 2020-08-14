import configargparse
from tqdm import tqdm

import jax
from jax.random import split
from jax.tree_util import Partial as partial
from jax import ops, nn, jit, grad, value_and_grad, lax, vmap, random, numpy as jnp

from jax.experimental import optix
from jax.experimental import optimizers
from jax.experimental import stax

from test import test
from omniglot.data import *
from omniglot import models
from lib import xe_and_acc, flatten


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


def make_outer_loop_loss_fn(loss_acc_fn, inner_opt_init, inner_loop_fn):
    def outer_loss_fn(rln_params, pln_params, x_spt, y_spt, x_qry, y_qry):
        initial_loss, initial_acc = loss_acc_fn(rln_params, pln_params, x_qry, y_qry,)
        inner_opt_state = inner_opt_init(pln_params)
        pln_params, inner_info = inner_loop_fn(
            rln_params, pln_params, x_spt, y_spt, inner_opt_state,
        )
        final_loss, final_acc = loss_acc_fn(rln_params, pln_params, x_qry, y_qry)
        return (
            final_loss,
            {
                "inner": inner_info,
                "outer": {
                    "initial_loss": initial_loss,
                    "initial_acc": initial_acc,
                    "final_loss": final_loss,
                    "final_acc": final_acc,
                },
            },
        )

    return outer_loss_fn


def load_data(size=28, train_size=600, val_size=364):
    if size == 28:
        train_images, train_labels = prepare_omniglot_data28("train", normalize=False)
    elif size == 84:
        train_images, train_labels = prepare_omniglot_data84("train", normalize=False)

    train_images = train_images[..., None]  # Unsqueeze

    val_images, val_labels = train_images[-val_size:], train_labels[-val_size:]
    train_images, train_labels = train_images[:train_size], train_labels[:train_size]

    mean, std = train_images.mean(), train_images.std()
    print()
    print("Mean and stddev used for normalization")
    print(mean, std)

    train_images = (train_images - mean) / std
    val_images = (val_images - mean) / std

    print()
    print("Train split")
    print("Images and labels shape")
    print(train_images.shape, train_labels.shape)
    print("Mean and stddev")
    print(train_images.mean(), train_images.std())

    print()
    print("Val split")
    print("Images and labels shape")
    print(val_images.shape, val_labels.shape)
    print("Mean and stddev")
    print(val_images.mean(), val_images.std())
    print()

    return train_images, train_labels, val_images, val_labels


def make_flat_set(arr):
    return flatten(arr, 1)


class Parser(configargparse.Parser):
    def __init__(self):
        super().__init__()
        self.add("--steps", type=int, help="Number of outer steps", default=20000)
        self.add(
            "--image_size", type=int, help="Image size", default=28, choices=[28, 84]
        )
        self.add("--num_tasks_per_step", type=int, default=1)
        self.add("--num_samples_per_task", type=int, default=20)
        self.add("--num_outer_samples", type=int, default=64)
        self.add("--inner_lr", type=float, default=1e-2)
        self.add("--inner_lr_val", type=float, default=1e-3)
        self.add("--outer_lr", type=float, default=1e-3)
        # self.add("--num_rln_layers", type=float, default=13)
        self.add("--seed", type=int, default=0)
        self.add("--num_train_tasks", type=int, default=600)
        self.add("--num_val_tasks", type=int, default=364)
        self.add("--treatment", type=str, default="OML", choices=["OML", "NM"])
        self.add("--num_fc_layers", type=int, default=2, help="Only aplicable to ANML")
        self.add(
            "--num_rln_layers",
            type=int,
            default=13,
            help="Number of layers not modified in inner updates",
        )
        self.add(
            "--num_val_rln_layers",
            type=int,
            default=13,
            help="Number of layers not modified in inner updates during evaluation",
        )
        self.add(
            "--no_reset",
            action="store_true",
            help="Disable reset of classification layer before each inner loop",
        )
        self.add("--progress_bar_refresh_rate", type=int, default=50)


if __name__ == "__main__":
    args = Parser().parse_args()
    print(args)

    print("Using default device:", jax.devices()[0])
    rng = random.PRNGKey(args.seed)
    size = args.image_size
    num_tasks_per_step = args.num_tasks_per_step
    num_inner_samples = args.num_samples_per_task
    num_outer_samples = args.num_outer_samples

    data = [
        jax.device_put(d)
        for d in load_data(
            size, train_size=args.num_train_tasks, val_size=args.num_val_tasks
        )
    ]
    train_images_flat, train_labels_flat, val_images_flat, val_labels_flat = [
        make_flat_set(arr) for arr in data
    ]
    train_images, train_labels, val_images, val_labels = data
    tasks = jnp.arange(train_images.shape[0])

    net_init, net_apply = models.make_oml_net(size, num_fc_layers=args.num_fc_layers)

    def net_forward(rln_params, pln_params, inputs):
        return net_apply(rln_params + pln_params, inputs)

    def loss_acc_fn(rln_params, pln_params, inputs, targets):
        logprobs = net_forward(rln_params, pln_params, inputs)
        loss, acc = xe_and_acc(logprobs, targets)
        return loss.mean(), acc.mean()

    inner_loop_sampler = partial(
        sample_tasks_and_samples,
        # images=train_images,
        # labels=train_labels,
        tasks=tasks,
        num_tasks=num_tasks_per_step,
        num_samples=num_inner_samples,
        shuffle=True,
    )

    outer_loop_sampler = partial(
        random_samples,
        # images=flatten(train_images, 1),
        # labels=flatten(train_labels, 1),
        num_samples=num_outer_samples,
    )

    inner_opt = optix.chain(optix.sgd(args.inner_lr))
    inner_loop_fn = make_inner_loop_fn(loss_acc_fn, inner_opt.update)
    outer_loop_loss_fn = make_outer_loop_loss_fn(
        loss_acc_fn, inner_opt.init, inner_loop_fn
    )

    rng, rng_net = split(rng)
    (out_shape), params = net_init(rng_net, (-1, size, size, 1))

    rln_params, pln_params = (
        params[: args.num_rln_layers],
        params[args.num_rln_layers :],
    )

    outer_opt_init, outer_opt_update, outer_get_params = optimizers.adam(
        step_size=args.outer_lr
    )
    outer_opt_state = outer_opt_init((rln_params, pln_params))

    # @jit
    def step(rng, i, opt_state):
        rng_spt, rng_qry, rng_reinit = split(rng, 3)
        rln_params, pln_params = outer_get_params(opt_state)
        x_spt, y_spt, sampled_tasks = inner_loop_sampler(
            rng_spt, train_images, train_labels
        )

        if not args.no_reset:
            for sampled_task in sampled_tasks:
                cls_w = pln_params[-2][0]
                pln_params[-2] = (
                    ops.index_update(
                        cls_w,
                        ops.index[:, [sampled_task]],
                        nn.initializers.he_normal()(rng_reinit, (cls_w.shape[0], 1)),
                    ),  # Reset W
                    *pln_params[-2][1:],  # Keep bias (?)
                )

        x_qry, y_qry = outer_loop_sampler(rng_qry, train_images_flat, train_labels_flat)
        x_spt, y_spt = flatten(x_spt, 1), flatten(y_spt, 1)
        x_qry, y_qry = (
            jnp.concatenate((x_spt, x_qry), 0),
            jnp.concatenate((y_spt, y_qry), 0),
        )
        x_spt, y_spt = x_spt[:, None, ...], y_spt[:, None, ...]

        (outer_loss, info), outer_grads = value_and_grad(
            outer_loop_loss_fn, argnums=(0, 1), has_aux=True
        )(rln_params, pln_params, x_spt, y_spt, x_qry, y_qry)

        opt_state = outer_opt_update(i, outer_grads, opt_state)

        return opt_state, info

    step = jit(step)
    num_outer_steps = args.steps
    pbar = tqdm(range(num_outer_steps))
    net_loss = jnp.zeros(num_outer_steps)
    for i in pbar:
        rng, rng_task = split(rng, 2)
        outer_opt_state, info = step(rng_task, i, outer_opt_state)

        if i % args.progress_bar_refresh_rate == 0:
            pbar.set_postfix(
                loss=f"{info['outer']['final_loss']:.3f}",
                iia=f"{info['inner']['initial_acc']:.3f}",
                iil=f"{info['inner']['initial_loss']:.3f}",
                fia=f"{info['inner']['final_acc']:.3f}",
                fil=f"{info['inner']['final_loss']:.3f}",
                ioa=f"{info['outer']['initial_acc']:.3f}",
                iol=f"{info['outer']['initial_loss']:.3f}",
                foa=f"{info['outer']['final_acc']:.3f}",
                # fil=f"{info['inner']['final_loss']:.3f}",
            )
        net_loss = ops.index_update(net_loss, i, info["outer"]["final_loss"])

    net_params = sum(outer_get_params(outer_opt_state), start=[])
    rng, rng_net = split(rng)
    _, new_pln_params = net_init(rng_net, (-1, size, size, 1))
    new_pln_params = new_pln_params[args.num_val_rln_layers :]
    new_rln_params = net_params[: args.num_val_rln_layers]

    test_train_images = train_images[:, :15]
    test_train_labels = train_labels[:, :15]
    test_test_images = train_images[:, 15:]
    test_test_labels = train_labels[:, 15:]

    for num_tasks in [50, 100, 200, 364]:

        print()
        print("----- Evaluating %d continual tasks ------" % num_tasks)

        test(
            rng,
            num_tasks,
            loss_acc_fn,
            new_rln_params,
            new_pln_params,
            args.inner_lr_val,
            test_train_images,
            test_train_labels,
            test_test_images,
            test_test_labels,
        )

