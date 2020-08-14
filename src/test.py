import numpy as onp
from tqdm import tqdm

from jax.random import split
from jax.experimental import optix
from jax.tree_util import Partial as partial
from jax import jit, random, numpy as jnp, vmap, ops

from lib import make_inner_loop_fn


def test(
    rng,
    num_tasks,
    loss_acc_fn,
    rln_params,
    pln_params,
    lr,
    test_train_images,
    test_train_labels,
    test_test_images,
    test_test_labels,
):
    print("Testing")
    opt = optix.adam(lr)
    inner_loop = jit(make_inner_loop_fn(loss_acc_fn, opt.update))
    opt_state = opt.init(pln_params)

    rng, rng_tasks = split(rng)

    tasks = jnp.arange(0, test_train_images.shape[0])
    sampled_tasks = random.choice(rng_tasks, tasks, (num_tasks,), replace=False)

    def run_eval(rln_params, pln_params, images, labels):
        means = jnp.zeros(images.shape[0])
        for i in range(images.shape[0]):
            task_mean = loss_acc_fn(rln_params, pln_params, images[i], labels[i])[1].mean()
            means = ops.index_update(means, i, task_mean)
        # return task_mean = vmap(partial(loss_acc_fn, rln_params, pln_params))(images, labels,)[
        #     1
        # ].mean(-1)
        return means.mean()

    print("Accuracy before training:")
    print(
        "Train:",
        run_eval(
            rln_params,
            pln_params,
            test_train_images[sampled_tasks],
            test_train_labels[sampled_tasks],
        ),
    )
    print(
        "Test:",
        run_eval(
            rln_params,
            pln_params,
            test_test_images[sampled_tasks],
            test_test_labels[sampled_tasks],
        ),
    )

    orders = jnp.arange(0, test_train_images.shape[1])
    for task_num in tqdm(sampled_tasks):
        rng, rng_shuffle = split(rng)
        for sample_num in random.permutation(rng_shuffle, orders):
            pln_params, inner_info = inner_loop(
                rln_params,
                pln_params,
                # lax.dynamic_slice(val_train_images, (task_num, sample_num, 0, 0, 0), (1, 1, *val_train_images.shape[2:])),
                # lax.dynamic_slice(val_train_labels, (task_num, sample_num), (1, 1)),
                test_train_images[task_num : task_num + 1, [sample_num]],
                test_test_labels[task_num : task_num + 1, [sample_num]],
                # flatten(val_train_images[:num_test_tasks], 1)[:, None, ...],
                # flatten(val_train_labels[:num_test_tasks], 1)[:, None, ...],
                opt_state,
            )
            opt_state = inner_info["opt_state"]
            # pbar.update()

    print("Accuracy after training:")
    print(
        "Train:",
        run_eval(
            rln_params,
            pln_params,
            test_train_images[sampled_tasks],
            test_train_labels[sampled_tasks],
        ),
    )
    print(
        "Test:",
        run_eval(
            rln_params,
            pln_params,
            test_test_images[sampled_tasks],
            test_test_labels[sampled_tasks],
        ),
    )

