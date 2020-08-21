import os.path as osp

import pickle
import numpy as onp

import jax
from jax import random, numpy as jnp
from jax.random import split


def shuffle_along_axis(rng, a, axis):
    idx = random.uniform(rng, a.shape).argsort(axis=axis)
    return jnp.take_along_axis(a, idx, axis=axis)


def sample_task(
    rng, images, labels, way, shot,
):
    rng, rng_classes, rng_shuffle = split(rng, 3)
    sampled_classes = random.choice(
        rng_classes, onp.arange(images.shape[0]), (way, 1), replace=False,
    )

    sampled_classes = sampled_classes.repeat(shot, 1)
    sampled_idxs = shuffle_along_axis(
        rng_shuffle, onp.arange(images.shape[1])[None].repeat(way, 0), 1
    )[:, :shot]

    sampled_images = images[sampled_classes, sampled_idxs]
    sampled_labels = labels[sampled_classes, sampled_idxs]

    return sampled_images, sampled_labels


def sample_tasks_with_overlap(rng, images, labels, num_tasks, way, shot):
    rng, rng_classes, rng_shuffle = split(rng, 3)
    sampled_classes = shuffle_along_axis(
        rng_classes, onp.arange(images.shape[0])[None, :].repeat(num_tasks, 0), 1
    )[:, :way]
    sampled_classes = sampled_classes.reshape(num_tasks * way, 1)
    sampled_classes = sampled_classes.repeat(shot, 1)

    sampled_idxs = shuffle_along_axis(
        rng_shuffle, onp.arange(images.shape[1])[None].repeat(num_tasks * way, 0), 1
    )[:, :shot]

    sampled_images = images[sampled_classes, sampled_idxs]
    sampled_labels = labels[sampled_classes, sampled_idxs]

    return sampled_images, sampled_labels


def sample_tasks(rng, images, labels, num_tasks, way, shot, disjoint=True):
    if disjoint:
        sampled_images, sampled_labels = sample_task(
            rng, images, labels, way=num_tasks * way, shot=shot
        )
    else:
        sampled_images, sampled_labels = sample_tasks_with_overlap(
            rng, images, labels, num_tasks=num_tasks, way=way, shot=shot
        )

    sampled_images = sampled_images.reshape(
        num_tasks, way, shot, *sampled_images.shape[2:]
    )
    sampled_labels = sampled_labels.reshape(num_tasks, way, shot)

    return sampled_images, sampled_labels


def prepare_data(data_dir, split="train", preprocessed=None):
    if preprocessed:
        fname = f"mini-imagenet-cache-{split}-preprocessed.pkl"
    else:
        fname = f"mini-imagenet-cache-{split}.pkl"
    with open(osp.join(data_dir, fname), "rb") as f:
        data = pickle.load(f)

    images = data["image_data"]
    num_classes = len(data["class_dict"])
    num_samples = images.shape[0] // num_classes
    labels = onp.tile(onp.arange(num_classes).reshape(num_classes, 1), num_samples)
    images = images.reshape(num_classes, num_samples, *images.shape[1:])

    return images, labels, data["class_dict"]

