import numpy as onp
from PIL import Image
from tqdm import tqdm

from jax.random import split
from jax.image import resize as im_resize
from jax import vmap, numpy as jnp, random
from jax.tree_util import Partial as partial

import tensorflow_datasets as tfds

mean28, std28 = (0.9220594763755798, 0.2477845698595047)
mean84, std84 = (0.9220604300498962, 0.2642029821872711)

# def prepare_omniglot_data(split="train", resize=(28, 28), normalize=True):

#     data, = tfds.as_numpy(
#         tfds.load(
#             name="omniglot", shuffle_files=False, batch_size=-1, split=[split],
#     ))

#     images = jnp.array((data["image"][:, :, :, [0]])) / 255
#     if resize:
#          images = vmap(lambda img: im_resize(img, (*resize, 1), "lanczos5"))(images)
#     if normalize:
#         if normalize is True:
#             mean, std = images.mean(), images.std()
#         else:
#             mean, std = normalize
#         images = (images - mean) / std
#     labels = jnp.array(data["label"])
#     order = jnp.argsort(labels)
#     uniques, counts = jnp.unique(labels, return_counts=True)
#     assert (counts[0] == counts).all()

#     images = images[order]
#     images = images.reshape(len(uniques), counts[0], *images.shape[1:])
#     labels = labels[order].reshape(len(uniques), counts[0], *labels.shape[1:])

#     return images, labels


def prepare_omniglot_data(split="train", resize=(28, 28), normalize=False, convert="L"):
    (data,) = tfds.as_numpy(
        tfds.load(name="omniglot", shuffle_files=False, batch_size=-1, split=[split],)
    )

    if resize or convert:
        to_pil = lambda x: Image.fromarray(x)
        if convert:
            _convert = lambda x: to_pil(x).convert(convert)
        else:
            _convert = lambda x: to_pil(x)
        if resize:
            _resize = lambda x: _convert(x).resize(resize, Image.LANCZOS)
        else:
            _resize = lambda x: _convert(x)
        transform = _resize
        images = onp.stack(
            [transform(img) for img in tqdm(data["image"], ncols=0)]
        ).astype(onp.float32)
    else:
        images = data["image"]

    images = images / 255
    labels = data["label"]
    order = onp.argsort(labels)
    uniques, counts = onp.unique(labels, return_counts=True)
    assert (counts[0] == counts).all()

    images = images[order]
    images = images.reshape(len(uniques), counts[0], *images.shape[1:])
    labels = labels[order].reshape(len(uniques), counts[0], *labels.shape[1:])

    return images, labels


def prepare_omniglot_data28(split, normalize=True):
    if normalize:
        normalize = (mean28, std28)
    return prepare_omniglot_data(split, resize=(28, 28), normalize=normalize)


def prepare_omniglot_data84(split, normalize=True):
    if normalize:
        normalize = (mean84, std84)
    return prepare_omniglot_data(split, resize=(84, 84), normalize=normalize)


def random_samples_choice(rng, x, y, num_samples):
    idxs = random.choice(rng, jnp.arange(x.shape[0]), (num_samples,), replace=False)
    return x[idxs], y[idxs]


def sample_tasks_and_samples(
    rng, images, labels, tasks, num_tasks, num_samples, shuffle=True
):
    rng_tasks, rng_shuffle = split(rng)
    sampled_tasks = random.choice(rng_tasks, tasks, (num_tasks,), replace=False)
    if shuffle:
        sampled_images, labels = vmap(
            partial(random_samples_choice, num_samples=num_samples)
        )(split(rng_shuffle, num_tasks), images[sampled_tasks], labels[sampled_tasks],)
    else:
        sampled_images = images[sampled_tasks, :num_samples]
        labels = labels[sampled_tasks, :num_samples]
    return sampled_images, labels, sampled_tasks


def random_samples(rng, images, labels, num_samples):
    _, rng = split(rng, 2)
    sampled_idxs = random.choice(
        rng, jnp.arange(images.shape[0]), (num_samples,), replace=False
    )
    return images[sampled_idxs], labels[sampled_idxs]
