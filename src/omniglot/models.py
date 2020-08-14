import jax
from jax.random import split
from jax.experimental import stax
from jax import numpy as jnp, nn, random


def DenseNoBias(out_dim, W_init=nn.initializers.glorot_normal()):
    """Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        (k1,) = random.split(rng, 1)
        W = W_init(k1, (input_shape[-1], out_dim))
        return output_shape, (W,)

    def apply_fun(params, inputs, **kwargs):
        (W,) = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def Reshape(newshape):
    """Layer construction function for a reshape layer."""
    init_fun = lambda rng, input_shape: (newshape, ())
    apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, newshape)
    return init_fun, apply_fun


def make_conv(
    strides=None, num_channels=256,
):
    return stax.Conv(
        out_chan=num_channels,
        filter_shape=(3, 3),
        padding="VALID",
        strides=strides,
        W_init=nn.initializers.he_normal(),
        b_init=nn.initializers.zeros,
    )


def make_oml_net(size, num_fc_layers=2, bias=False, num_channels=256):
    strides = [(1, 1) for _ in range(6)]
    if size == 28:
        strides[3] = (2, 2)
        strides[5] = (2, 2)
    elif size == 84:
        strides[0] = (2, 2)
        strides[2] = (2, 2)
        strides[4] = (2, 2)
        strides[5] = (2, 2)

    layers = [
        make_conv(strides[0], num_channels),
        stax.Relu,
        make_conv(strides[1], num_channels),
        stax.Relu,
        make_conv(strides[2], num_channels),
        stax.Relu,
        make_conv(strides[3], num_channels),
        stax.Relu,
        make_conv(strides[4], num_channels),
        stax.Relu,
        make_conv(strides[5], num_channels),
        stax.Relu,
        Reshape((-1, 9 * num_channels)),
    ]

    for _ in range(num_fc_layers - 1):
        layers += [
            stax.Dense(
                1024, W_init=nn.initializers.he_normal(), b_init=nn.initializers.zeros,
            ),
            stax.Relu,
        ]
    layers += [
        DenseNoBias(1000, W_init=nn.initializers.he_normal(),),
        stax.LogSoftmax,
    ]
    net_init, net_apply = stax.serial(*layers)

    return net_init, net_apply
