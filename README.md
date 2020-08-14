# oaml-jax
Online Aware Meta Learning on JAX

JAX implementation of [Meta-Learning Representations for Continual Learning](https://arxiv.org/abs/1905.12588)

Using image size 28 and only the last fully connected layer as the PLN, prefetching all the omniglot dataset to GPU and training for 20k (~25 mins on GTX 1080Ti) it reaches similar performance to the one reported in [ANML: Learning to Continually Learn](https://arxiv.org/abs/2002.09571).
