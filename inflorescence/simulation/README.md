# Flower simulation module

This module is copied from the flwr Python package with one modification to
change `@ray.remote()` to `@ray.remote(max_calls=1)` to prevent Ray from
consuming all CUDA memory with idle processes. For more information, see:
https://github.com/adap/flower/pull/1384