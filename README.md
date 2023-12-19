# steganogan-rs

Trying to port [SteganoGAN](https://github.com/DAI-Lab/SteganoGAN/) to Rust using [candle](https://github.com/huggingface/candle) framework.

## Progress

Implemented `dense` variant of SteganoGAN architecture, so it is possible to reexport weights to safetensors and load them here. Already reexported weights are in [safetensors/](safetensors/) dir.

Now there are some troubles with batch-norm layer:
  1. Not working as expected. Same weights, same hyperparams (or not?), same running mean and var. But different output.
  2. For now they aren't trainable, so it isn't possible to just train model from start. Maybe later I'll just try replacing batch-norm with some other norm-layer and train it from start.
