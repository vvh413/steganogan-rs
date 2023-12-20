use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::ops::leaky_relu;
use candle_nn::{batch_norm, conv2d, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, VarBuilder};

#[derive(Debug)]
pub struct ConvBlock {
  conv: Conv2d,
  bn: BatchNorm,
}

impl ConvBlock {
  pub fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
    let conv_config = Conv2dConfig {
      padding: 1,
      ..Default::default()
    };
    let bn_config = BatchNormConfig::default();
    Ok(Self {
      conv: conv2d(in_channels, out_channels, 3, conv_config, vb.pp("0"))?,
      bn: batch_norm(out_channels, bn_config, vb.pp("2"))?,
    })
  }
}

impl Module for ConvBlock {
  fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    let x = self.conv.forward(x)?;
    let x = leaky_relu(&x, 0.01)?;
    self.bn.forward(&x)
  }
}
