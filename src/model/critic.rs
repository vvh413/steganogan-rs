use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{batch_norm, conv2d, seq, Activation, BatchNormConfig, Conv2dConfig, Sequential, VarBuilder};

pub struct Critic {
  layers: Sequential,
}

impl Critic {
  pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
    let conv_config = Conv2dConfig {
      padding: 1,
      ..Default::default()
    };
    let bn_config = BatchNormConfig {
      eps: 1e-5,
      remove_mean: true,
      affine: true,
    };
    let vb = vb.pp("layers");
    Ok(Self {
      layers: seq()
        .add(conv2d(3, hidden_size, 3, conv_config, vb.pp("0"))?)
        .add(Activation::LeakyRelu(0.01))
        .add(batch_norm(hidden_size, bn_config, vb.pp("2"))?)
        .add(conv2d(hidden_size, hidden_size, 3, conv_config, vb.pp("3"))?)
        .add(Activation::LeakyRelu(0.01))
        .add(batch_norm(hidden_size, bn_config, vb.pp("5"))?)
        .add(conv2d(hidden_size, hidden_size, 3, conv_config, vb.pp("6"))?)
        .add(Activation::LeakyRelu(0.01))
        .add(batch_norm(hidden_size, bn_config, vb.pp("8"))?)
        .add(conv2d(hidden_size, 1, 3, conv_config, vb.pp("9"))?)
        .add_fn(|x| x.mean((1, 2, 3))),
    })
  }
}

impl Critic {
  fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    self.layers.forward(x)
  }
}

#[cfg(test)]
mod tests {
  use candle_nn::VarMap;

  use super::*;

  #[test]
  fn test_to_string() -> Result<()> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(
      &varmap,
      candle_core::DType::F32,
      &candle_core::Device::cuda_if_available(0)?,
    );
    let _ = Critic::new(32, vb)?;
    assert_eq!(
      crate::utils::varmap_to_string(&varmap),
      r"layers
 0
  bias: [32]
  weight: [32, 3, 3, 3]
 2
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
 3
  bias: [32]
  weight: [32, 32, 3, 3]
 5
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
 6
  bias: [32]
  weight: [32, 32, 3, 3]
 8
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
 9
  bias: [1]
  weight: [1, 32, 3, 3]"
    );

    Ok(())
  }

  #[test]
  fn test_out_shape() -> Result<()> {
    let varmap = VarMap::new();
    let device = &candle_core::Device::cuda_if_available(0)?;
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let encoder = Critic::new(32, vb)?;
    let x = Tensor::randn(0f32, 1f32, (16, 3, 127, 127), device)?;
    let out = encoder.forward(&x)?;
    assert_eq!(out.shape().dims(), [16]);
    Ok(())
  }

  #[test]
  fn test_load() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let _ = Critic::new(32, vb.clone());
    varmap.load("pretrained/critic.safetensors")?;
    assert_eq!(vb.get((1,), "layers.9.bias")?.to_vec1::<f32>()?, vec![-0.03937991]);
    Ok(())
  }

  #[test]
  fn test_forward() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let critic = Critic::new(32, vb)?;
    varmap.load("pretrained/critic.safetensors")?;
    let out = critic
      .forward(&Tensor::ones((1, 3, 127, 127), candle_core::DType::F32, device)?)?
      .mean_all()?;
    assert_eq!(candle_core::test_utils::to_vec0_round(&out, 4)?, 1.2609);
    Ok(())
  }
}
