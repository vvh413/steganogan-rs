use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

use super::conv_block::ConvBlock;

pub struct Decoder {
  initial: ConvBlock,
  convs: Vec<ConvBlock>,
  out: Conv2d,
}

impl Decoder {
  pub fn new(data_depth: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
    let conv_config = Conv2dConfig {
      padding: 1,
      ..Default::default()
    };
    Ok(Self {
      initial: ConvBlock::new(3, hidden_size, vb.pp("conv1"))?,
      convs: vec![
        ConvBlock::new(hidden_size, hidden_size, vb.pp("conv2"))?,
        ConvBlock::new(2 * hidden_size, hidden_size, vb.pp("conv3"))?,
      ],
      out: conv2d(3 * hidden_size, data_depth, 3, conv_config, vb.pp("conv4.0"))?,
    })
  }

  pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
    let mut x = self.initial.forward(x)?;
    let mut xc = Tensor::cat(&[&x], 1)?;
    for layer in self.convs.iter() {
      x = layer.forward(&xc)?;
      xc = Tensor::cat(&[&xc, &x], 1)?;
    }
    self.out.forward(&xc)
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
    let _ = Decoder::new(4, 32, vb)?;
    assert_eq!(
      crate::utils::varmap_to_string(&varmap),
      r"conv1
 0
  bias: [32]
  weight: [32, 3, 3, 3]
 2
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
conv2
 0
  bias: [32]
  weight: [32, 32, 3, 3]
 2
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
conv3
 0
  bias: [32]
  weight: [32, 64, 3, 3]
 2
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
conv4
 0
  bias: [4]
  weight: [4, 96, 3, 3]"
    );

    Ok(())
  }

  #[test]
  fn test_out_shape() -> Result<()> {
    let varmap = VarMap::new();
    let device = &candle_core::Device::cuda_if_available(0)?;
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let decoder = Decoder::new(4, 32, vb)?;
    let x = Tensor::randn(0f32, 1f32, (16, 3, 127, 127), device)?;
    let out = decoder.forward(&x)?;
    assert_eq!(out.shape().dims(), [16, 4, 127, 127]);
    Ok(())
  }

  #[test]
  fn test_load() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let decoder = Decoder::new(8, 32, vb.clone())?;
    varmap.load("pretrained/decoder.safetensors")?;
    let conv4bias = vec![
      0.020432772,
      0.062261935,
      -0.062574305,
      -0.035477437,
      0.012979559,
      -0.051983964,
      -0.0016545125,
      0.01834147,
    ];
    assert_eq!(vb.get((8,), "conv4.0.bias")?.to_vec1::<f32>()?, conv4bias);
    assert_eq!(decoder.out.bias().unwrap().to_vec1::<f32>()?, conv4bias);
    Ok(())
  }

  #[test]
  fn test_forward() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let decoder = Decoder::new(8, 32, vb)?;
    varmap.load("pretrained/decoder.safetensors")?;
    let out = decoder
      .forward(&Tensor::ones((1, 3, 127, 127), candle_core::DType::F32, device)?)?
      .mean_all()?;
    assert_eq!(candle_core::test_utils::to_vec0_round(&out, 4)?, -0.1408);
    Ok(())
  }
}
