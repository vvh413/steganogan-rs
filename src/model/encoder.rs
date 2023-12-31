use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

use super::conv_block::ConvBlock;

pub struct Encoder {
  initial: ConvBlock,
  convs: Vec<ConvBlock>,
  out: Conv2d,
  add_image: bool,
}

impl Encoder {
  pub fn new(data_depth: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
    let conv_config = Conv2dConfig {
      padding: 1,
      ..Default::default()
    };
    Ok(Self {
      initial: ConvBlock::new(3, hidden_size, vb.pp("conv1"))?,
      convs: vec![
        ConvBlock::new(hidden_size + data_depth, hidden_size, vb.pp("conv2"))?,
        ConvBlock::new(2 * hidden_size + data_depth, hidden_size, vb.pp("conv3"))?,
      ],
      out: conv2d(3 * hidden_size + data_depth, 3, 3, conv_config, vb.pp("conv4.0"))?,
      add_image: true,
    })
  }

  pub fn forward(&self, image: &Tensor, data: &Tensor) -> candle_core::Result<Tensor> {
    let mut x = self.initial.forward(image)?;
    let mut xc = x;
    for layer in self.convs.iter() {
      x = layer.forward(&Tensor::cat(&[&xc, data], 1)?)?;
      xc = Tensor::cat(&[&xc, &x], 1)?;
    }
    x = self.out.forward(&Tensor::cat(&[&xc, data], 1)?)?;
    if self.add_image {
      x = image.add(&x)?
    }
    Ok(x)
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
    let _ = Encoder::new(4, 32, vb)?;
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
  weight: [32, 36, 3, 3]
 2
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
conv3
 0
  bias: [32]
  weight: [32, 68, 3, 3]
 2
  bias: [32]
  running_mean: [32]
  running_var: [32]
  weight: [32]
conv4
 0
  bias: [3]
  weight: [3, 100, 3, 3]"
    );

    Ok(())
  }

  #[test]
  fn test_out_shape() -> Result<()> {
    let varmap = VarMap::new();
    let device = &candle_core::Device::cuda_if_available(0)?;
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let encoder = Encoder::new(4, 32, vb)?;
    let image = Tensor::randn(0f32, 1f32, (16, 3, 127, 127), device)?;
    let data = Tensor::randn(0f32, 1f32, (16, 4, 127, 127), device)?;
    let out = encoder.forward(&image, &data)?;
    assert_eq!(out.shape(), image.shape());
    Ok(())
  }

  #[test]
  fn test_load() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let encoder = Encoder::new(8, 32, vb.clone())?;
    varmap.load("pretrained/encoder.safetensors")?;
    let conv4bias = [-0.0060789185, 0.03809108, -0.018308492];
    assert_eq!(vb.get((3,), "conv4.0.bias")?.to_vec1::<f32>()?, conv4bias);
    assert_eq!(encoder.out.bias().unwrap().to_vec1::<f32>()?, conv4bias);
    Ok(())
  }

  #[test]
  fn test_forward() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let encoder = Encoder::new(8, 32, vb.clone())?;
    varmap.load("pretrained/encoder.safetensors")?;
    let image = (Tensor::ones((1, 3, 127, 127), candle_core::DType::F32, device)? * 0.2)?;
    let data = (Tensor::ones((1, 8, 127, 127), candle_core::DType::F32, device)? * 0.3)?;
    let out = encoder.forward(&image, &data)?.mean_all()?;
    assert_eq!(candle_core::test_utils::to_vec0_round(&out, 3)?, 0.201);
    Ok(())
  }
}
