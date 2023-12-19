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
}

impl Encoder {
  pub fn forward(&self, image: &Tensor, data: &Tensor) -> candle_core::Result<Tensor> {
    let mut x = self.initial.forward(image)?;
    let mut xc = Tensor::cat(&[data, &x], 1)?;
    for layer in self.convs.iter() {
      x = layer.forward(&xc)?;
      xc = Tensor::cat(&[&xc, &x], 1)?;
    }
    x = self.out.forward(&xc)?;
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
    let conv4bias = vec![-0.0060789185, 0.03809108, -0.018308492];
    assert_eq!(vb.get((3,), "conv4.0.bias")?.to_vec1::<f32>()?, conv4bias);
    assert_eq!(encoder.out.bias().unwrap().to_vec1::<f32>()?, conv4bias);
    Ok(())
  }

  #[test]
  fn test_forward() -> Result<()> {
    let device = &candle_core::Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let encoder = Encoder::new(8, 32, vb)?;
    varmap.load("pretrained/encoder.safetensors")?;
    let image = Tensor::new(&[0.2f32], device)?.broadcast_as((1, 3, 127, 127))?;
    let data = Tensor::new(&[0.3f32], device)?.broadcast_as((1, 8, 127, 127))?;
    let out_initial_conv = encoder.initial.conv.forward(&image)?.mean_all()?;
    let out_initial = encoder.initial.forward(&image)?.mean_all()?;
    let out = encoder.forward(&image, &data)?.mean_all()?;
    assert_eq!(candle_core::test_utils::to_vec0_round(&out_initial_conv, 4)?, 0.1312);
    assert_eq!(candle_core::test_utils::to_vec0_round(&out_initial, 4)?, -0.1724);
    assert_eq!(candle_core::test_utils::to_vec0_round(&out, 4)?, 0.2006);
    Ok(())
  }
}
