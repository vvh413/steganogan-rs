use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::ops::leaky_relu;
use candle_nn::{batch_norm, conv2d, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, VarBuilder};

#[derive(Debug)]
pub struct ConvBlock {
  pub conv: Conv2d,
  pub bn: BatchNorm,
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

#[cfg(test)]
mod tests {
  use candle_core::DType;

  use super::*;

  #[test]
  fn test_bn() -> Result<()> {
    let device = &candle_core::Device::cuda_if_available(0)?;
    let bn = BatchNorm::new(
      5,
      Tensor::zeros(5, DType::F32, device)?,
      Tensor::ones(5, DType::F32, device)?,
      Tensor::ones(5, DType::F32, device)?,
      Tensor::zeros(5, DType::F32, device)?,
      1e-5,
    )?;
    println!("{:?}", bn.weight_and_bias());
    println!("{}", bn.running_mean());
    println!("{}", bn.running_var());

    let input: [f32; 120] = [
      -0.749, -1.041, 1.698, -0.658, 1.798, -0.009, 0.281, -0.119, 0.291, -0.597, -0.028, -0.214, -1.313, -1.605,
      -2.203, 0.945, 0.400, 0.083, 1.000, 0.186, 0.500, 0.554, 0.999, -0.254, -0.070, -0.375, -0.110, -0.237, 1.026,
      -2.221, -0.026, 0.607, -1.163, -0.096, -1.972, 1.658, 0.193, -0.369, -0.801, 0.906, 0.480, 0.652, -0.016, -0.668,
      -0.415, 2.065, -0.828, 1.795, -0.206, 0.581, -1.360, 1.619, 1.047, -0.442, 0.420, 0.175, 0.697, 0.262, -0.037,
      -1.495, -0.081, -0.188, 0.027, 0.615, 0.240, -1.144, -2.007, 0.603, -2.664, 0.826, 0.108, -0.169, 1.280, 0.765,
      -0.493, 0.377, 1.131, 0.230, 0.295, -0.263, -0.522, 0.427, 0.634, 1.574, 0.983, -1.250, 0.351, -1.624, -0.812,
      0.763, -0.305, 0.014, -0.403, 0.054, 0.702, 0.841, -1.222, -1.685, -0.071, -0.161, 0.558, -1.586, 0.462, -0.648,
      0.133, 0.042, -0.978, 0.417, 1.231, -1.905, -0.166, 0.126, 0.076, 1.425, -0.911, -0.109, -0.310, -0.673, -1.436,
      0.921,
    ];
    let input = Tensor::new(&input, device)?.reshape((2, 5, 3, 4))?;
    let output = bn.forward_learning(&input)?;
    assert_eq!(output.dims(), &[2, 5, 3, 4]);

    let true_outout: [f32; 120] = [
      -0.639, -0.941, 1.897, -0.544, 2.001, 0.128, 0.429, 0.014, 0.439, -0.482, 0.109, -0.084, -1.681, -2.006, -2.671,
      0.833, 0.226, -0.127, 0.894, -0.012, 0.338, 0.397, 0.893, -0.502, 0.086, -0.232, 0.045, -0.088, 1.231, -2.160,
      0.133, 0.794, -1.055, 0.059, -1.900, 1.891, 0.292, -0.325, -0.799, 1.074, 0.606, 0.796, 0.062, -0.654, -0.375,
      2.346, -0.828, 2.050, -0.201, 0.648, -1.445, 1.767, 1.149, -0.456, 0.474, 0.210, 0.772, 0.303, -0.019, -1.590,
      0.053, -0.057, 0.165, 0.775, 0.386, -1.048, -1.942, 0.762, -2.623, 0.993, 0.250, -0.038, 1.206, 0.633, -0.768,
      0.200, 1.040, 0.037, 0.109, -0.512, -0.801, 0.256, 0.486, 1.532, 1.186, -1.146, 0.526, -1.537, -0.689, 0.957,
      -0.159, 0.175, -0.262, 0.216, 0.893, 1.038, -1.261, -1.769, 0.001, -0.097, 0.692, -1.661, 0.587, -0.631, 0.226,
      0.126, -0.994, 0.538, 1.348, -2.032, -0.157, 0.157, 0.103, 1.557, -0.961, -0.097, -0.313, -0.705, -1.526, 1.013,
    ];
    let true_output = Tensor::new(&true_outout, device)?;
    let output = output.flatten_all()?;
    let diff = (output - true_output)?.sqr()?.sum_keepdim(0)?;
    assert_eq!(candle_core::test_utils::to_vec1_round(&diff, 4)?, &[0f32]);

    Ok(())
  }
}
