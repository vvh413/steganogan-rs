use anyhow::Result;
use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};
use model::decoder::Decoder;
use model::encoder::Encoder;

mod model;
#[allow(dead_code)]
mod utils;

fn main() -> Result<()> {
  let device = &Device::cuda_if_available(0)?;

  {
    println!("loading encoder");
    let mut enc_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&enc_varmap, candle_core::DType::F32, device);
    let encoder = Encoder::new(8, 32, vb.clone())?;
    enc_varmap.load("pretrained/encoder.safetensors")?;

    println!("preparing cover image");
    let img = image::open("/media/c/pic/coffee.png")?;
    let img_bytes = img.to_rgb8().into_raw();
    let img_tensor =
      candle_core::Tensor::from_vec(img_bytes, (img.width() as usize, img.height() as usize, 3), device)?
        .permute((2, 1, 0))?
        .unsqueeze(0)?;
    let img_tensor = ((img_tensor.to_dtype(candle_core::DType::F32)? / 127.5)? - 1.)?;

    println!("preparing data");
    let data_size = (img.height() * img.width() * 8) as usize;
    let mut message = utils::data_to_bits("кофе ".as_bytes());
    message.extend([0; 32]);
    let mut data = message.clone();
    while data.len() < data_size {
      data.extend(message.clone());
    }
    data.truncate(data_size);
    let data = candle_core::Tensor::from_vec(data, (1, 8, img.width() as usize, img.height() as usize), device)?;
    let data = data.to_dtype(candle_core::DType::F32)?;

    println!("encoding");
    let x = encoder.forward(&img_tensor, &data)?;

    println!("saving stegimage");
    let x = ((x.get(0)?.clamp(-1., 1.)?.permute((2, 1, 0))? + 1.)? * 127.5)?;
    let img = image::RgbImage::from_raw(
      img.width(),
      img.height(),
      x.flatten_all()?.to_dtype(candle_core::DType::U8)?.to_vec1::<u8>()?,
    )
    .unwrap();
    img.save("test.png")?;
  }
  {
    println!("loading decoder");
    let mut dec_varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&dec_varmap, candle_core::DType::F32, device);
    let decoder = Decoder::new(8, 32, vb.clone())?;
    dec_varmap.load("pretrained/decoder.safetensors")?;

    println!("loading stegimage");
    let img = image::open("test.png")?;
    let img_bytes = img.to_rgb8().into_raw();
    let img_tensor =
      candle_core::Tensor::from_vec(img_bytes, (img.width() as usize, img.height() as usize, 3), device)?
        .permute((2, 1, 0))?
        .unsqueeze(0)?;
    let img_tensor = (img_tensor.to_dtype(candle_core::DType::F32)? / 255.)?;

    println!("decoding");
    let data = decoder
      .forward(&img_tensor)?
      .flatten_all()?
      .gt(0.)?
      .to_dtype(candle_core::DType::U8)?
      .to_vec1::<u8>()?;

    println!("decoding data");
    let data = utils::bits_to_bytes(&data);
    let idxs: Vec<usize> = data
      .windows(4)
      .enumerate()
      .filter(|(_, window)| *window == [0; 4])
      .map(|(idx, _)| idx)
      .collect();
    let mut parts = Vec::new();
    let mut cur = data.as_slice();
    for idx in idxs.iter().rev() {
      if idx + 4 > cur.len() {
        continue;
      }
      let (other, part) = cur.split_at(idx + 4);
      parts.push(part);
      cur = other.split_at(*idx).0;
    }
    for part in parts.iter() {
      match std::panic::catch_unwind(|| utils::bytes_to_data(part).unwrap()) {
        Ok(data) => println!("{}", String::from_utf8(data)?),
        Err(_) => continue,
      }
    }
  }

  Ok(())
}
