use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};
use clap::{Args, Parser, Subcommand};
use model::decoder::Decoder;
use model::encoder::Encoder;

mod model;
#[allow(dead_code)]
mod utils;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
  #[command(subcommand)]
  command: Command,
}

#[derive(Subcommand)]
enum Command {
  Encode(EncodeArgs),
  Decode(DecodeArgs),
}

#[derive(Args)]
struct EncodeArgs {
  #[arg(short)]
  input: PathBuf,
  #[arg(short)]
  output: PathBuf,
  #[arg(short)]
  data: String,
}

#[derive(Args)]
struct DecodeArgs {
  #[arg(short)]
  input: PathBuf,
}

fn encode(args: EncodeArgs) -> Result<()> {
  let device = &Device::cuda_if_available(0)?;

  let mut enc_varmap = VarMap::new();
  let vb = VarBuilder::from_varmap(&enc_varmap, candle_core::DType::F32, device);
  let encoder = Encoder::new(8, 32, vb.clone())?;
  enc_varmap.load("pretrained/encoder.safetensors")?;

  let img = image::open(args.input)?;
  let img_bytes = img.to_rgb8().into_raw();
  let img_tensor = candle_core::Tensor::from_vec(img_bytes, (img.width() as usize, img.height() as usize, 3), device)?
    .permute((2, 1, 0))?
    .unsqueeze(0)?;
  let img_tensor = ((img_tensor.to_dtype(candle_core::DType::F32)? / 127.5)? - 1.)?;

  let data_size = (img.height() * img.width() * 8) as usize;
  let mut message = utils::bytes_to_encoded_bits(args.data.as_bytes());
  message.extend([0; 32]);
  let mut data = message.clone();
  while data.len() < data_size {
    data.extend(message.clone());
  }
  data.truncate(data_size);
  let data = candle_core::Tensor::from_vec(data, (1, 8, img.height() as usize, img.width() as usize), device)?;
  let data = data.to_dtype(candle_core::DType::F32)?;

  let x = encoder.forward(&img_tensor, &data)?;

  let x = ((x.get(0)?.clamp(-1., 1.)?.permute((2, 1, 0))? + 1.)? * 127.5)?;
  let img = image::RgbImage::from_raw(
    img.width(),
    img.height(),
    x.flatten_all()?.to_dtype(candle_core::DType::U8)?.to_vec1::<u8>()?,
  )
  .unwrap();

  img.save(args.output)?;

  println!("done");
  Ok(())
}

fn map_inc(map: &mut HashMap<String, usize>, k: String) {
  *map.entry(k).or_default() += 1;
}

fn decode(args: DecodeArgs) -> Result<()> {
  let device = &Device::cuda_if_available(0)?;

  let mut dec_varmap = VarMap::new();
  let vb = VarBuilder::from_varmap(&dec_varmap, candle_core::DType::F32, device);
  let decoder = Decoder::new(8, 32, vb.clone())?;
  dec_varmap.load("pretrained/decoder.safetensors")?;

  let img = image::open(args.input)?;
  let img_bytes = img.to_rgb8().into_raw();
  let img_tensor = candle_core::Tensor::from_vec(img_bytes, (img.width() as usize, img.height() as usize, 3), device)?
    .permute((2, 1, 0))?
    .unsqueeze(0)?;
  let img_tensor = (img_tensor.to_dtype(candle_core::DType::F32)? / 255.)?;

  let data = decoder
    .forward(&img_tensor)?
    .flatten_all()?
    .gt(0.)?
    .to_dtype(candle_core::DType::U8)?
    .to_vec1::<u8>()?;

  let data = utils::bits_to_bytes(&data);
  let parts = utils::split_bytes(data.as_slice(), &[0; 4]);
  let mut results: HashMap<String, usize> = HashMap::new();
  for part in parts.iter() {
    match utils::encoded_bytes_to_data(part).and_then(|part| Ok(String::from_utf8(part)?)) {
      Ok(result) => {
        let result = result.replace('\0', "");
        if !result.is_empty() {
          map_inc(&mut results, result)
        }
      }
      Err(_) => continue,
    }
  }
  match results.iter().max_by_key(|(_, v)| *v).map(|(k, _)| k) {
    Some(result) => println!("{result}"),
    None => println!("No data found"),
  }

  Ok(())
}

fn main() -> Result<()> {
  let args = Cli::parse();
  match args.command {
    Command::Encode(args) => encode(args),
    Command::Decode(args) => decode(args),
  }
}
