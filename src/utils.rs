use std::collections::BTreeMap;

use anyhow::Result;
use candle_nn::VarMap;
use lazy_static::lazy_static;

const CHUNK_SIZE: usize = 5;
const ENCODED_SIZE: usize = 30;
lazy_static! {
  static ref RS_ENC: reed_solomon::Encoder = reed_solomon::Encoder::new(ENCODED_SIZE - CHUNK_SIZE);
  static ref RS_DEC: reed_solomon::Decoder = reed_solomon::Decoder::new(ENCODED_SIZE - CHUNK_SIZE);
}

pub fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
  data
    .iter()
    .flat_map(|byte| {
      let mut byte = *byte;
      let mut bits = Vec::new();
      for _ in 0..8 {
        bits.push(byte & 1);
        byte >>= 1;
      }
      bits
    })
    .collect()
}

pub fn bytes_to_encoded_bits(data: &[u8]) -> Vec<u8> {
  let compressed =
    miniz_oxide::deflate::compress_to_vec(data, miniz_oxide::deflate::CompressionLevel::DefaultLevel as u8);
  compressed
    .chunks(CHUNK_SIZE)
    .flat_map(|chunk| RS_ENC.encode(chunk).to_vec())
    .flat_map(|mut byte| {
      let mut bits = Vec::new();
      for _ in 0..8 {
        bits.push(byte & 1);
        byte >>= 1;
      }
      bits
    })
    .collect()
}

pub fn encoded_bytes_to_data(bytes: &[u8]) -> Result<Vec<u8>> {
  let mut decoded = Vec::with_capacity(bytes.len() / ENCODED_SIZE * CHUNK_SIZE);
  for chunk in bytes.chunks(ENCODED_SIZE) {
    let decoded_chunk: Vec<u8> = match RS_DEC.correct(chunk, None) {
      Ok(decoded_chunk) => decoded_chunk.iter().take(CHUNK_SIZE).copied().collect(),
      Err(_) => chunk.iter().take(CHUNK_SIZE).copied().collect(),
    };
    decoded.extend(decoded_chunk);
  }

  match miniz_oxide::inflate::decompress_to_vec(&decoded) {
    Ok(decompressed) => Ok(decompressed),
    Err(err) => Ok(err.output),
  }
}

pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
  bits
    .chunks(8)
    .map(|byte| byte.iter().enumerate().map(|(i, bit)| bit << i).sum())
    .collect()
}

pub fn split_bytes<'a>(bytes: &'a [u8], delimeter: &[u8]) -> Vec<&'a [u8]> {
  let idxs: Vec<usize> = bytes
    .windows(4)
    .enumerate()
    .filter(|(_, window)| *window == delimeter)
    .map(|(idx, _)| idx)
    .collect();
  let mut parts = Vec::new();
  let mut cur = bytes;
  for idx in idxs.iter().rev() {
    if idx + 4 > cur.len() {
      continue;
    }
    let (other, part) = cur.split_at(idx + 4);
    parts.push(part);
    cur = other.split_at(*idx).0;
  }
  parts
}

#[derive(Debug)]
enum TreeNode {
  Leaf(String),
  Branch(BTreeMap<String, TreeNode>),
}

impl TreeNode {
  fn as_branch_mut(&mut self) -> Option<&mut BTreeMap<String, TreeNode>> {
    if let TreeNode::Branch(ref mut map) = self {
      Some(map)
    } else {
      None
    }
  }
}

fn tree_to_string(tree: &BTreeMap<String, TreeNode>, indent: usize, s: &mut String) {
  for (key, value) in tree.iter() {
    match value {
      TreeNode::Branch(branch) => {
        *s = format!("{s}\n{:indent$}{}", "", key);
        tree_to_string(branch, indent + 1, s);
      }
      TreeNode::Leaf(var) => {
        *s = format!("{s}: {var}");
      }
    }
  }
}

pub fn varmap_to_string(varmap: &VarMap) -> String {
  let varmap = varmap.data().lock().unwrap();
  let mut vars: Vec<_> = varmap.iter().collect();
  vars.sort_by_key(|(prefix, _)| *prefix);

  let mut tree: BTreeMap<String, TreeNode> = BTreeMap::new();

  for (path, var) in vars.iter() {
    let path_parts: Vec<&str> = path.split('.').collect();
    let mut current = &mut tree;
    for part in path_parts.iter() {
      current = current
        .entry((*part).to_string())
        .or_insert(TreeNode::Branch(BTreeMap::new()))
        .as_branch_mut()
        .unwrap();
    }
    current.insert(
      path_parts.last().unwrap().to_string(),
      TreeNode::Leaf(format!("{:?}", var.shape())),
    );
  }

  let mut s = String::new();
  tree_to_string(&tree, 0, &mut s);
  s.chars().skip(1).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test() -> Result<()> {
    let data = vec![1, 2, 3, 4, 5, 6];
    let bits = bytes_to_encoded_bits(&data);
    assert_eq!(data, encoded_bytes_to_data(&bits_to_bytes(&bits))?);
    Ok(())
  }
}
