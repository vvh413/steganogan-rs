use std::collections::BTreeMap;

use anyhow::Result;
use candle_nn::VarMap;
use lazy_static::lazy_static;

lazy_static! {
  static ref RS_ENC: reed_solomon::Encoder = reed_solomon::Encoder::new(250);
  static ref RS_DEC: reed_solomon::Decoder = reed_solomon::Decoder::new(250);
}

pub fn data_to_bits(data: &[u8]) -> Vec<u8> {
  let compressed =
    miniz_oxide::deflate::compress_to_vec(data, miniz_oxide::deflate::CompressionLevel::DefaultLevel as u8);
  compressed
    .chunks(5)
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

pub fn bytes_to_data(bytes: &[u8]) -> Result<Vec<u8>> {
  let decoded: Vec<_> = bytes
    .chunks(255)
    .map(|chunk| RS_DEC.correct(chunk, None).map(|corrected| corrected.data().to_vec()))
    .collect::<Result<Vec<_>, reed_solomon::DecoderError>>()
    .map_err(|e| anyhow::Error::msg(format!("{e:?}")))?;

  let decoded: Vec<_> = decoded.iter().flatten().copied().collect();
  miniz_oxide::inflate::decompress_to_vec(&decoded).map_err(|e| anyhow::Error::msg(format!("{e:?}")))
}

pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
  bits
    .chunks(8)
    .map(|byte| byte.iter().enumerate().map(|(i, bit)| bit << i).sum())
    .collect()
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
    let bits = data_to_bits(&data);
    assert_eq!(data, bytes_to_data(&bits_to_bytes(&bits))?);
    Ok(())
  }
}
