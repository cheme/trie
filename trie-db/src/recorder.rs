// Copyright 2017, 2021 Parity Technologies
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Trie query recorder.

use crate::{
	rstd::vec::Vec, CError, CacheNode, CachedValue, Context, NodeCodec, NodeOwned, OwnedNode,
	RecordedForKey, Result, TrieAccess, TrieError, TrieHash, TrieLayout, TrieRecorder,
};
use hashbrown::HashMap;

/// Records trie nodes as they pass it.
#[cfg_attr(feature = "std", derive(Debug))]
pub struct Recorder<L: TrieLayout> {
	nodes: Vec<(TrieHash<L>, Vec<u8>)>,
	recorded_keys: HashMap<Vec<u8>, RecordedForKey>,
	current: CacheNode<L>,
}

impl<L: TrieLayout> Default for Recorder<L> {
	fn default() -> Self {
		Recorder::new()
	}
}

impl<L: TrieLayout> Recorder<L> {
	/// Create a new `Recorder` which records all given nodes.
	pub fn new() -> Self {
		Self {
			nodes: Default::default(),
			recorded_keys: Default::default(),
			current: CacheNode::new(),
		}
	}

	/// Drain all visited records.
	pub fn drain(&mut self) -> Vec<(TrieHash<L>, Vec<u8>)> {
		self.recorded_keys.clear();
		crate::rstd::mem::take(&mut self.nodes)
	}
}

impl<L: TrieLayout> TrieRecorder<TrieHash<L>> for Recorder<L> {
	fn record<'a>(&mut self, access: TrieAccess<'a, TrieHash<L>>) {
		match access {
			TrieAccess::EncodedNode { hash, encoded_node, .. } => {
				self.nodes.push((hash, encoded_node.to_vec()));
			},
			TrieAccess::NodeOwned { hash, node_owned, .. } => {
				self.nodes.push((hash, node_owned.to_encoded::<L::Codec>()));
			},
			TrieAccess::Value { hash, value, full_key } => {
				self.nodes.push((hash, value.to_vec()));
				self.recorded_keys.entry(full_key.to_vec()).insert(RecordedForKey::Value);
			},
			TrieAccess::Hash { full_key } => {
				self.recorded_keys.entry(full_key.to_vec()).or_insert(RecordedForKey::Hash);
			},
			TrieAccess::NonExisting { full_key } => {
				// We handle the non existing value/hash like having recorded the value.
				self.recorded_keys.entry(full_key.to_vec()).insert(RecordedForKey::Value);
			},
		}
	}

	fn trie_nodes_recorded_for_key(&self, key: &[u8]) -> RecordedForKey {
		self.recorded_keys.get(key).copied().unwrap_or(RecordedForKey::None)
	}
}

impl<L: TrieLayout> Context<L> for Recorder<L> {
	fn restart_record_node(&mut self) {
		self.recorded_keys.clear();
	}

	fn lookup_value_for_key(&mut self, _key: &[u8]) -> Option<&CachedValue<TrieHash<L>>> {
		None
	}

	fn cache_value_for_key(&mut self, key: &[u8], value: CachedValue<TrieHash<L>>) {
		// TODO recorded_keys is useless here: we don't cache value!! -> simply rem field.
		match &value {
			CachedValue::NonExisting => {
				self.recorded_keys.entry(key.to_vec()).insert(RecordedForKey::Value);
			},
			CachedValue::ExistingHash(_h) => {
				self.recorded_keys.entry(key.to_vec()).or_insert(RecordedForKey::Hash);
			},
			CachedValue::Existing { .. } => {
				self.recorded_keys.entry(key.to_vec()).insert(RecordedForKey::Value);
			},
		}
	}

	fn get_or_insert_node(
		&mut self,
		hash: TrieHash<L>,
		is_value: bool,
		fetch_node: &mut dyn FnMut() -> Result<Vec<u8>, TrieHash<L>, CError<L>>,
	) -> Result<&NodeOwned<TrieHash<L>>, TrieHash<L>, CError<L>> {
		let node_data = (*fetch_node)()?;
		if is_value {
			let node = NodeOwned::Value(node_data.clone().into(), hash);
			self.nodes.push((hash, node_data));
			self.current = CacheNode { encoded: None, owned: Some(node) };
			Ok(self.current.node_owned())
		} else {
			let node = match L::Codec::decode(&node_data[..]) {
				Ok(node) => node,
				Err(e) => return Err(Box::new(TrieError::DecoderError(hash, e))),
			};
			let node = node.to_owned_node::<L>()?;

			self.nodes.push((hash, node_data));
			self.current = CacheNode { encoded: None, owned: Some(node) };
			Ok(self.current.node_owned())
		}
	}

	fn get_or_insert_node2(
		&mut self,
		hash: TrieHash<L>,
		is_value: bool,
		fetch_node: &mut dyn FnMut() -> Result<Vec<u8>, TrieHash<L>, CError<L>>,
	) -> Result<&OwnedNode<Vec<u8>>, TrieHash<L>, CError<L>> {
		let node_data = (*fetch_node)()?;
		if is_value {
			unreachable!("TODO never use this for value, just use the NodeOwned variant");
		} else {
			let node = match OwnedNode::new::<L::Codec>(node_data.clone()) {
				Ok(node) => node,
				Err(e) => return Err(Box::new(TrieError::DecoderError(hash, e))),
			};

			self.nodes.push((hash, node_data));
			self.current = CacheNode { owned: None, encoded: Some(node) };
			Ok(self.current.owned_node())
		}
	}

	// TODO should be removable (get_or_insert as single entry point).
	fn get_node(&mut self, _hash: &TrieHash<L>) -> Option<&NodeOwned<TrieHash<L>>> {
		None
	}

	fn record<'a>(&mut self, access: TrieAccess<'a, TrieHash<L>>) {
		match access {
			TrieAccess::EncodedNode { hash, encoded_node, .. } => {
				self.nodes.push((hash, encoded_node.to_vec()));
			},
			TrieAccess::NodeOwned { hash, node_owned, .. } => {
				self.nodes.push((hash, node_owned.to_encoded::<L::Codec>()));
			},
			TrieAccess::Value { hash, value, full_key } => {
				self.nodes.push((hash, value.to_vec()));
				self.recorded_keys.entry(full_key.to_vec()).insert(RecordedForKey::Value);
			},
			TrieAccess::Hash { full_key } => {
				self.recorded_keys.entry(full_key.to_vec()).or_insert(RecordedForKey::Hash);
			},
			TrieAccess::NonExisting { full_key } => {
				// We handle the non existing value/hash like having recorded the value.
				self.recorded_keys.entry(full_key.to_vec()).insert(RecordedForKey::Value);
			},
		}
	}

	fn trie_nodes_recorded_for_key(&self, key: &[u8]) -> RecordedForKey {
		self.recorded_keys.get(key).copied().unwrap_or(RecordedForKey::None)
	}
}
