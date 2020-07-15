// Copyright 2017, 2019 Parity Technologies
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

//! Linear backend possibly stored into multiple nodes.

use crate::rstd::marker::PhantomData;
use crate::rstd::btree_map::BTreeMap;
use crate::rstd::cell::RefCell;
use crate::rstd::rc::Rc;
use super::{LinearStorage};
use crate::historied::HistoriedValue;
use derivative::Derivative;
use crate::InitFrom;

/// Node storage metadata
pub trait NodesMeta: Sized {
	fn max_head_len() -> usize;
	/// for imbrincated nodes we can limit
	/// by number of head items instead of
	/// max_head_len.
	fn max_head_items() -> Option<usize>;
	fn max_node_len() -> usize;
	fn max_index_len() -> usize;
	fn storage_prefix() -> &'static [u8];
}

pub trait NodeStorageHandle<V, S, D, M: NodesMeta>: Clone {
	fn get_node(&self, reference_key: &[u8], relative_index: u32) -> Option<Node<V, S, D, M>>;

	/// a default addressing scheme for storage that natively works
	/// as a simple key value storage.
	fn vec_address(reference_key: &[u8], relative_index: u32) -> Vec<u8> {
		let storage_prefix = M::storage_prefix();
		let mut result = Vec::with_capacity(reference_key.len() + storage_prefix.len() + 8);
		result.extend_from_slice(storage_prefix);
		result.extend_from_slice(&(reference_key.len() as u32).to_be_bytes());
		result.extend_from_slice(reference_key);
		result.extend_from_slice(&relative_index.to_be_bytes());
		result
	}

}

pub trait NodeStorageMut<V, S, D, M> {
	fn set_node(&mut self, reference_key: &[u8], relative_index: u32, node: &Node<V, S, D, M>);
	fn remove_node(&mut self, reference_key: &[u8], relative_index: u32);
}

// Note that this should not be use out of test as it clone the whole btree map many times.
impl<V, S, D: Clone, M: NodesMeta> NodeStorageHandle<V, S, D, M> for BTreeMap<Vec<u8>, Node<V, S, D, M>> {
	fn get_node(&self, reference_key: &[u8], relative_index: u32) -> Option<Node<V, S, D, M>> {
		let key = Self::vec_address(reference_key, relative_index);
		self.get(&key).cloned()
	}
}

impl<V, S, D: Clone, M: NodesMeta> NodeStorageMut<V, S, D, M> for BTreeMap<Vec<u8>, Node<V, S, D, M>> {
	fn set_node(&mut self, reference_key: &[u8], relative_index: u32, node: &Node<V, S, D, M>) {
		let key = Self::vec_address(reference_key, relative_index);
		self.insert(key, node.clone());
	}
	fn remove_node(&mut self, reference_key: &[u8], relative_index: u32) {
		let key = Self::vec_address(reference_key, relative_index);
		self.remove(&key);
	}
}

#[derive(Derivative)]
#[derivative(Clone(bound="D: Clone"))]
/// A node is a linear backend and some meta information.
pub struct Node<V, S, D, M> {
	data: D,
	changed: bool,
	reference_len: usize,
	_ph: PhantomData<(V, S, D, M)>,
}

/// We manipulate Node from head to be able to share max length.
pub struct NodeFromHead<V, S, D, M, B>(Head<V, S, D, M, B>, usize);

/// Head is the entry node, it contains fetched nodes and additional
/// information about this backend state.
pub struct Head<V, S, D, M, B> {
	inner: Node<V, S, D, M>,
	fetched: RefCell<Vec<Node<V, S, D, M>>>, // TODO consider smallvec
	old_start_index: u32,
	start_node_state: Option<S>,
	start_node_index: u32,
	end_node_index: u32,
	reference_key: Vec<u8>,
	nodes_storage: B,
}

impl<V, S, D: Clone, M, B> Head<V, S, D, M, B>
	where
		M: NodesMeta,
		B: NodeStorageMut<V, S, D, M>,
{
	pub fn flush(&mut self) {
		for d in self.old_start_index .. self.start_node_index {
			self.nodes_storage.remove_node(&self.reference_key[..], d);
		}
		self.old_start_index = self.start_node_index;
		for (index, mut node) in self.fetched.borrow_mut().iter_mut().enumerate() {
			if node.changed {
				self.nodes_storage.set_node(&self.reference_key[..], index as u32 + self.start_node_index, node);
				node.changed = false;
			}
		}
	}
}

#[derive(Clone)]
pub struct InitHead<B> {
	key: Vec<u8>,
	backend: B,
}

impl<V, S, D, M, B> InitFrom for Head<V, S, D, M, B>
	where
		D: InitFrom<Init = ()>,
		B: Clone,
//		M: NodesMeta,
{
	type Init = InitHead<B>; // TODO key to clone and backend refcell.
	fn init_from(init: Self::Init) -> Self {
		Head {
			inner: Node {
				data: D::init_from(()),
				changed: false,
				reference_len: 0,
				_ph: PhantomData,
			},
			fetched: RefCell::new(Vec::new()),
			old_start_index: 0,
			start_node_state: None,
			start_node_index: 0,
			end_node_index: 0,
			reference_key: init.key,
			nodes_storage: init.backend,
		}
	}
}
	
impl<V, S, D, M, B> LinearStorage<V, S> for Head<V, S, D, M, B>
	where
		D: InitFrom<Init = ()>,
		B: Clone,
{
	fn truncate_until(&mut self, split_off: usize) {
		unimplemented!()
	}
	fn len(&self) -> usize {
		unimplemented!()
	}
	fn st_get(&self, index: usize) -> Option<HistoriedValue<V, S>> {
		unimplemented!()
	}
	fn get_state(&self, index: usize) -> Option<S> {
		unimplemented!()
	}
	fn push(&mut self, value: HistoriedValue<V, S>) {
		unimplemented!()
	}
	fn insert(&mut self, index: usize, value: HistoriedValue<V, S>) {
		unimplemented!()
	}
	fn remove(&mut self, index: usize) {
		unimplemented!()
	}
	fn pop(&mut self) -> Option<HistoriedValue<V, S>> {
		unimplemented!()
	}
	fn clear(&mut self) {
		unimplemented!()
	}
	fn truncate(&mut self, at: usize) {
		unimplemented!()
	}
	fn emplace(&mut self, at: usize, value: HistoriedValue<V, S>) {
		unimplemented!()
	}
}
