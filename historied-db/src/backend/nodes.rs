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
use crate::rstd::cell::{RefCell, Ref, RefMut};
use crate::rstd::rc::Rc;
use crate::rstd::vec::Vec;
use super::{LinearStorage};
use crate::historied::HistoriedValue;
use derivative::Derivative;
use crate::InitFrom;
use crate::backend::encoded_array::EncodedArrayValue;

pub trait EstimateSize {
	fn estimate_size(&self) -> usize;
}

/// Node storage metadata
pub trait NodesMeta: Sized {
	fn max_head_len() -> usize;
	/// for imbrincated nodes we can limit
	/// by number of head items instead of
	/// max_head_len.
	fn max_head_items() -> Option<usize>;
	fn max_node_len() -> usize;
	fn max_node_items() -> Option<usize>;
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
/// TODO useless? or factor over it??
pub struct NodeFromHead<V, S, D, M, B>(Head<V, S, D, M, B>, usize);

/// Head is the entry node, it contains fetched nodes and additional
/// information about this backend state.
pub struct Head<V, S, D, M, B> {
	inner: Node<V, S, D, M>,
	fetched: RefCell<Vec<Node<V, S, D, M>>>, // TODO consider smallvec
	old_start_index: u32,
	start_node_index: u32,
	end_node_index: u32,
	start_index: usize,
	end_index: usize,
	reference_key: Vec<u8>,
	backend: B,
}

impl<V, S, D: Clone, M, B> Head<V, S, D, M, B>
	where
		M: NodesMeta,
		B: NodeStorageMut<V, S, D, M>,
{
	pub fn flush(&mut self) {
		for d in self.old_start_index .. self.start_node_index {
			self.backend.remove_node(&self.reference_key[..], d);
		}
		self.old_start_index = self.start_node_index;
		for (index, mut node) in self.fetched.borrow_mut().iter_mut().enumerate() {
			if node.changed {
				self.backend.set_node(&self.reference_key[..], index as u32 + self.start_node_index, node);
				node.changed = false;
			}
		}
	}
}

#[derive(Clone)]
pub struct InitHead<B> {
	pub key: Vec<u8>,
	pub backend: B,
}

impl<V, S, D, M, B> InitFrom for Head<V, S, D, M, B>
	where
		D: InitFrom<Init = ()>,
		B: Clone,
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
			start_node_index: 0,
			end_node_index: 0,
			start_index: 0,
			end_index: 0,
			reference_key: init.key,
			backend: init.backend,
		}
	}
}

impl<V, S, D, M, B> LinearStorage<V, S> for Head<V, S, D, M, B>
	where
		D: InitFrom<Init = ()> + LinearStorage<V, S>,
		B: NodeStorageHandle<V, S, D, M>,
		M: NodesMeta,
		S: EstimateSize,
		V: EstimateSize,
{
	fn len(&self) -> usize {
		(self.end_node_index - self.start_node_index) as usize
	}
	fn st_get(&self, index: usize) -> Option<HistoriedValue<V, S>> {
		if self.start_index <= index && self.end_index > index {
			let mut start = self.end_index as usize - self.inner.data.len();
			if index < start {
				return self.inner.data.st_get(index - start);
			}
			let mut i = self.end_node_index as usize;
			while i > 0 {
				i -= 1;
				if let Some(node) = self.fetched.borrow().get(self.end_node_index as usize - i - 1) {
					start -= node.data.len();
					if index >= start {
						return node.data.st_get(index - start);
					}
				} else {
					if let Some(node) = self.backend.get_node(self.reference_key.as_slice(), i as u32) {
						start -= node.data.len();
						let r = if index < start {
							node.data.st_get(index - start)
						} else {
							None
						};
						self.fetched.borrow_mut().push(node);

						if r.is_some() {
							return r;
						}
					} else {
						return None;
					}
				}
			}
		}
		None
	}
	fn get_state(&self, index: usize) -> Option<S> {
		// TODO macroed this rendundant code
		if self.start_index <= index && self.end_index > index {
			let mut start = self.end_index as usize - self.inner.data.len();
			if index >= start {
				return self.inner.data.get_state(index - start);
			}
			let mut i = self.end_node_index as usize;
			while i > 0 {
				i -= 1;
				if let Some(node) = self.fetched.borrow().get(self.end_node_index as usize - i - 1) {
					start -= node.data.len();
					if index >= start {
						return node.data.get_state(index - start);
					}
				} else {
					if let Some(node) = self.backend.get_node(self.reference_key.as_slice(), i as u32) {
						start -= node.data.len();
						let r = if index < start {
							node.data.get_state(index - start)
						} else {
							None
						};
						self.fetched.borrow_mut().push(node);

						if r.is_some() {
							return r;
						}
					} else {
						return None;
					}
				}
			}
		}
		None
	}
	fn truncate_until(&mut self, split_off: usize) {
		unimplemented!()
	}
	fn push(&mut self, value: HistoriedValue<V, S>) {
		let mut additional_size: Option<usize> = None;
		
		if let Some(nb) = M::max_head_items() {
			if self.inner.data.len() < nb {
				self.inner.data.push(value);
				self.end_index += 1;
				return;
			}
		} else {
			let add_size = value.value.estimate_size() + value.state.estimate_size(); 
			additional_size = Some(add_size);
			if self.inner.reference_len + add_size < M::max_head_len() {
				self.inner.reference_len += add_size;
				self.inner.data.push(value);
				self.end_index += 1;
				return;
			}
		}

		let add_size = additional_size.unwrap_or_else(||
			value.value.estimate_size() + value.state.estimate_size()
		);
		self.end_node_index += 1;
		self.end_index += 1;
		let mut data = D::init_from(());
		data.push(value);
		let new_node = Node::<V, S, D, M> {
			data,
			changed: true,
			reference_len: add_size,
			_ph: PhantomData,
		};
		self.inner.changed = true;
		let prev = crate::rstd::mem::replace(&mut self.inner, new_node);
		self.fetched.borrow_mut().insert(0, prev);
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

// TODO use size of instead of u8
impl EstimateSize for Vec<u8> {
	fn estimate_size(&self) -> usize {
		self.len()
	}
}

impl EstimateSize for u32 {
	fn estimate_size(&self) -> usize {
		4
	}
}

impl EstimateSize for u16 {
	fn estimate_size(&self) -> usize {
		2
	}
}

impl<V: EstimateSize> EstimateSize for Option<V> {
	fn estimate_size(&self) -> usize {
		1 + self.as_ref().map(|v| v.estimate_size()).unwrap_or(0)
	}
}

impl<V: EstimateSize, S: EstimateSize> EstimateSize for crate::backend::in_memory::MemoryOnly<V, S> {
	fn estimate_size(&self) -> usize {
		unimplemented!("This should be avoided");
	}
}

//D is backend::encoded_array::EncodedArray<'_, std::vec::Vec<u8>, backend::encoded_array::DefaultVersion>
// B is std::collections::BTreeMap<std::vec::Vec<u8>, backend::nodes::Node<std::vec::Vec<u8>, u32, backend::encoded_array::EncodedArray<'_, std::vec::Vec<u8>, backend::encoded_array::DefaultVersion>, backend::nodes::test::MetaSize>>
impl<D, M, B> EncodedArrayValue for Head<Vec<u8>, u32, D, M, B>
	where
		D: EncodedArrayValue,
{
	fn from_slice(_slice: &[u8]) -> Self {
		unimplemented!("Require a backend : similar to switch from default to init from, also required to parse meta: using specific size of version would allow fix length meta encode")
	}
}

impl<D, M, B> AsRef<[u8]> for Head<Vec<u8>, u32, D, M, B>
	where
		D: AsRef<[u8]>,
{
	fn as_ref(&self) -> &[u8] {
		self.inner.data.as_ref()
	}
}

impl<D, M, B> AsMut<[u8]> for Head<Vec<u8>, u32, D, M, B>
	where
		D: AsMut<[u8]>,
{
	fn as_mut(&mut self) -> &mut [u8] {
		self.inner.data.as_mut()
	}
}

impl<V, S, D: EstimateSize, M, B> EstimateSize for Head<V, S, D, M, B> {
	fn estimate_size(&self) -> usize {
		// TODO this is true for encoded but not inmemory where we encode
		// consider different implemetation
		self.inner.data.estimate_size()
	}
}

#[cfg(test)]
pub(crate) mod test {
	use super::*;

	use crate::backend::in_memory::MemoryOnly;
	use crate::backend::encoded_array::{EncodedArray, DefaultVersion};

	#[derive(Clone)]
	pub(crate) struct MetaSize;
	impl NodesMeta for MetaSize {
		fn max_head_len() -> usize { 25 }
		fn max_head_items() -> Option<usize> { None }
		fn max_node_len() -> usize { 30 }
		fn max_node_items() -> Option<usize> { None }
		fn max_index_len() -> usize {
			unimplemented!("no index");
		}
		fn storage_prefix() -> &'static [u8] { b"nodes1" }
	}
	#[derive(Clone)]
	pub(crate) struct MetaNb;
	impl NodesMeta for MetaNb {
		fn max_head_len() -> usize { 25 }
		fn max_head_items() -> Option<usize> { Some(3) }
		fn max_node_len() -> usize { 30 }
		fn max_node_items() -> Option<usize> { Some(3) }
		fn max_index_len() -> usize {
			unimplemented!("no index");
		}
		fn storage_prefix() -> &'static [u8] { b"nodes2" }
	}

	#[test]
	fn nodes_push_and_query() {
		nodes_push_and_query_inner::<MemoryOnly<Vec<u8>, u32>, MetaSize>();
		nodes_push_and_query_inner::<MemoryOnly<Vec<u8>, u32>, MetaNb>();
		nodes_push_and_query_inner::<EncodedArray<Vec<u8>, DefaultVersion>, MetaSize>();
		nodes_push_and_query_inner::<EncodedArray<Vec<u8>, DefaultVersion>, MetaNb>();
	}

	fn nodes_push_and_query_inner<D, M>()
		where
			D: InitFrom<Init = ()> + LinearStorage<Vec<u8>, u32> + Clone,
			M: NodesMeta + Clone,
	{
		let init_head = InitHead {
			backend: BTreeMap::<Vec<u8>, Node<Vec<u8>, u32, D, M>>::new(),
			key: b"any".to_vec(),
		};
		let mut head = Head::<Vec<u8>, u32, D, M, _>::init_from(init_head);
		assert_eq!(head.get_state(0), None);
		for i in 0usize..30 {
			let modu = i % 3;
			head.push(HistoriedValue {
				value: vec![i as u8; 2 + modu],
				state: i as u32,
			});
			for j in 0..i + 1 {
				assert_eq!(head.get_state(j), Some(j as u32));
			}
			assert_eq!(head.get_state(i + 1), None);
		}
	}
}
