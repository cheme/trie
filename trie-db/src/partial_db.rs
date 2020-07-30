// Copyright 2020, 2020 Parity Technologies
//
// Licensed under the Apache License, Version .0 (the "License");
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

//! Storage of trie state with only key values and possibly indexes.
//!
//! Indexes are stored by respective depth and are only iterable for
//! a given depth.

use crate::rstd::btree_map::BTreeMap;
use crate::rstd::cmp::Ordering;
use crate::nibble::nibble_ops;
use crate::nibble::LeftNibbleSlice;

#[cfg(feature = "std")]
use std::sync::atomic::Ordering as AtomOrd;

#[cfg(feature = "std")]
use std::sync::atomic::AtomicUsize;

#[cfg(feature = "std")]
use std::sync::Arc;

/// Storage of key values.
pub trait KVBackend {
	/// Key query for a value.
	fn read(&self, key: &[u8]) -> Option<Vec<u8>>;
	/// Insert a value for a key.
	fn write(&mut self, key: &[u8], value: &[u8]);
	/// Remove any value at a key.
	fn remove(&mut self, key: &[u8]);
	/// Iterate over the values.
	fn iter<'a>(&'a self) -> KVBackendIter<'a>;
	/// Iterate over the values starting at a given position.
	// TODO see if iter_prefix might be more relevant
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a>;
}

/// Iterator over key values of a `KVBackend`.
pub type KVBackendIter<'a> = Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)> + 'a>;

/// Iterator over encoded indexes of a IndexBackend, encoded indexes also have
/// a depth (which can differ from the depth we queried).
pub type IndexBackendIter<'a> = Box<dyn Iterator<Item = (Vec<u8>, Index)> + 'a>;

impl KVBackend for BTreeMap<Vec<u8>, Vec<u8>> {
	fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
		self.get(key).cloned()
	}
	fn write(&mut self, key: &[u8], value: &[u8]) {
		self.insert(key.to_vec(), value.to_vec());
	}
	fn remove(&mut self, key: &[u8]) {
		self.remove(key);
	}
	fn iter<'a>(&'a self) -> KVBackendIter<'a> {
		Box::new(self.iter().map(|(k, v)| (k.to_vec(), v.to_vec())))
	}
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
		Box::new(self.range(start.to_vec()..).into_iter().map(|(k, v)| (k.to_vec(), v.to_vec())))
	}
}

#[cfg(feature = "std")]
#[derive(Debug)]
/// Count number of key accesses, to check if indexes are used properly.
pub struct CountCheck<DB: KVBackend>(DB, Option<Arc<AtomicUsize>>);

#[cfg(feature = "std")]
impl<DB: KVBackend> CountCheck<DB> {
	/// Instantiate a new count overlay on `KVBackend`.
	pub fn new(db: DB, active: bool) -> Self {
		let counter = if active {
			Some(Arc::new(AtomicUsize::new(0)))
		} else {
			None
		};
		CountCheck(db, counter)
	}
	/// Get current number of access to a value of the backend.
	pub fn get_count(&self) -> Option<usize> {
		self.1.as_ref().map(|counter| counter.load(AtomOrd::Acquire))
	}
}

#[cfg(feature = "std")]
impl<DB: KVBackend> KVBackend for CountCheck<DB> {
	fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
		let r = self.0.read(key);
		if r.is_some() {
			self.1.as_ref().map(|counter| counter.fetch_add(1, AtomOrd::Relaxed));
		}
		r
	}
	fn write(&mut self, key: &[u8], value: &[u8]) {
		self.0.write(key, value);
	}
	fn remove(&mut self, key: &[u8]) {
		self.0.remove(key);
	}
	fn iter<'a>(&'a self) -> KVBackendIter<'a> {
		Box::new(CountIter(self.0.iter(), self.1.clone()))
	}
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
		Box::new(CountIter(self.0.iter_from(start), self.1.clone()))
	}
}

#[cfg(feature = "std")]
/// Count number of key accesses over a KVBackendIter
pub struct CountIter<'a>(KVBackendIter<'a>, Option<Arc<AtomicUsize>>);

#[cfg(feature = "std")]
impl<'a> Iterator for CountIter<'a> {
	type Item = (Vec<u8>, Vec<u8>);

	fn next(&mut self) -> Option<Self::Item> {
		let r = self.0.next();
		if r.is_some() {
			self.1.as_ref().map(|counter| counter.fetch_add(1, AtomOrd::Relaxed));
		}
		r
	}
}

pub type IndexPosition = crate::nibble::BackingByteVec;

/// Storage of encoded index for a trie.
/// Depth is a nibble depth.
pub trait IndexBackend {
	/// Query for an index, also return the depth of this index (it can differs from
	/// the index depth when there is a nibble encoded in the node).
	fn read(&self, depth: usize, index: &[u8]) -> Option<Index>;
	/// Insert an `encode_index` with and `actual_depth` at configured `depth` for a given `index`.
	fn write(&mut self, depth: usize, index: IndexPosition, value: Index);
	/// Remove any value at a key.
	fn remove(&mut self, depth: usize, index: IndexPosition);
	/// Iterate over the index from a key.
	fn iter<'a>(&'a self, depth: usize, from_index: &[u8]) -> IndexBackendIter<'a>;
}

#[derive(Debug, Clone)]
/// Content of an index.
/// Index are only branches.
pub struct Index {
	pub encoded_node: Vec<u8>,
	pub actual_depth: usize,
}

impl Index {
	fn compare(&self, index_key: &[u8], change_key: &[u8]) -> Ordering {
		let end = self.actual_depth;
		let odd = end % nibble_ops::NIBBLE_PER_BYTE;
		let end = end / nibble_ops::NIBBLE_PER_BYTE;
		if odd == 0 {
			index_key[..end].cmp(change_key)
		} else {
			let size = self.actual_depth;
			let slice1 = LeftNibbleSlice::new(index_key).truncate(size);
			let slice2 = LeftNibbleSlice::new(change_key);
			slice1.cmp(&slice2)
		}
	}
}

/// Calculate the prefix to apply for value iterator start (and end).
/// TODO consider changing `iter_from` proto to use this input.
fn value_prefix(actual_index_depth: usize, change_key: &[u8]) -> (Vec<u8>, Option<Vec<u8>>) {
	let start = actual_index_depth;
	let odd = start % nibble_ops::NIBBLE_PER_BYTE;
	let start = start / nibble_ops::NIBBLE_PER_BYTE + if odd > 0 { 2 } else { 1 };
	// we can round index start since values are only on even position.
	let index_start = change_key[..start].to_vec();
	let index_end = end_prefix(index_start.as_slice());
	(index_start, index_end)
}

// TODO consider trying to use small vec (need to be usable in range) for result, we can even use
// nibblevec internally
fn value_prefix_index(actual_index_depth: usize, mut change_key: Vec<u8>) -> (Vec<u8>, Option<Vec<u8>>) {
	// TODO change key is always odd, some code here is useless.
	// TODO consider not returning start (especially since it can allocates).
	let start = change_key.len() * nibble_ops::NIBBLE_PER_BYTE;
	let odd = start % nibble_ops::NIBBLE_PER_BYTE;
	let start_byte = start / nibble_ops::NIBBLE_PER_BYTE + if odd > 0 { 1 } else { 0 };

	change_key.resize(start_byte + 1, 0);

	// we can round index start since values are only on even position.
	let index_start = change_key[..start_byte].to_vec();
	let index_end = end_prefix_index(index_start.as_slice(), start);
	(index_start, index_end)
}

// TODO consider trying to use small vec (need to be usable in range) for result
fn end_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
	let mut end_range = prefix.to_vec();
	while let Some(0xff) = end_range.last() {
		end_range.pop();
	}
	if let Some(byte) = end_range.last_mut() {
		*byte += 1;
		Some(end_range)
	} else {
		None
	}
}

fn end_prefix_index(prefix: &[u8], index: usize) -> Option<Vec<u8>> {
	if index == 0 {
		return None;
	}
	let mut slice = LeftNibbleSlice::new(prefix);
	let mut index_truncate = index;
	while let Some(0x0f) = slice.at(index_truncate - 1) {
		index_truncate -= 1;
		if index_truncate == 0 {
			return None;
		}
	}
	if index_truncate > 0 {
		let mut result = prefix.to_vec();
		let odd = (index_truncate) % nibble_ops::NIBBLE_PER_BYTE;
		let pos = (index_truncate - 1) / nibble_ops::NIBBLE_PER_BYTE;
		result.resize(pos + 1, 0);
		if odd > 0 {
			result[pos] += 0x10;
			result[pos] &= 0xf0;
		} else {
			result[pos] += 0x01; 
		}
		result.truncate(pos + 1);
		Some(result)
	} else {
		None
	}
}


/// Key for an index.
fn index_tree_key(depth: usize, index: &[u8]) -> IndexPosition {
	let mut result: IndexPosition = index.into();
	result.insert_from_slice(0, &(depth as u32).to_be_bytes()[..]);
	result
}

/// Key for an index with owned input.
fn index_tree_key_owned(depth: usize, mut index: IndexPosition) -> IndexPosition {
	index.insert_from_slice(0, &(depth as u32).to_be_bytes()[..]);
	index
}

/// Note that this is a test implementation, sparse matrix should be use here. 
impl IndexBackend for BTreeMap<Vec<u8>, Index> {
	fn read(&self, depth: usize, index: &[u8]) -> Option<Index> {
		self.get(&index_tree_key(depth, index)[..]).cloned()
	}
	fn write(&mut self, depth: usize, mut position: IndexPosition, index: Index) {
		let odd = index.actual_depth % nibble_ops::NIBBLE_PER_BYTE;
		if odd != 0 {
			position.last_mut().map(|l| 
				*l = *l & !(255 >> (odd * nibble_ops::BIT_PER_NIBBLE))
			);
		}
		self.insert(index_tree_key_owned(depth, position).to_vec(), index);
	}
	fn remove(&mut self, depth: usize, mut index: IndexPosition) {
		let l_size = crate::rstd::mem::size_of::<u32>();
		let start = index_tree_key_owned(depth, index);
		let range = if let Some(end_range) = end_prefix_index(&start[..], depth + l_size * nibble_ops::NIBBLE_PER_BYTE) {
			self.range(start.to_vec()..end_range)
		} else {
			self.range(start.to_vec()..)
		};

		let mut range_iter = range.into_iter();
		let first = range_iter.next().map(|kv| kv.0.clone());
		debug_assert!(range_iter.next().is_none());
		first.map(|key| self.remove(&key));
	}
	fn iter<'a>(&'a self, depth: usize, from_index: &[u8]) -> IndexBackendIter<'a> {
		let l_size = crate::rstd::mem::size_of::<u32>();
		let depth_prefix = &(depth as u32).to_be_bytes()[..];
		let start = &index_tree_key(depth, from_index);
		let range = if let Some(end_range) = end_prefix(&depth_prefix[..]) {
			self.range(start.to_vec()..end_range)
		} else {
			self.range(start.to_vec()..)
		};
		Box::new(range.into_iter().map(move |(k, ix)| {
			let mut k = k[l_size..].to_vec();
			(k, ix.clone())
		}))
	}
}

/// Depths to use for indexing.
/// We use u32 internally which means that deepest supported index is 32 byte deep.
///
/// The root (depth 0) is always indexed.
/// TODO consider switching to u64??
/// TODO put behind trait to use with more complex definition (substrate uses
/// prefixes and some indexes depth should only be define for those).
pub struct DepthIndexes(smallvec::SmallVec<[u32; 16]>);

impl crate::rstd::ops::Deref for DepthIndexes {
	type Target = smallvec::SmallVec<[u32; 16]>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl DepthIndexes {
	/// Instantiate a new depth indexes configuration.
	pub fn new(indexes: &[u32]) -> Self {
		DepthIndexes(indexes.into())
	}

	/// Returns next_index depth to register, starting at a given included depth.
	///
	/// We use an index position to be able to define indexing only for given prefixes
	/// (needed only for unbalanced trie).
	/// TODO this is not really efficient and use on every node
	/// TODO put in IdenxesConf trait.
	pub fn next_depth(&self, depth: usize, _index: &[u8]) -> Option<usize> {
		if depth == 0 {
			// 0 is always indexed to remove many corner case at very small cost.
			return Some(0);
		}
		for i in self.0.iter() {
			let i = *i as usize;
			if i >= depth {
				return Some(i)
			}
		}
		None
	}
	
	/// See `next_depth`.
	pub fn previous_depth(&self, depth: usize, _index: &[u8]) -> Option<usize> {
		let mut result = None;
		for i in self.0.iter() {
			let i = *i as usize;
			if i >= depth {
				break;
			}
			result = Some(i);
		}
		result
	}
}

/// Enum containing either a value or an index, mostly for internal use
/// (see `RootIndexIterator` and `iter_build::trie_visit_with_indexes`).
pub enum IndexOrValue<B> {
	/// Contains depth as number of bit and the encoded value of the node for this index.
	/// Also an optional `Change of value is attached`.
	/// TODO probably do not need the usize
	Index(Index, Option<Option<B>>),
	/// Value node value, from change set.
	Value(B),
	/// Value node value, from existing values.
	StoredValue(Vec<u8>),
}

/// Iterator over index and value for root calculation
/// and index update.
pub struct RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	values: &'a KB,
	indexes: &'a IB,
	indexes_conf: &'a DepthIndexes,
	values_delta: ID,
	pub deleted_indexes: Vec<(usize, IndexPosition)>,
	current_index_depth: usize,
	/// value iterator in use and the range that needs to be covered.
	current_value_iter: Option<(KVBackendIter<'a>, (Vec<u8>, Option<Vec<u8>>))>,
	index_iter: Vec<StackedIndex<'a>>,
	next_change: Option<(Vec<u8>, Option<V>)>,
	next_value: Option<(Vec<u8>, Vec<u8>)>,
	// instead of an iteration we try to stack an index over
	stack_index_next: bool,
}

struct StackedIndex<'a> {
	iter: IndexBackendIter<'a>, 
	range: (Vec<u8>, Option<Vec<u8>>),
	next_index: Option<(Vec<u8>, Index)>,
}

impl<'a, KB, IB, V, ID> RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	pub fn new(
		values: &'a KB,
		indexes: &'a IB,
		indexes_conf: &'a DepthIndexes,
		values_delta: ID,
		deleted_indexes: Vec<(usize, IndexPosition)>,
	) -> Self {

		RootIndexIterator {
			values,
			indexes,
			indexes_conf,
			values_delta,
			deleted_indexes,
			current_index_depth: 0,
			current_value_iter: None,
			index_iter: Vec::new(),
			next_change: None,
			next_value: None,
			stack_index_next: true,
		}
	}
}
	
impl<'a, KB, IB, V, ID> Iterator for RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	type Item = (Vec<u8>, IndexOrValue<V>);

	fn next(&mut self) -> Option<Self::Item> {
		// init next iterator values
		if self.next_change.is_none() {
			if let Some(next_change) = self.values_delta.next() {
				if next_change.0.len() > self.current_index_depth {
					// TODO this should trigger directly
					self.stack_index_next = true;
				}
				self.next_change = Some(next_change)
			}
		}
		// first index init
		if !self.is_buffed_next_index() {
			if self.stack_index_next {
//				if let Some(next_change) = self.next_change.as_ref() { // TODO could avoid clone by restructuring struct
//					let next_change_index = next_change.0.as_slice();
				if self.next_change.is_some() {
					let next_change_index = self.next_change.as_ref().map(|n| n.0.clone()).expect("checked above");
					self.stack_index(next_change_index, false);
				} else {
					// no change and index at root, just iterate all first level indexes
					self.stack_index(Vec::new(), true);
				}
			}
			self.index_iter.last_mut().map(|i| {
				i.next_index = i.iter.next()
					.filter(|kv| {
						i.range.1.as_ref().map(|end| &kv.0 < end).unwrap_or(true)
					});
			});
		}
		// end of an indexing range
		while !self.is_buffed_next_index() {
			if !self.pop_index() {
				break;
			}
			self.current_value_iter = None;
			self.index_iter.last_mut().map(|i| {
				i.next_index = i.iter.next()
					.filter(|kv| {
						i.range.1.as_ref().map(|end| &kv.0 < end).unwrap_or(true)
					});
			});
		}
		if let Some(next_change) = &self.next_change {
			if self.current_value_iter.is_none() {
				// init a new
				let start_depth = self.current_index_depth + 1;
				let range = value_prefix(start_depth, next_change.0.as_slice());
				let iter = self.values.iter_from(range.0.as_slice());
				self.current_value_iter = Some((iter, range));
			}
		}
		if self.next_value.is_none() {
			self.next_value = self.current_value_iter.as_mut().and_then(|iter|
				iter.0.next().filter(|kv| (iter.1).1.as_ref().map(|end| &kv.0 < end).unwrap_or(true)));
		}

		// return lesser item and apply equality rule.
		match (&self.next_change, &self.next_value) {
			(Some(next_change), Some(next_value)) => {
				match next_change.0.cmp(&next_value.0) {
					Ordering::Equal => {
						// use change (not that a delete change also drop value)
						self.next_value = None;
						self.next_change_or_index()
					},
					Ordering::Greater => {
						self.next_value_or_index()
					},
					Ordering::Less => {
						self.next_change_or_index()
					},
				}
			},
			(Some(_next_change), None) => {
				self.next_change_or_index()
			},
			(None, Some(_next_value)) => {
				self.next_value_or_index()
			},
			(None, None) => {
				self.next_index(None)
			},
		}
	}
}

impl<'a, KB, IB, V, ID> RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	//fn stack_index(&mut self, current_key: &[u8]) -> bool {
	fn stack_index(&mut self, current_key_vec: Vec<u8>, is_root_last: bool) -> bool {
		self.stack_index_next = false;
		let current_key = current_key_vec.as_slice();
		let current_index_depth = if self.index_iter.is_empty() {
			0
		} else {
			self.current_index_depth + 1
		};
		let index_iter = &mut self.index_iter;
		let indexes = &self.indexes;
		self.indexes_conf.next_depth(current_index_depth, current_key)
			.map(move |d| (d, {
				let current_key = current_key_vec.as_slice();
				if !is_root_last && current_key.len() * nibble_ops::NIBBLE_PER_BYTE < d {
					let iter = indexes.iter(d, current_key);
					//let mut current_key = current_key.to_vec();
					
					let range = value_prefix_index(d, current_key_vec);
					// TODO need to store current value of next_index too?? as current_index_depth!!
					index_iter.push(StackedIndex {
						iter,
						range,
						next_index: None, // TODO could init here but lazy at this point.
					});
				} else {
					let iter = indexes.iter(d, current_key);
					let range = if is_root_last {
						(current_key_vec, None)
					} else {
						value_prefix_index(d, current_key_vec)
					};
					// TODO need to store current value of next_index too??
					index_iter.push(StackedIndex {
						iter,
						range,
						next_index: None, // TODO could init here but lazy at this point.
					});
				}
				//}
			})).is_some()
	}

	fn pop_index(&mut self) -> bool {
		self.index_iter.pop().is_some()
	}

	fn next_change_or_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		match (self.buffed_next_index(), &self.next_change) {
			(Some(next_index), Some(next_change)) => {
				match next_index.1.compare(&next_index.0, &next_change.0) {
					Ordering::Equal => {
						// use a fused changed index
						let next_change = self.next_change.take();
						self.next_index(Some(next_change.and_then(|c| c.1)))
					},
					Ordering::Greater => self.next_change(),
					Ordering::Less => self.next_index(None),
				}
			},
			(Some(_next_index), None) => self.next_index(None),
			(None, Some(_next_change)) => self.next_change(),
			(None, None) => None,
		}
	}

	fn next_change(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		if self.stack_index_next {
			let next_change_index = self.next_change.as_ref().map(|n| n.0.clone())
				.expect("stack_index_next is set through an initialized next change");
			self.stack_index(next_change_index, false);
			// compare again.
			return self.next();
		}
		let next_change = self.next_change.take();
		if let Some((key, Some(change))) = next_change {
			Some((key, IndexOrValue::Value(change)))
		} else {
			// skip delete
			self.next()
		}
	}

	fn is_buffed_next_index(&self) -> bool {
		self.index_iter.last().map(|item| item.next_index.is_some()).unwrap_or(false)
	}

	fn buffed_next_index(&self) -> Option<&(Vec<u8>, Index)> {
		self.index_iter.last().and_then(|item| item.next_index.as_ref())
	}

	fn next_index(&mut self, change: Option<Option<V>>) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		let current_index_depth = &mut self.current_index_depth;
		// stop value iteration after index
		self.next_value = None;
		self.current_value_iter = None;
		self.index_iter.last_mut().and_then(|i|
			i.next_index.take().map(|index| {
				*current_index_depth = (index.1).actual_depth;
				(index.0, IndexOrValue::Index(index.1, change))
			})
		)
	}

	fn next_value(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		self.next_value.take().map(|value| (value.0, IndexOrValue::StoredValue(value.1)))
	}

	fn next_value_or_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		match (self.buffed_next_index(), &self.next_value) {
			(Some(next_index), Some(next_value)) => {
				match next_index.1.compare(&next_index.0, &next_value.0) {
					Ordering::Equal => unreachable!("Value are not indexed"),
/*						// TODO could debug assert equality of values
						self.next_index(None)
					},*/
					Ordering::Less => self.next_index(None),
					Ordering::Greater => self.next_value(),
				}
			},
			(Some(_next_index), None) => self.next_index(None),
			(None, Some(_next_value)) => self.next_value(),
			(None, None) => None,
		}
	}
}

pub enum Next {
	Ascend,
	Descend,
	Value,
	Index(Option<usize>),
}
