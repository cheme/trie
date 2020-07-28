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

#[derive(Debug, Clone)]
// TODO is 32 enough??
pub struct IndexPosition(smallvec::SmallVec<[u8; 32]>);

impl crate::rstd::ops::Deref for IndexPosition {
	type Target = smallvec::SmallVec<[u8; 32]>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl crate::rstd::ops::DerefMut for IndexPosition {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}


impl IndexPosition {
	/// Instantiate a new depth indexes configuration.
	pub fn new(position: &[u8]) -> Self {
		IndexPosition(position.into())
	}
}

/// Storage of encoded index for a trie.
/// Depth is a bit depth. For hexary trie user should only provide multiple
/// of 4 as depth.
pub trait IndexBackend {
	/// Query for an index, also return the depth of this index (it can differs from
	/// the index depth when there is a nibble encoded in the node).
	fn read(&self, depth: usize, index: &[u8]) -> Option<Index>;
	/// Insert an `encode_index` with and `actual_depth` at configured `depth` for a given `index`.
	fn write(&mut self, depth: usize, index: IndexPosition, index: Index);
	/// Remove any value at a key.
	fn remove(&mut self, depth: usize, index: IndexPosition);
	/// Iterate over the index from a key.
	fn iter<'a>(&'a self, depth: usize, from_index: &[u8]) -> IndexBackendIter<'a>;
}

#[derive(Debug, Clone)]
/// Content of an index.
pub struct Index {
	pub encoded_node: Vec<u8>,
	pub actual_depth: usize,
	pub is_leaf: bool, // TODO probably useless.
}

impl Index {
	fn compare(&self, index_key: &[u8], change_key: &[u8]) -> Ordering {
		let end = self.actual_depth;
		let odd = end % 8;
		let end = end / 8;
		if odd == 0 {
			index_key[..end].cmp(change_key)
		} else {
			let size = self.actual_depth / nibble_ops::BIT_PER_NIBBLE;
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
	let odd = start % 8;
	let start = start / 8 + if odd > 0 { 2 } else { 1 };
	// we can round index start since values are only on even position.
	let index_start = change_key[..start].to_vec();
	let index_end = end_prefix(index_start.as_slice());
	(index_start, index_end)
}

/// Key for an index.
fn index_tree_key(depth: usize, index: &[u8]) -> IndexPosition {
	let mut result = IndexPosition::new(index);
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
		let odd = index.actual_depth % 8;
		if odd != 0 {
			position.last_mut().map(|l| 
				*l = *l & !(255 >> odd)
			);
		}
		self.insert(index_tree_key_owned(depth, position).to_vec(), index);
	}
	fn remove(&mut self, depth: usize, mut index: IndexPosition) {
/*		let odd = index.actual_depth % 8;
		if odd != 0 {
			index.last_mut().map(|l| 
				*l = *l & !(255 >> odd)
			);
		}*/ // TODO the masking need an additional param
		self.remove(&index_tree_key_owned(depth, index)[..]);
	}
	fn iter<'a>(&'a self, depth: usize, from_index: &[u8]) -> IndexBackendIter<'a> {
		let depth_prefix = &(depth as u32).to_be_bytes()[..];
		let start = &index_tree_key(depth, from_index);
		let range = if let Some(end_range) = end_prefix(&depth_prefix[..]) {
			self.range(start.to_vec()..end_range)
		} else {
			self.range(start.to_vec()..)
		};
		Box::new(range.into_iter().map(|(k, ix)| {
			let mut k = k[crate::rstd::mem::size_of::<u32>()..].to_vec();
			(k, ix.clone())
		}))
	}
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


/// Depths to use for indexing.
/// We use u32 internally which means that deepest supported index is 32 byte deep.
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

	/// Return number of index for a given depth.
	/// Panic for `depth` out of range (maximum 32).
	pub fn indexing_range_nb(depth: usize) -> usize {
		1 << depth
	}

/*	/// Return indexing range (inclusive, exclusive) for a given index.
	/// End if None is the end of the trie
	pub fn indexing_range(depth: usize, index: usize) -> (usize, Option<usize>) {
		if depth == 0 {
			return (0, None);
		}
		let size = self.indexing_range_nb(depth - 1);

		unimplemented!()
	}*/

	/// Returns prefix of keys needed to calculate this key change.
	/// We also return the corresponding index depth (also calculable from key length)
	/// and its indexing at this depth.
	pub fn prefix(depth: usize, index: usize) -> [u8; 32] {
		unimplemented!()
	}

	/// return depth of best range
	pub fn best_depth(&self, key: &[u8]) -> usize {
		let depth = key.len() * 8;
		let mut result = 0;
		for i in self.0.iter() {
			let i = *i as usize;
			if depth > i {
				 break;
			}
			result = i;
		}
		result	
	}
	
	/// Return first index position corresponding to a parent depth at a child depth.
	/// Assumes child depth is strictly > to parent depth and key is long enough for parent depth.
	pub fn first_index(&self, parent_depth: usize, child_depth: usize, key: &[u8]) -> IndexPosition {
		unimplemented!()
	}
	/// TODO consider modifing an input IndexPositon
	pub fn next_index(&self, parent_depth: usize, child_depth: usize, key: &[u8]) -> Option<IndexPosition> {
		unimplemented!()
	}
	/// Same as first index but returning the last index.
	/// This last index is inclusive.
	pub fn last_index(&self, parent_depth: usize, child_depth: usize, key: &[u8]) -> IndexPosition {
		unimplemented!()
	}


	/// Range of value to process.
	/// Return None if we do not change range.
	/// If returned range is a value range, it contains next_change_key.
	pub fn next_range(
		&self,
		previous_range: &Range,
		next_change_key: Option<&[u8]>,
	) -> NextRange {
		// TODO could memoize a lot of processing here (all resolution of target and
		// next key target depending on common range).

	let index_depth = self.0[0] as usize;
	let (old_target, old_position, was_value) = match previous_range {
		Range::ValuesPrefix(old_target, old_position) => (old_target, old_position, true),
		Range::Index(old_target, old_position) => (old_target, old_position, false),
	};
	if old_target == &0 {
		if let Some(next_change_key) = next_change_key.as_ref() {
			let next_depth = index_depth;
			if next_change_key.len() * 8 >= index_depth {
				let position = IndexPosition::new(next_change_key);
				let first_index = self.first_index(*old_target, next_depth, position.as_slice());
				let mut range = Range::Index(next_depth, first_index.clone());
				if range.contains(next_change_key) {
					range = Range::ValuesPrefix(next_depth, first_index);
				}
				NextRange::Descend(range)
			} else {
				NextRange::Stay
			}
		} else {
			NextRange::Ascend // actually sibling of current but since level 0 it is stay TODO generalize to multiple levels
		}
	} else {
		assert!(old_target == &index_depth, "first implementation on single index only");
		let parent_target = 0;
		if let Some(next_change_key) = next_change_key.as_ref() {
			if previous_range.contains(next_change_key) {
				return NextRange::Stay;
			}
		}
		if let Some(i) = self.next_index(parent_target, index_depth, old_position.as_slice()) {
			let mut range = Range::Index(index_depth, i.clone());
			if let Some(next_change_key) = next_change_key.as_ref() {
				if range.contains(next_change_key) {
					range = Range::ValuesPrefix(index_depth, i);
				}
			}
			NextRange::Sibling(range)
		} else {
			NextRange::Ascend
		}
	}
/* TODO use multiple level but first run on first level only
		if let Range::ValuesPrefix(oldtarget, _) = previous_range {
			if let Some(next_change_key) = next_change_key.as_ref() {
				if previous_range.contains(next_change_key) {
					if let Some(target) = self.next_depth(*oldtarget, next_change_key) {
						if next_change_key.len() >= target
					}
					if &target == oldtarget {
						// stay on same range
						return None;
					} else {
						let position = IndexPosition::new(next_change_key);
						let first_index = self.first_index(*oldtarget, target, position.as_slice());
						if position.starts_with(first_index.as_slice()) {
							return Some(Range::ValuesPrefix(target, position));
						} else {
							return Some(Range::Index(target, first_index));
						}

					}
				}
			}
		}
*/

	}
}

pub enum NextRange {
	/// End of range, unstack context.
	Ascend,
	/// Advance.
	Sibling(Range),
	/// Descend, contains first next range.
	Descend(Range),
	/// Next value is in same range.
	Stay,
}

pub enum Range { // TODO switch to a struct
	ValuesPrefix(usize, IndexPosition),
	// depth and index
	Index(usize, IndexPosition),
}

impl Range {
	fn is_value(&self) -> bool {
		if let Range::ValuesPrefix(_, _) = self {
			true
		} else {
			false
		}
	}
	fn contains(&self, key: &[u8]) -> bool {
		let (depth, position) = match self {
			Range::ValuesPrefix(depth, pref) => {
				(depth, pref)
			},
			Range::Index(depth, position) => {
				(depth, position)
			},
		};
		if key.len() * 8 < *depth {
			return false;
		}
		let odd = depth % 8;
		if odd == 0 {
			key.starts_with(position.as_slice())
		} else {
			let position_len = position.as_slice().len();
			key.starts_with(&position[..position_len - 1])
				&& key[position_len - 1] & !(255 >> odd)
					== position[position_len - 1] & !(255 >> odd)
		}
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
	index_iter: Vec<(IndexBackendIter<'a>, (Vec<u8>, Option<Vec<u8>>))>,
	next_change: Option<(Vec<u8>, Option<V>)>,
	next_value: Option<(Vec<u8>, Vec<u8>)>,
	next_index: Option<(Vec<u8>, Index)>,
	// instead of an iteration we try to stack an index over
	stack_index_next: bool,
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
			next_index: None,
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
					self.stack_index_next = true;
				}
				self.next_change = Some(next_change)
			}
		}
		if self.next_index.is_none() {
			if self.stack_index_next {
				self.stack_index_next = false;
				if let Some(next_change) = self.next_change.as_ref() {
					let next_change_index = next_change.0.as_slice();
					let current_index_depth = if self.index_iter.is_empty() {
						0
					} else {
						self.current_index_depth + 1
					};
					let index_iter = &mut self.index_iter;
					let indexes = &self.indexes;
					self.indexes_conf.next_depth(current_index_depth, next_change_index)
						.map(|d| (d, {
							let range = value_prefix(d, next_change_index);
							index_iter.push((indexes.iter(d, next_change_index), range))
						}));
				} else {
					// no change and index at root, just iterate all first level indexes
					let index_iter = &mut self.index_iter;
					let indexes = &self.indexes;
					self.indexes_conf.next_depth(0, &[])
						.map(|d| (d, {
							index_iter.push((indexes.iter(d, &[]), (Vec::new(), None)))
						}));
				}
			}
			self.next_index = self.index_iter.last_mut().and_then(|i| {
				i.0.next()
					.filter(|kv| {
						(i.1).1.as_ref().map(|end| &kv.0 < end).unwrap_or(true)
					})
			});
		}
		// end of an indexing range
		while self.next_index.is_none() {
			if self.index_iter.pop().is_none() {
				break;
			}
			self.current_value_iter = None;
			self.next_index = self.index_iter.last_mut().and_then(|i| {
				i.0.next()
					.filter(|kv| (i.1).1.as_ref().map(|end| &kv.0 < end).unwrap_or(true))
			});
		}
		if let Some(next_change) = &self.next_change {
			if self.current_value_iter.is_none() {
				// init a new
				let start_depth = self.current_index_depth + nibble_ops::BIT_PER_NIBBLE; // TODO switch depth to nibble depth
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
				return self.next_index(None);
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
	fn next_change_or_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		match (&self.next_index, &self.next_change) {
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
		let next_change = self.next_change.take();
		if let Some((key, Some(change))) = next_change {
			Some((key, IndexOrValue::Value(change)))
		} else {
			// skip delete
			self.next()
		}
	}

	fn next_index(&mut self, change: Option<Option<V>>) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		self.next_index.take().map(|index| {
			self.current_index_depth = (index.1).actual_depth;
			(index.0, IndexOrValue::Index(index.1, change))
		})
	}

	fn next_value(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		self.next_value.take().map(|value| (value.0, IndexOrValue::StoredValue(value.1)))
	}

	fn next_value_or_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		match (&self.next_index, &self.next_value) {
			(Some(_next_index), Some(_next_value)) => {
				unreachable!("value iterator should not overlap an index");
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
