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


// TODO when stacking index, value iter can be reuse if following change is bellow index.

use crate::rstd::btree_map::BTreeMap;
use crate::rstd::cmp::Ordering;
use crate::nibble::nibble_ops;
use crate::nibble::LeftNibbleSlice;
use crate::nibble::NibbleVec;
pub use crate::iter_build::SubIter;
use crate::rstd::vec;
use crate::rstd::vec::Vec;
use crate::rstd::boxed::Box;

#[cfg(feature = "std")]
use std::sync::atomic::Ordering as AtomOrd;

#[cfg(feature = "std")]
use std::sync::atomic::AtomicUsize;

use crate::rstd::sync::Arc;

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
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a>;
}

impl KVBackend for Box<dyn KVBackend + Send + Sync> {
	fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
		KVBackend::read(self.as_ref(), key) 
	}
	fn write(&mut self, key: &[u8], value: &[u8]) {
		KVBackend::write(self.as_mut(), key, value) 
	}
	fn remove(&mut self, key: &[u8]) {
		KVBackend::remove(self.as_mut(), key) 
	}
	fn iter<'a>(&'a self) -> KVBackendIter<'a> {
		KVBackend::iter(&*self) 
	}
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
		KVBackend::iter_from(self.as_ref(), start) 
	}
}

impl KVBackend for Arc<dyn KVBackend + Send + Sync> {
	fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
		KVBackend::read(self.as_ref(), key) 
	}
	fn write(&mut self, key: &[u8], value: &[u8]) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn remove(&mut self, key: &[u8]) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn iter<'a>(&'a self) -> KVBackendIter<'a> {
		KVBackend::iter(&*self) 
	}
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
		KVBackend::iter_from(self.as_ref(), start) 
	}
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
	/// Iterate over the index from a key. TODO change this to include range (so depth is not all depth
	/// items but those in range corresponding to parent index depth) 
	/// Depth base is the previous index plus one (or 0 if no previous index).
	fn iter<'a>(&'a self, depth: usize, depth_base: usize, change: &[u8]) -> IndexBackendIter<'a>;
}

impl IndexBackend for Arc<dyn IndexBackend + Send + Sync> {
	fn read(&self, depth: usize, index: &[u8]) -> Option<Index> {
		IndexBackend::read(self.as_ref(), depth, index)
	}
	fn write(&mut self, depth: usize, index: IndexPosition, value: Index) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn remove(&mut self, depth: usize, index: IndexPosition) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn iter<'a>(&'a self, depth: usize, depth_base: usize, change: &[u8]) -> IndexBackendIter<'a> {
		IndexBackend::iter(self.as_ref(), depth, depth_base, change)
	}
}

impl IndexBackend for Box<dyn IndexBackend + Send + Sync> {
	fn read(&self, depth: usize, index: &[u8]) -> Option<Index> {
		IndexBackend::read(self.as_ref(), depth, index)
	}
	fn write(&mut self, depth: usize, index: IndexPosition, value: Index) {
		IndexBackend::write(self.as_mut(), depth, index, value)
	}
	fn remove(&mut self, depth: usize, index: IndexPosition) {
		IndexBackend::remove(self.as_mut(), depth, index)
	}
	fn iter<'a>(&'a self, depth: usize, depth_base: usize, change: &[u8]) -> IndexBackendIter<'a> {
		IndexBackend::iter(self.as_ref(), depth, depth_base, change)
	}
}

#[derive(Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
/// Content of an index.
/// Index is a hash, and its actual depth is parent branch depth + 1 or
/// 0 if root.
/// actual depth is <= index depth. (< when there is a partial nible overlapping
/// the index position).
pub struct Index {
	pub hash: Vec<u8>, // TODO hash as inner type ?? (no need to encode vec length here)
	/// Nibbled depth of the node in the trie.
	/// Root is 0 but not indexable. Generally the depth is the length of the node prefix.
	pub actual_depth: usize,
	/// Depth to start looking for parent index.
	pub top_depth: usize,
	/// Wether an index exists over this one.
	pub is_top_index: bool,
}

impl Index {
	// TODO check if used or can be replaced
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

/// Calculate the prefix to apply for index iterator start (and end). TODO parameters are confusing (actual_index_depth seems to be actual_index_depth + 1)
// TODO consider trying to use small vec (need to be usable in range) for result, we can even use
// nibblevec internally
// TODO input change key should be slice by refact end_prefix index??
pub fn value_prefix_index(actual_index_depth: usize, mut change_key: Vec<u8>, depth_base: usize) -> Option<Vec<u8>> {
	// TODO change key is always odd, some code here is useless.
	// TODO consider not returning start (especially since it can allocates).
	let start = actual_index_depth;
	let odd = start % nibble_ops::NIBBLE_PER_BYTE;
	let start_byte = start / nibble_ops::NIBBLE_PER_BYTE + if odd > 0 { 1 } else { 0 };

	change_key.resize(start_byte + depth_base, 0);

	// we can round index start since values are only on even position.
	let index_start = &change_key[..start_byte + depth_base];
	let index_end = if start == 0 {
		end_prefix(&index_start[..])
	} else {
		end_prefix_index(index_start, (depth_base * nibble_ops::NIBBLE_PER_BYTE) + start - 1)
	};
	index_end
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
	let slice = LeftNibbleSlice::new(prefix);
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
pub fn index_tree_key(depth: usize, index: &[u8]) -> IndexPosition {
	let mut result: IndexPosition = index.into();
	result.insert_from_slice(0, &(depth as u32).to_be_bytes()[..]);
	result
}

/// Key for an index with owned input.
pub fn index_tree_key_owned(depth: usize, mut index: IndexPosition) -> IndexPosition {
	index.insert_from_slice(0, &(depth as u32).to_be_bytes()[..]);
	index
}

/// Note that this is a test implementation, sparse matrix should be use here. 
impl IndexBackend for BTreeMap<Vec<u8>, Index> {
	fn read(&self, depth: usize, index: &[u8]) -> Option<Index> {
		self.get(&index_tree_key(depth, index)[..]).cloned()
	}
	fn write(&mut self, depth: usize, mut position: IndexPosition, index: Index) {
		// do not write single element index
		if index.actual_depth > 0 {
			let truncate = ((index.actual_depth - 1) / nibble_ops::NIBBLE_PER_BYTE) + 1;
			position.truncate(truncate);
			let unaligned = index.actual_depth % nibble_ops::NIBBLE_PER_BYTE;
			if unaligned != 0 {
				position.last_mut().map(|l| 
					*l = *l & !(255 >> (unaligned * nibble_ops::BIT_PER_NIBBLE))
				);
			}
			self.insert(index_tree_key_owned(depth, position).to_vec(), index);
		}
	}
	fn remove(&mut self, depth: usize, index: IndexPosition) {
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
	fn iter<'a>(&'a self, depth: usize, depth_base: usize, change: &[u8]) -> IndexBackendIter<'a> {
		// TODO consider util function for it as this code will be duplicated in any ordered db.
//		unimplemented!("handle parent depth");
		let l_size = crate::rstd::mem::size_of::<u32>();
		let depth_prefix = &(depth as u32).to_be_bytes()[..];
		let base = depth_prefix.len();
		let start = if depth_base > 0 {

			let mut start = index_tree_key(depth, change);
			let truncate = ((depth_base - 1) / nibble_ops::NIBBLE_PER_BYTE) + 1;
			start.truncate(truncate + base);
			let unaligned = depth_base % nibble_ops::NIBBLE_PER_BYTE;
			if unaligned != 0 {
				start.last_mut().map(|l| 
					*l = *l & !(255 >> (unaligned * nibble_ops::BIT_PER_NIBBLE))
				);
			}
			start
		} else {
			index_tree_key(depth, &[])
		};
		
		// TODO switch to IndexPosition instead of vecs
		let range = if let Some(end_range) = value_prefix_index(depth_base, start.to_vec(), base) {
			self.range(start.to_vec()..end_range)
		} else {
			self.range(start.to_vec()..)
		};
		Box::new(range.into_iter().map(move |(k, ix)| {
			let k = k[l_size..].to_vec();
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
		// No index 0 as they are ignored.
		let depth = if depth == 0 {
			1
		} else{
			depth
		};
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
	Index(Index),
	/// Value node value, from change set.
	Value(B),
	/// Dropped value node, from change set.
	DroppedValue,
	/// Value node value, from existing values.
	StoredValue(Vec<u8>),
}

impl<B> IndexOrValue<B> {
	/// Access the index branch depth (including the partial key).
	pub fn index_depth(&self) -> Option<usize> {
		match self {
			IndexOrValue::Index(index) => Some(index.actual_depth),
			_ => None,
		}
	}
	/// Access the index branch depth (including the partial key).
	pub fn index_parent_depth(&self) -> Option<usize> {
		match self {
			IndexOrValue::Index(index) => Some(index.actual_depth - 1),
			_ => None,
		}
	}
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
	changes_iter: ID,
	pub deleted_indexes: Vec<(usize, IndexPosition)>,
	current_index_depth: usize,
	current_value_iter: KVBackendIter<'a>,
	index_iter: Vec<StackedIndex<'a>>,
	next_change: Option<(Vec<u8>, Option<V>)>,
	next_value: Option<(Vec<u8>, Vec<u8>)>,
	// TODO can it be a local variable of next_index fn only??
	previous_touched_index_depth: Option<(Vec<u8>, usize)>,
	sub_iterator: Option<SubIterator<'a, V>>,
}

struct SubIterator<'a, V> {
	end_iter: Option<Vec<u8>>,
	index_iter: Option<StackedIndex<'a>>,
	current_value_iter: KVBackendIter<'a>,
	next_value: Option<(Vec<u8>, Vec<u8>)>,
	buffed_next_value: Option<(Vec<u8>, IndexOrValue<V>)>,
	previous_touched_index_depth: Option<(Vec<u8>, usize)>,
}

struct StackedIndex<'a> {
	iter: IndexBackendIter<'a>, 
	next_index: Option<(Vec<u8>, Index)>,
	conf_index_depth: usize,
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
		changes_iter: ID,
		deleted_indexes: Vec<(usize, IndexPosition)>,
	) -> Self {
		let current_value_iter = values.iter_from(&[]);
		let mut iter = RootIndexIterator {
			values,
			indexes,
			indexes_conf,
			changes_iter,
			deleted_indexes,
			current_index_depth: 0,
			current_value_iter,
			index_iter: Vec::new(),
			next_change: None,
			next_value: None,
			sub_iterator: None,
			previous_touched_index_depth: None,
		};

		// buff first change
		iter.advance_change();
		iter.advance_value();

		// always stack first level of indexes.
		iter.stack_index();
	
		iter
	}
}


enum Element {
	Value,
	Index,
	IndexChange,
	Change,
	ChangeValue,
	None,
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
		let mut last_sub = false;
		// TODO factor with standard iter
		if let Some(sub_iter) = self.sub_iterator.as_mut() {
			let action = if let Some(index_iter) = sub_iter.index_iter.as_ref() {
				match (sub_iter.next_value.as_ref(), index_iter.next_index.as_ref()) {
					(Some(next_value), Some(next_index)) => {
						match next_value.0.cmp(&next_index.0) {
							Ordering::Equal => {
								sub_iter.advance_value();
								Element::Index
							},
							Ordering::Greater => Element::Index,
							Ordering::Less => Element::Value,
						}
					},
					(Some(_next_value), None) => {
						Element::Value
					},
					(None, Some(_index)) => {
						Element::Index
					},
					(None, None) => {
						Element::None
					},
				}
			} else {
				if sub_iter.next_value.is_some() {
					Element::Value
				} else {
					Element::None
				}
			};
			match action {
				Element::Index => {
					return sub_iter.next_index(self.values);
				},
				Element::Value => {
					let result = sub_iter.next_value.take().map(|value|
						(value.0, IndexOrValue::StoredValue(value.1)));
					if result.is_some() {
						sub_iter.advance_value();
					}
					return result;
				},
				Element::None => {
					last_sub = true;
				},
				_ => unreachable!(),
			}
		}
		if last_sub {
			let result = self.sub_iterator.take()
				.expect("last_sub only when exists")
				.buffed_next_value;
			if result.is_some() {
				return result;
			} else {
				return self.next();
			}
		}
		while self.buffed_next_index().is_none() {
			if !self.pop_index() {
				break;
			}
		}
		let next_element = self.next_element();
		match next_element {
			Element::Value => false,
			Element::None => false,
			Element::IndexChange
			| Element::Change
			| Element::ChangeValue => false,
			Element::Index  => if let Some((kv, depth)) = self.buffed_next_index().map(|kv| (kv.0.clone(), kv.1.actual_depth)) {
				if self.try_stack_on_index(&kv, depth) {
					self.stack_index(); // TODO if false continue with returning index
					return self.next();
				} else {
					false
				}
			} else {
				false
			},
		};
		match next_element {
			Element::Value => self.next_value(),
			Element::Index => {
				let r = self.next_index();
				if r.is_none() {
					// TODO is that reachable??
					self.next();
				}
				r
			},
			Element::IndexChange => {
				self.advance_index();
				self.next_change()
			},
			Element::Change => self.next_change(),
			Element::ChangeValue => {
				self.advance_value();
				self.next_change()
			},
			Element::None => {
				if self.pop_index() {
					return self.next();
				}
				None
			},
		}
	}
}

impl<'a, KB, IB, V, ID> SubIter<Vec<u8>, V> for RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	fn sub_iterate(
		&mut self,
		key: &[u8],
		depth: usize,
		child_index: usize,
		buffed: Option<(Vec<u8>, IndexOrValue<V>)>,
	) {
		let mut base = NibbleVec::new();
		base.append_partial(((0, 0), key));
		let len = base.len();
		let to_drop = len - depth;
		base.drop_lasts(to_drop);
		base.push(child_index as u8); // TODO see if we don't always rebuild key??
		let key = base.inner();
		let indexes = &mut self.indexes;
		let indexes_conf = &mut self.indexes_conf;
		let end_iter = end_prefix_index(&key[..], depth + 1);
		let mut start_index_depth = depth + 1;
		let mut index_iter = None;
		// get index iterator
		while let Some(d) = indexes_conf.next_depth(start_index_depth, key) {
			let last_start_index_depth = start_index_depth;
			start_index_depth = d + 1;
			let mut iter = indexes.iter(d, last_start_index_depth, key);
			if let Some(first) = iter.next()
				.filter(|kv| end_iter.as_ref().map(|end| {
						&kv.0 < end
					}).unwrap_or(true))
			{
				if first.1.actual_depth > depth + 1 {
					index_iter = Some(StackedIndex {
						iter,
						next_index: None,
						conf_index_depth: d,
					});
					break;
				}
			} else {
				break;
			}
		};
		let current_value_iter = self.values.iter_from(&key[..]);

		let mut sub_iter = SubIterator {
			end_iter,
			index_iter,
			current_value_iter,
			buffed_next_value: buffed,
			next_value: None,
			previous_touched_index_depth: None,
		};
		sub_iter.advance_index();
		sub_iter.advance_value();
		self.sub_iterator = Some(sub_iter);
	}
}

impl<'a, KB, IB, V, ID> RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	// Return true if we stacked something and the stack item is not over current item.
	fn try_stack_on_index(&self, current_key_vec: &Vec<u8>, actual_depth: usize) -> bool {
		// see if next change is bellow
		if let Some((next_k, _)) = self.next_change.as_ref() {
			let common_depth = nibble_ops::biggest_depth(
				&current_key_vec[..],
				&next_k[..],
			);
			if common_depth < actual_depth {
				return false;
			}
			// the index is contain in the change
			true
		} else {
			false
		}
	}
/*	fn try_stack_on_change(&self, current_key_vec: &Vec<u8>) -> bool {
		// see if next change is bellow
		if let Some((index_k, actual_depth)) = self.previous_touched_index_depth() {
			let common_depth = nibble_ops::biggest_depth(
				&current_key_vec[..],
				&index_k[..],
			);
			if common_depth < actual_depth {
				return false;
			}
			// the index is contain in the change
			true
		} else {
			false
		}
	}*/

	fn stack_index(&mut self) -> bool {
		let (mut first_possible_next_index, last) = if let Some(index) = self.index_iter.last() {
			index.next_index.as_ref().map(|index| (index.1.top_depth + 1, index.1.is_top_index))
				// TODO this unwrap expression should be unreachable (condition to enter stack index).
				.unwrap_or_else(|| (index.conf_index_depth + 1, false))
		} else {
			(0, false)
		};
		self.advance_index(); // Skip this index
		if last {
			return false;
		}
		let empty: Vec<u8> = Default::default();
		let next_change_key = if let Some((next_change_key, _)) = self.next_change.as_ref() {
			next_change_key
		} else {
			&empty
		};
		let index_iter = &mut self.index_iter;
		let indexes = &self.indexes;

		// TODO this loop is a bit awkward, could be alot of empty query that are
		// a costy iterator: so it should be done at the IndexBackend level. TODO add
		// info of lower index in index ???
		loop {
			if let Some(d) = self.indexes_conf.next_depth(first_possible_next_index, next_change_key) {
				let mut iter = indexes.iter(d, first_possible_next_index, next_change_key);
				// TODO try avoid this alloc
				let first = iter.next();
	
				if first.is_some() {
					index_iter.push(StackedIndex {
						iter,
						next_index: first,
						conf_index_depth: d,
					});
					return true;
				} else {
					if d < next_change_key.len() * nibble_ops::NIBBLE_PER_BYTE {
						// TODO might/should be unreachable (if well formated trie of index)
						first_possible_next_index = d + 1
					} else {
						break;
					};
				}
			} else {
				break;
			}
		}
		false
	}

	fn advance_index(&mut self) -> bool {
		self.index_iter.last_mut().map(|i| {
			i.next_index = i.iter.next();
			i.next_index.is_some()
		}).unwrap_or(false)
	}

	fn advance_change(&mut self) {
		if let Some(next_change) = self.changes_iter.next() {
			self.next_change = Some(next_change)
		}
	}

	fn pop_index(&mut self) -> bool {
		if let Some(index) = self.index_iter.pop() {
			if self.buffed_next_index().is_none() {
				self.pop_index()
			} else {
				true
			}
		} else {
			false
		}
	}

	fn value_or_index(&mut self) -> bool {
		match (self.buffed_next_index(), &self.next_value) {
			(Some(next_index), Some(next_value)) => {
				match next_index.1.compare(&next_index.0, &next_value.0) {
					// Always favor index over value
					Ordering::Equal => {
						self.advance_value();
						false
					},
					Ordering::Less => false,
					Ordering::Greater => true,
				}
			},
			(None, Some(_next_value)) => true,
			_ => false,
		}
	}
	fn is_change_bellow_index(&self) -> bool {
		match (self.buffed_next_index(), &self.next_change) {
			(Some(next_index), Some(next_change)) => {
				let common_depth =  nibble_ops::biggest_depth(
					&next_change.0[..],
					&next_index.0[..],
				);
				let common_depth = crate::rstd::cmp::min(common_depth, next_index.1.actual_depth);
				common_depth > next_index.1.actual_depth
/*	
				// TODO this is not bellow
				match next_index.1.compare(&next_index.0, &next_change.0) {
					Ordering::Equal => false,
					Ordering::Less => true,
					Ordering::Greater => false,
				}*/
			},
			(None, Some(_next_value)) => true,
			_ => false,
		}
	}

	fn value_or_change(&self) -> Element {
		match (&self.next_change, &self.next_value) {
			(Some(next_change), Some(next_value)) => {
				match next_change.0.cmp(&next_value.0) {
					Ordering::Equal => Element::ChangeValue,
					Ordering::Less => Element::Change,
					Ordering::Greater => Element::Value,
				}
			},
			(Some(_next_change), None) => Element::Change,
			(None, Some(_next_value)) => Element::Value,
			(None, None) => Element::None,
		}
	}
	fn index_or_change(&self) -> Element {
		match (self.buffed_next_index(), &self.next_change) {
			(Some(next_index), Some(next_change)) => {
				match next_index.1.compare(&next_index.0, &next_change.0) {
					Ordering::Equal => Element::IndexChange,
					Ordering::Less => Element::Index,
					Ordering::Greater => Element::Change,
				}
			},
			(Some(_next_index), None) => Element::Index,
			(None, Some(_next_change)) => Element::Change,
			(None, None) => Element::None,
		}
	}
	
	fn next_element(&mut self) -> Element {
		if self.value_or_index() {
			self.value_or_change()
		} else {
			self.index_or_change()
		}
	}

	fn next_change(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		let next_change = self.next_change.take();
		self.advance_change();
		match next_change {
			Some((key, Some(change))) => Some((key, IndexOrValue::Value(change))),
			Some((key, None)) => Some((key, IndexOrValue::DroppedValue)),
			None => None
		}
	}

	fn buffed_next_index(&self) -> Option<&(Vec<u8>, Index)> {
		self.index_iter.last().and_then(|item| item.next_index.as_ref())
	}

	// TODO factor with next_sub_index
	fn next_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		//self.previous_touched_index_depth = None;
		let current_index_depth = &mut self.current_index_depth;
		let previous_touched_index_depth = &mut self.previous_touched_index_depth;
		// stop value iteration after index
		let result = self.index_iter.last_mut().and_then(|i|
			i.next_index.take().map(|index| {
				*current_index_depth = (index.1).actual_depth; // TODO remove current_index_depth??
				*previous_touched_index_depth = Some((index.0.clone(), (index.1).actual_depth));
				(index.0, index.1)
			})
		);
		debug_assert!(result.is_some());
		self.advance_index();
		if let Some(i) = self.index_iter.last() {
			if let Some(index) = i.next_index.as_ref() {
				if let Some(start) = self.previous_touched_index_depth.as_ref().and_then(|last_index| {
					let common_depth =  nibble_ops::biggest_depth(
						&last_index.0[..],
						&index.0[..],
					);
					// base upon last index at minimum.
					let common_depth = crate::rstd::cmp::max(common_depth, last_index.1 - 1);
					let base_depth = (common_depth + 1 + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE;
					end_prefix_index(&last_index.0[..base_depth], common_depth + 1) // TODO is there a need for base depth??(end_prefix_index doing trim for us)
				}) {
					let values = self.values.iter_from(start.as_slice()); // TODO common depth -1 of depth do not need ix
					self.current_value_iter = values;
					self.advance_value();
				} else {
					self.next_value = None; // this is same as no iterator (see next_value implementation)				
				}
			} else {
				// last interval
				if let Some(previous_touched_index_depth) = self.previous_touched_index_depth.as_ref() {
					let base_depth = if previous_touched_index_depth.1 % nibble_ops::NIBBLE_PER_BYTE > 0 {
						((previous_touched_index_depth.1) / nibble_ops::NIBBLE_PER_BYTE) + 1
					} else {
						(previous_touched_index_depth.1) / nibble_ops::NIBBLE_PER_BYTE
					};
					if let Some(start) = end_prefix_index(&previous_touched_index_depth.0[..base_depth], previous_touched_index_depth.1) {
						let values = self.values.iter_from(start.as_slice());
						self.current_value_iter = values;
						self.advance_value();
					} else {
						self.next_value = None; // this is same as no iterator (see next_value implementation)
					}
				} else {
					unreachable!("we do not stack index iter without content");
				}
			}
		}
		result.map(|(k, i)| (k, IndexOrValue::Index(i)))
	}

	fn next_value(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		let r = self.next_value.take()
			.map(|value| (value.0, IndexOrValue::StoredValue(value.1)));
		if r.is_some() {
			self.advance_value();
		}
		r
	}

	fn advance_value(&mut self) {
		self.next_value = self.current_value_iter.next();
	}
}

impl<'a, V> SubIterator<'a, V>
	where
		V: AsRef<[u8]>,
{
	fn advance_index(
		&mut self,
	) {
		let end_iter = &self.end_iter;
		if let Some(index_iter) = self.index_iter.as_mut() {
			index_iter.next_index = index_iter.iter.next()
				.filter(|kv| end_iter.as_ref().map(|end| {
					&kv.0 < end
				}).unwrap_or(true));
		}
	}

	fn advance_value(
		&mut self,
	) {
		self.next_value = self.current_value_iter.next()
			.filter(|kv| self.end_iter.as_ref().map(|end| {
				&kv.0 < end
			}).unwrap_or(true));
	}

	fn next_index<KV: KVBackend>(&mut self, values_backend: &'a KV) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		// TODO this is a copy of root 'next_index' function, could factor later.
		let previous_touched_index_depth = &mut self.previous_touched_index_depth;
		// stop value iteration after index
		let result = self.index_iter.as_mut().and_then(|i|
			i.next_index.take().map(|index| {
				*previous_touched_index_depth = Some((index.0.clone(), (index.1).actual_depth));
				(index.0, index.1)
			})
		);
		debug_assert!(result.is_some());
		self.advance_index();
		if let Some(i) = self.index_iter.as_ref() {
			if let Some(index) = i.next_index.as_ref() {
				if let Some(start) = self.previous_touched_index_depth.as_ref().and_then(|last_index| {
					let common_depth =  nibble_ops::biggest_depth(
						&last_index.0[..],
						&index.0[..],
					);
					// base upon last index at minimum.
					let common_depth = crate::rstd::cmp::max(common_depth, last_index.1 - 1);
					let base_depth = (common_depth + 1 + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE;
					end_prefix_index(&last_index.0[..base_depth], last_index.1)
				}) {
					let values = values_backend.iter_from(start.as_slice());
					self.current_value_iter = values;
					self.advance_value();
				} else {
					self.next_value = None; // this is same as no iterator (see next_value implementation)				
				}
			} else {
				// last interval
				if let Some(previous_touched_index_depth) = self.previous_touched_index_depth.as_ref() {
					let base_depth = if previous_touched_index_depth.1 % nibble_ops::NIBBLE_PER_BYTE > 0 {
						((previous_touched_index_depth.1) / nibble_ops::NIBBLE_PER_BYTE) + 1
					} else {
						(previous_touched_index_depth.1) / nibble_ops::NIBBLE_PER_BYTE
					};
					if let Some(start) = end_prefix_index(&previous_touched_index_depth.0[..base_depth], previous_touched_index_depth.1) {
						let values = values_backend.iter_from(start.as_slice());
						self.current_value_iter = values;
						self.advance_value();
					} else {
						self.next_value = None; // this is same as no iterator (see next_value implementation)
					}
				} else {
					unreachable!("we do not stack index iter without content");
				}
			}
		}
		result.map(|(k, i)| (k, IndexOrValue::Index(i)))
	}
}

pub enum Next {
	Ascend,
	Descend,
	Value,
	Index(Option<usize>),
}

mod test {
	use super::*;

	/// A filled (up to a maximum non include size key) key value backend.
	/// Second usize is a width (255 for all keys).
	struct TestKVBackend(usize, u8);

	/// A filled (up to a maximum non include size key) key value backend.
	struct TestKVBackendIter(Vec<u8>, usize, bool, u8);

	impl KVBackend for TestKVBackend {
		fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
			if key.len() < self.0 {
				Some(vec![1u8])
			} else {
				None
			}
		}
		fn write(&mut self, _key: &[u8], _value: &[u8]) {
			unreachable!("Unsupported")
		}
		fn remove(&mut self, _key: &[u8]) {
			unreachable!("Unsupported")
		}
		fn iter<'a>(&'a self) -> KVBackendIter<'a> {
			self.iter_from(&[])
		}
		fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
			Box::new(TestKVBackendIter(start.to_vec(), self.0, false, self.1))
		}
	}

	impl Iterator for TestKVBackendIter {
		type Item = (Vec<u8>, Vec<u8>);
		fn next(&mut self) -> Option<Self::Item> {
			if self.1 == 0 {
				return None;
			}
			let key = self.0.clone();


			if self.2 {
				// going upward
				loop {
					let len = self.0.len();
					let last = self.0[len - 1];
					if last == self.3 {
						self.0.pop();
						if self.0.is_empty() {
							self.1 = 0;
							break;
						}
					} else {
						self.0[len - 1] += 1;
						self.2 = false;
						break;
					}
				}
			} else {
				// going downward
				if self.0.len() == self.1 - 1 {
					self.2 = true;
					return self.next();
				} else {
					self.0.push(0u8);
				}
			}
			Some((key, vec![1u8]))
		}
	}

	#[test]
	fn test_root_iter() {
		let width = 16;
		let mut kvbackend = TestKVBackend(4, width);
		let mut kvbackenditer = kvbackend.iter();
		let mut nb = 0;
		for (k, v) in kvbackenditer {
			nb += 1;
//			println!("{:?} at {:?}", k, ix);
		}
		let mut index_backend: BTreeMap<Vec<u8>, Index> = Default::default();
		let idepth1: usize = 3;
		let depth_index = DepthIndexes::new(&[idepth1 as u32]);
		let mut root_iter = RootIndexIterator::<_, _, Vec<u8>, _>::new(
			&kvbackend,
			&index_backend,
			&depth_index,
			std::iter::empty(),
			Default::default(),
		);
		let mut nb2 = 0;
		for (k, v) in root_iter {
			nb2 += 1;
		}
		assert_eq!(nb, nb2);
		let mut index_backend: BTreeMap<Vec<u8>, Index> = Default::default();
		let index1 = vec![0];
		let index2 = vec![5];
		index_backend.write(idepth1, index1.clone().into(), Index{ hash: Default::default(), actual_depth: 2, is_top_index: true, top_depth: 9});
		index_backend.write(idepth1, index2.clone().into(), Index{ hash: Default::default(), actual_depth: 2, is_top_index: true, top_depth: 2});
		let mut root_iter = RootIndexIterator::<_, _, Vec<u8>, _>::new(
			&kvbackend,
			&index_backend,
			&depth_index,
			std::iter::empty(),
			Default::default(),
		);
		let mut nb3 = 0;
		for (k, v) in root_iter {
			if let IndexOrValue::Index(..) = v {
			} else {
				let common_depth = nibble_ops::biggest_depth(
					&k[..],
					index1.as_slice(),
				);
				assert!(common_depth < 2);
				let common_depth = nibble_ops::biggest_depth(
					&k[..],
					index2.as_slice(),
				);
				assert!(common_depth < 2);
			}
			nb3 += 1;
		}
		assert_ne!(nb2, nb3);
		let depth_index = DepthIndexes::new(&[3, 6]);
		let mut index_backend: BTreeMap<Vec<u8>, Index> = Default::default();
		let index1 = vec![0, 0];
		let index11 = vec![0, 1, 0];
		let index12 = vec![0, 1, 5];
		index_backend.write(3, index1.clone().into(), Index{ hash: Default::default(), actual_depth: 3, is_top_index: false, top_depth: 3});
		index_backend.write(6, index11.clone().into(), Index{ hash: Default::default(), actual_depth: 6, is_top_index: true, top_depth: 9});
		index_backend.write(6, index12.clone().into(), Index{ hash: Default::default(), actual_depth: 6, is_top_index: true, top_depth: 6});
		let mut root_iter = RootIndexIterator::<_, _, Vec<u8>, _>::new(
			&kvbackend,
			&index_backend,
			&depth_index,
			std::iter::empty(),
			Default::default(),
		);
		let mut nb3 = 0;
		for (k, v) in root_iter {
			if let IndexOrValue::Index(..) = v {
			} else {
				let common_depth = nibble_ops::biggest_depth(
					&k[..],
					index1.as_slice(),
				);
				assert!(common_depth < 3);
			}
			nb3 += 1;
		}
		assert_ne!(nb2, nb3);
		let mut root_iter = RootIndexIterator::<_, _, Vec<u8>, _>::new(
			&kvbackend,
			&index_backend,
			&depth_index,
			// change to stack second layer iter
			vec![(index1.clone(), None)].into_iter(),
			Default::default(),
		);
		let mut nb3 = 0;
		let mut nb4 = 0;
		for (k, v) in root_iter {
			if let IndexOrValue::Index(..) = v {
			} else {
				let common_depth = nibble_ops::biggest_depth(
					&k[..],
					index11.as_slice(),
				);
				assert!(common_depth < 6);
				let common_depth = nibble_ops::biggest_depth(
					&k[..],
					index12.as_slice(),
				);
				assert!(common_depth < 6);
				nb3 += 1;
			}
			nb4 += 1;
		}
		assert_ne!(nb2, nb3);
		assert_eq!(nb2, nb4);
	}

	#[test]
	fn test_root_index_1() {
		let width = 16;
		let mut kvbackend = TestKVBackend(4, width);
		let mut kvbackenditer = kvbackend.iter();
		let mut nb = 0;
		let mut indexes_backend = BTreeMap::new();

		let indexes = reference_trie::DepthIndexes::new(&[1, 2, 3]);
		reference_trie::build_index(&mut indexes_backend, &indexes, kvbackenditer);
		//panic!("{:?}, {:?}", indexes_backend, indexes_backend.len())
	}
	#[test]
	fn test_root_index_2() {
		let mut kvbackenditer = vec![
			(vec![1;32], vec![0;32]),
			(vec![1;64], vec![3;32]),
		];
		let mut nb = 0;
		let mut indexes_backend = BTreeMap::new();

		let indexes = reference_trie::DepthIndexes::new(&[]);
		let root_1 = reference_trie::build_index(&mut indexes_backend, &indexes, kvbackenditer.clone().into_iter());
		let mut indexes_backend = BTreeMap::new();
		let indexes = reference_trie::DepthIndexes::new(&[65]);
		let root_2 = reference_trie::build_index(&mut indexes_backend, &indexes, kvbackenditer.into_iter());
		assert_eq!(root_1, root_2);
//		panic!("{:?}, {:?}, {:?}", indexes_backend, indexes_backend.len(), root_1 == root_2);
	}

	#[test]
	fn test_root_index_runs() {
		test_root_index(&[32], 500, 4);
		//test_root_index(&[15], 500, 60);
//		test_root_index(&[1, 2, 3, 4, 5, 15, 20], 500, 160);
		test_root_index(&[15, 25, 30], 50, 600);
//		test_root_index(&[15, 25, 30], 1, 600_000);
	}
	#[cfg(test)]
	fn test_root_index(indexes: &'static [u32], nb_iter: usize, count: u32) {
		use trie_standardmap::*;

		let mut seed: [u8; 32] = Default::default();
		for _ in 0..nb_iter {
			// TODO should move to iter_build
			let x = StandardMap {
				alphabet: Alphabet::Custom(b"@QWERTYUIOPASDFGHJKLZXCVBNM[/]^_".to_vec()),
				min_key: 5,
				journal_key: 32 - 5,
				value_mode: ValueMode::Index,
				count,
			}.make_with(&mut seed);

			use memory_db::{MemoryDB, HashKey, PrefixedKey};
			use keccak_hasher::KeccakHasher;

			let indexes_conf = reference_trie::DepthIndexes::new(indexes);
			let memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, Vec<u8>>::default();
			let mut indexes = std::collections::BTreeMap::new();
			let change = Vec::new();
			let data: BTreeMap<_, _> = x.into_iter().collect();
			let data: Vec<_> = data.into_iter().collect();
			reference_trie::compare_index_calc(data, change, memdb, &mut indexes, &indexes_conf, None);
		}
	}
}
