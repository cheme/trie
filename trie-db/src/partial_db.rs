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
// TODO consider trying to use small vec (need to be usable in range) for result, we can even use
// nibblevec internally
fn value_prefix_index(actual_index_depth: usize, mut change_key: Vec<u8>) -> (Vec<u8>, Option<Vec<u8>>) {
	// TODO change key is always odd, some code here is useless.
	// TODO consider not returning start (especially since it can allocates).
	let start = actual_index_depth;
	let odd = start % nibble_ops::NIBBLE_PER_BYTE;
	let start_byte = start / nibble_ops::NIBBLE_PER_BYTE + if odd > 0 { 1 } else { 0 };

	change_key.resize(start_byte, 0);

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
		// TODO EMCH can trim the position to actual size of index (just gain size storage).
		if odd != 0 {
			position.last_mut().map(|l| 
				*l = *l & !(255 >> (odd * nibble_ops::BIT_PER_NIBBLE))
			);
		}
		self.insert(index_tree_key_owned(depth, position).to_vec(), index);
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
	/// Dropped value node, from change set.
	DroppedValue,
	/// Value node value, from existing values.
	StoredValue(Vec<u8>),
}

impl<B> IndexOrValue<B> {
	/// Access the index branch depth (including the partial key).
	pub fn index_depth(&self) -> Option<usize> {
		match self {
			IndexOrValue::Index(index, _change_value) => Some(index.actual_depth),
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
	/// value iterator in use and the range that needs to be covered.
	/// A boolean record first touched value or change, at end if no touch
	/// value or change, a delete child is returned.
	/// TODO the end range is currently unnecessary as the next index would
	/// also end it.
	current_value_iter: Option<(KVBackendIter<'a>, (Vec<u8>, Option<Vec<u8>>))>,
	index_iter: Vec<StackedIndex<'a>>,
	next_change: Option<(Vec<u8>, Option<V>)>,
	next_value: Option<(Vec<u8>, Vec<u8>)>,
	// key and depth of previous index touch, get reset each time we change
	// prefix (return index or pop)
	previous_touched_index_depth: Option<(Vec<u8>, usize)>,

	sub_iterator: Option<SubIterator<'a, V>>,
}

struct SubIterator<'a, V> {
	base: NibbleVec,
	index_iter: Option<StackedIndex<'a>>,
	current_value_iter: Option<(KVBackendIter<'a>, (Vec<u8>, Option<Vec<u8>>))>,
	buffed_next_value: (Vec<u8>, IndexOrValue<V>),
	next_value: Option<(Vec<u8>, Vec<u8>)>,
	previous_touched_index_depth: Option<(Vec<u8>, usize)>,
}

struct StackedIndex<'a> {
	iter: IndexBackendIter<'a>, 
	range: (Vec<u8>, Option<Vec<u8>>),
	next_index: Option<(Vec<u8>, Index)>,
	conf_index_depth: usize,
	end_saved_value_iter: Option<(Vec<u8>, Option<Vec<u8>>)>,
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

/*		for a in changes_iter {
			println!("{:?}", (a.0, a.1.as_ref().map(|v| v.as_ref())));
		}
		panic!("dd");*/
		let mut iter = RootIndexIterator {
			values,
			indexes,
			indexes_conf,
			changes_iter,
			deleted_indexes,
			current_index_depth: 0,
			current_value_iter: None,
			index_iter: Vec::new(),
			next_change: None,
			next_value: None,
			previous_touched_index_depth: None,
			sub_iterator: None,
		};

		// get first change
		iter.advance_change();
		// 0 depth index is always defined.
		let mut root_iter = iter.indexes.iter(0, &[]);
		let first = root_iter.next();
		if first.is_some() {
			iter.index_iter.push(StackedIndex {
				iter: root_iter,
				range: (Vec::new(), None),
				next_index: first,
				conf_index_depth: 0,
				end_saved_value_iter: None,
			});
		}
	
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
		if let Some(sub_iter) = self.sub_iterator.as_mut() {
			unimplemented!("subiter iter");
		}
	/*	match self.state {
			State::ValueReachingTarget => self.next_change_or_value(),
			State::ValueReachingIndex => self.next_value_or_index(),
			State::ValueToEnd => self.next_value(),
			State::IndexReachingTarget => self.next_change_or_index(),
			State::IndexToEnd => self.next_index(None),
			State::Ended => None,
		}*/
		let next_element = self.next_element();
		let stacked = match next_element {
			Element::Value => false,
			Element::None => false,
			Element::IndexChange
			| Element::Change
			| Element::ChangeValue => if let Some(kv) = self.next_change.as_ref().map(|kv| &kv.0) {
				if let Some(i) = self.try_stack_index(&kv, kv.len() * nibble_ops::NIBBLE_PER_BYTE) { // TODO some internal do not make sense for change eg common path is always change length
					self.do_stack_index(i)
				} else {
					false
				}
			} else {
				false
			},
			Element::Index  => if let Some((kv, depth)) = self.buffed_next_index().map(|kv| (kv.0.clone(), kv.1.actual_depth)) {
				if let Some(i) = self.try_stack_index(&kv, depth) {
					self.do_stack_index(i)
				} else {
					false
				}
			} else {
				false
			},
		};
		if stacked {
			return self.next();
		}
		match next_element {
			Element::Value => self.next_value(),
			Element::Index => {
				let r = self.next_index(None);
				if r.is_none() {
					if self.pop_index() {
						return self.next();
					}
				}
				r
			},
			Element::IndexChange => {
				let next_change = self.next_change.take();
				self.advance_change();
				if let Some((_key, change)) = next_change {
					let r = self.next_index(Some(change));
					if r.is_none() {
						if self.pop_index() {
							return self.next();
						}
					}
					r
				} else {
					// skip delete
					self.next()
				}
			},
			Element::Change => if self.try_new_value_iter() {
				self.next()
			} else {
				self.next_change()
			},
			Element::ChangeValue => {
				self.advance_value();
				self.next_change()
			},
			Element::None => None,
		}
	}
/*
// TODO EMCH we stack iter to always try to have an iter above us
// we here get next element by comparing three (on start state must always be good
// We move index pop in advance index
// we match result and do change in this function accordingly.
// - on index, try advance index -> does pop internaly, then new index so clear value iter and init
// a new
// - on change, try advance change -> on fail nothing, on success try stack index and if advance
// and if init value iter
// - on value, try advance -> on fail clear and do nothing on success do nothing.

		// in fact we stack over change and vaul (all the stae pof new function with different starts orders.
		// -> probably lot of redundancy in compare but first get something working

		// value iteration is well define in interval between indexes and
		// only need to be compared with change.
		if self.current_value_iter.is_some() {
			if self.next_value.is_none() {
				self.advance_value();
			}
			if self.next_change.is_none() {
				self.advance_change();
			}
			let result = self.next_change_or_value();
			if result.is_some() {
				return result;
			}
		}

		// change init

		// index and value resolve.
		None
	}
*/
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
		buffed: (Vec<u8>, IndexOrValue<V>),
	) {
		let mut base = NibbleVec::new();
		base.append_partial(((0, 0), key));
		let len = base.len();
		let to_drop = len - depth;
		base.drop_lasts(to_drop);
		base.push(child_index as u8);
		let key = base.inner();
		let indexes = &mut self.indexes;
		// get index iterator
		let index_iter = self.indexes_conf.next_depth(depth + 1, key)
			.map(move |d| {
				let iter = indexes.iter(d, key);
				// TODO try avoid this alloc
				let range = value_prefix_index(d, key.to_vec());
				StackedIndex {
					iter,
					range,
					next_index: None,
					conf_index_depth: d,
					end_saved_value_iter: None,
				}
			});
		let current_value_iter = {
			let values = self.values.iter_from(&key[..]);
			let end = end_prefix(&key[..]);
			Some((values, (key.to_vec(), end)))
		};

		self.sub_iterator = Some(SubIterator {
			base,
			index_iter,
			current_value_iter,
			buffed_next_value: buffed,
			next_value: None,
			previous_touched_index_depth: None,
		});
		self.sub_advance_index();
		self.sub_advance_value();
	}
}

impl<'a, KB, IB, V, ID> RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	fn sub_next_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		let mut result = None;
		let mut common_depth = None;
		if let Some(sub_iter) = self.sub_iterator.as_mut() {
			if let Some(index_iter) = sub_iter.index_iter.as_mut() {
				if let Some(index) = index_iter.next_index.take() {
					common_depth = self.previous_touched_index_depth.as_ref().map(|last_index| {
						let common_depth =  nibble_ops::biggest_depth(
							&last_index.0[..],
							&index.0[..],
						);
						let common_depth = crate::rstd::cmp::min(common_depth, index.1.actual_depth);
						let common_depth = crate::rstd::cmp::min(common_depth, last_index.1);
						common_depth
					});
					sub_iter.previous_touched_index_depth = Some((index.0.clone(), (index.1).actual_depth));
					result = Some(index);
				} else {
					sub_iter.previous_touched_index_depth = None;
				}
			}
		}

		if result.is_none() {
			// last interval
			if let Some(previous_touched_index_depth) = self.previous_touched_index_depth.take() {
				if let Some(sub_iter) = self.sub_iterator.as_mut() {
					// TODO can factor with next result.map
					let end_value = if let Some(value_iter) = self.current_value_iter.as_mut() {
						(value_iter.1).1.take()
					} else {
						end_prefix(sub_iter.base.inner())
					};

					let base_depth = (previous_touched_index_depth.1 - 1 + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE;
					let values = self.values.iter_from(&previous_touched_index_depth.0[..base_depth]);
					sub_iter.current_value_iter = Some((values, (previous_touched_index_depth.0[..base_depth].to_vec(), end_value)));
				}
				self.sub_advance_value();
			}
		}
		result.map(|(k, i)| {
			if let Some(sub_iter) = self.sub_iterator.as_mut() {
				let end_value = if let Some(value_iter) = self.current_value_iter.as_mut() {
					(value_iter.1).1.take()
				} else {
					end_prefix(sub_iter.base.inner())
				};
				let base_depth = if let Some(common_depth) = common_depth {
					(common_depth + 1 + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE
				} else {
					unreachable!();
				};
				let values = self.values.iter_from(&k[..base_depth]);

				sub_iter.current_value_iter = Some((values, (k[..base_depth].to_vec(), end_value)));
			}

			self.sub_advance_value();
			self.sub_advance_index();
			(k, IndexOrValue::Index(i, None))
		})
	}

	fn sub_advance_index(
		&mut self,
	) {
		if let Some(sub_iter) = self.sub_iterator.as_mut() {
			if let Some(index_iter) = sub_iter.index_iter.as_mut() {
				index_iter.next_index = index_iter.iter.next().filter(|kv| {
					index_iter.range.1.as_ref().map(|end| &kv.0 < end).unwrap_or(true)
				});
			}
		}
	}

	fn sub_advance_value(
		&mut self,
	) {
		if let Some(sub_iter) = self.sub_iterator.as_mut() {
			if let Some(iter) = sub_iter.current_value_iter.as_mut() {
				sub_iter.next_value = iter.0.next()
					.filter(|kv| (iter.1).1.as_ref().map(|end| &kv.0 < end)
						.unwrap_or(true));
			}
		}
	}

	// Return true if we stacked something and the stack item is not over current item.
	fn try_stack_index(&self, current_key_vec: &Vec<u8>, current_depth: usize) -> Option<usize> {
		// see if next change is bellow
		if let Some((next_k, _)) = self.next_change.as_ref() {
			let common_depth = nibble_ops::biggest_depth(
				&current_key_vec[..],
				&next_k[..],
			);
			if common_depth < current_depth {
				return None;
			}
			// the index is contain in the change
			Some(current_depth + 1)
		} else {
			None
		}
	}

	fn do_stack_index(&mut self, first_possible_next_index: usize) -> bool {
		if let Some((next_change_key, _)) = self.next_change.as_ref() {
			let index_iter = &mut self.index_iter;
			let current_value_iter = &mut self.current_value_iter;
			let indexes = &self.indexes;
//			let change_depth = next_change_key.len() * nibble_ops::NIBBLE_PER_BYTE;
			self.indexes_conf.next_depth(first_possible_next_index, next_change_key)
				.map(move |d| {
					let mut iter = indexes.iter(d, next_change_key);
					// TODO try avoid this alloc
					let range = value_prefix_index(d, next_change_key.to_vec());
					let first = iter.next().filter(|kv| {
						range.1.as_ref().map(|end| &kv.0 < end).unwrap_or(true)
					});
		
					if first.is_some() {
						let end_saved_value_iter = current_value_iter.take()
							.and_then(|iter| {
								end_prefix(&next_change_key[..(d + 1) / nibble_ops::NIBBLE_PER_BYTE])
									.map(|start| (start, (iter.1).1))
							});
						index_iter.push(StackedIndex {
							iter,
							range,
							next_index: first,
							conf_index_depth: d,
							end_saved_value_iter,
						});
						true
					} else {
						false
					}
				}).unwrap_or(false)
		} else {
			false
		}
	}

	fn advance_index(&mut self) -> bool {
		self.index_iter.last_mut().map(|i| {
			i.next_index = i.iter.next()
				.filter(|kv| {
					i.range.1.as_ref().map(|end| &kv.0 < end)
						.unwrap_or(true)
				});
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
			if let Some((start, end)) = index.end_saved_value_iter {
				let values = self.values.iter_from(&start[..]);
				let range = (start, end);
				assert!(self.current_value_iter.is_none());
				self.current_value_iter = Some((values, range));
				self.advance_value();
				// TODO here we will totally consume value before next pop
				// -> could use a specific state.
				if self.next_value.is_some() {
					self.advance_index();
					return true;
				}
			}
			if !self.advance_index() {
				self.pop_index()
			} else {
				true
			}
		} else {
			false
		}
	}

	fn is_value_before_index(&self) -> bool {
		match (self.buffed_next_index(), &self.next_value) {
			(Some(next_index), Some(next_value)) => {
				match next_index.1.compare(&next_index.0, &next_value.0) {
					Ordering::Equal => unreachable!("Value are not indexed"),
					Ordering::Less => false,
					Ordering::Greater => true,
				}
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
	
	fn next_element(&self) -> Element {
		if self.is_value_before_index() {
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

	fn next_index(&mut self, change: Option<Option<V>>) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		self.previous_touched_index_depth = None;
		let current_index_depth = &mut self.current_index_depth;
		let previous_touched_index_depth = &mut self.previous_touched_index_depth;
		// stop value iteration after index
		self.next_value = None;
		self.current_value_iter = None;
		let r = self.index_iter.last_mut().and_then(|i|
			i.next_index.take().map(|index| {
				*current_index_depth = (index.1).actual_depth;
				*previous_touched_index_depth = Some((index.0.clone(), (index.1).actual_depth));
				(index.0, IndexOrValue::Index(index.1, change))
			})
		);
		// TODO no need for read ahead?
		self.advance_index();
		r
	}

	// attempt to init a new value iter (at start, on new index against next change).
	fn try_new_value_iter(&mut self) -> bool {
		if self.current_value_iter.is_some() {
			return false;
		}
		match (self.previous_touched_index_depth.as_ref(), &self.next_change) {
			(Some(previous_depth), Some(next_change)) => {
				let odd = previous_depth.1 % nibble_ops::NIBBLE_PER_BYTE;
				let ref_depth = previous_depth.1 / nibble_ops::NIBBLE_PER_BYTE; // TODO could we include an out of range in odd as first value (then advance twice)
				let mut start = next_change.0[..ref_depth].to_vec(); // TODO avoid clone by advance value once more on smaller first key and have end_prefix_odd variant
				if odd > 0 {
					start.push(0);
				}
				let values = self.values.iter_from(&start[..]);
				let end = end_prefix(&start[..]);
				// TODO we shall remove start
				let range = (start, end);
				self.current_value_iter = Some((values, range));
				self.advance_value();
				true
			},
			(None, Some(_next_change)) => {
				let values = self.values.iter_from(&[]);
				let range = (Vec::new(), None);
				self.current_value_iter = Some((values, range));
				self.advance_value();
				true
			},
			(Some(_), None)
			| (None, None) => false,
		}
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
		let next_value = &mut self.next_value;
		self.current_value_iter.as_mut().map(|iter| {
			*next_value = iter.0.next()
				.filter(|kv| (iter.1).1.as_ref().map(|end| &kv.0 < end)
					.unwrap_or(true));
		});
	}
}

pub enum Next {
	Ascend,
	Descend,
	Value,
	Index(Option<usize>),
}
