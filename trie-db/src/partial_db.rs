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
use crate::rstd::vec::Vec;
use crate::rstd::iter::Peekable;
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
	/// Iterate over the values. TODO remove: iter_from is enough.
	fn iter<'a>(&'a self) -> KVBackendIter<'a>;
	// TODO could also optionally restore from a previous iterator start (complicate types
	// a lot), would allow 'skipping' implementation: eg for a radix trie: do not
	// get from root but from last iter state which is closer in number of node than root
	// (may not be true: would need to be checked over depths).
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
pub type IndexBackendIter<'a> = Box<dyn Iterator<Item = (NibbleVec, Index)> + 'a>;

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

// TODO better name (this one sound like an index when it is a key).
pub type IndexPosition = crate::nibble::BackingByteVec;

/// Storage of encoded index for a trie.
/// Depth is a nibble depth, it is the indexing depth. So the actual indexing
/// for a index group is the part over depth from `IndexPosition`.
pub trait IndexBackend {
	/// Query for an index, also return the depth of this index (it can differs from
	/// the index depth when there is a nibble encoded in the node).
	fn read(&self, depth: usize, index: LeftNibbleSlice) -> Option<Index>;
	/// Insert an `encode_index` with and `actual_depth` at configured `depth` for a given `index`.
	fn write(&mut self, depth: usize, index: LeftNibbleSlice, value: Index);
	/// Remove any value at a key.
	fn remove(&mut self, depth: usize, index: LeftNibbleSlice);
	/// Iterate over the index from a key.
	/// `group` is common key at depth.
	fn iter<'a>(&'a self, depth: usize, parent_depth: usize, group: LeftNibbleSlice) -> IndexBackendIter<'a>;
	/// Same as iterate but starting at a given index: in this case the `index` parameter
	/// contains both the depth and the starting index.
	/// TODO could be a single iter function only (if start_index len > depth it is from variant).
	fn iter_from<'a>(&'a self, depth: usize, parent_depth: usize, start_index: LeftNibbleSlice) -> IndexBackendIter<'a>;
}

impl IndexBackend for Arc<dyn IndexBackend + Send + Sync> {
	fn read(&self, depth: usize, index: LeftNibbleSlice) -> Option<Index> {
		IndexBackend::read(self.as_ref(), depth, index)
	}
	fn write(&mut self, depth: usize, index: LeftNibbleSlice, value: Index) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn remove(&mut self, depth: usize, index: LeftNibbleSlice) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn iter<'a>(&'a self, depth: usize, parent_depth: usize, group: LeftNibbleSlice) -> IndexBackendIter<'a> {
		IndexBackend::iter(self.as_ref(), depth, parent_depth, group)
	}
	fn iter_from<'a>(&'a self, depth: usize, parent_depth: usize, start_index: LeftNibbleSlice) -> IndexBackendIter<'a> {
		IndexBackend::iter_from(self.as_ref(), depth, parent_depth, start_index)
	}
}

impl IndexBackend for Box<dyn IndexBackend + Send + Sync> {
	fn read(&self, depth: usize, index: LeftNibbleSlice) -> Option<Index> {
		IndexBackend::read(self.as_ref(), depth, index)
	}
	fn write(&mut self, depth: usize, index: LeftNibbleSlice, value: Index) {
		IndexBackend::write(self.as_mut(), depth, index, value)
	}
	fn remove(&mut self, depth: usize, index: LeftNibbleSlice) {
		IndexBackend::remove(self.as_mut(), depth, index)
	}
	fn iter<'a>(&'a self, depth: usize, parent_depth: usize, group: LeftNibbleSlice) -> IndexBackendIter<'a> {
		IndexBackend::iter(self.as_ref(), depth, parent_depth, group)
	}
	fn iter_from<'a>(&'a self, depth: usize, parent_depth: usize, start_index: LeftNibbleSlice) -> IndexBackendIter<'a> {
		IndexBackend::iter_from(self.as_ref(), depth, parent_depth, start_index)
	}
}

#[derive(Default)]
#[derive(Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
/// Content of an index.
/// Index is a hash, and its actual depth is parent branch depth + 1 or
/// 0 if root.
/// actual depth is <= index depth. (< when there is a partial nible overlapping
/// the index position).
pub struct Index {
	pub hash: Vec<u8>, // TODO hash as inner type ?? (no need to encode vec length here)
	/// Indicate if index matches the indexing depth or is bellow.
	pub on_index: bool,
}

impl Index {
	// TODO check if used or can be replaced
	fn compare(&self, index_key: &[u8], change_key: &[u8]) -> Ordering {
		unimplemented!()
/*		let end = self.actual_depth;
		let odd = end % nibble_ops::NIBBLE_PER_BYTE;
		let end = end / nibble_ops::NIBBLE_PER_BYTE;
		if odd == 0 {
			index_key[..end].cmp(change_key)
		} else {
			let size = self.actual_depth;
			let slice1 = LeftNibbleSlice::new(index_key).truncate(size);
			let slice2 = LeftNibbleSlice::new(change_key);
			slice1.cmp(&slice2)
		}*/
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

/// Key for an index stored in a tree.
/// Depth is use as a prefix to isolate iterators.
pub fn index_tree_key(depth: usize, index: &LeftNibbleSlice) -> IndexPosition {
	// writing directly the bytes with 0 as padding as we cannot have a child (would be
	// at a different indexing depth).
	let owned: NibbleVec = index.into();
	index_tree_key_owned(depth, owned)
}

/// Key for an index stored in a tree.
/// Depth is use as a prefix to isolate iterators.
pub fn index_tree_key_owned(depth: usize, index: NibbleVec) -> IndexPosition {
	let mut result = index.padded_buffer();
	result.insert_from_slice(0, &(depth as u32).to_be_bytes()[..]);
	result
}

/// Note that this is a test implementation, sparse matrix should be use here.
impl IndexBackend for BTreeMap<IndexPosition, Index> {
	fn read(&self, depth: usize, index: LeftNibbleSlice) -> Option<Index> {
	//fn read(&self, depth: usize, index: &[u8]) -> Option<Index> {
		// api expect index so if key longer, need truncate first
		debug_assert!(depth <= index.len());
		self.get(&index_tree_key(depth, &index.truncate(depth))[..]).cloned()
	}
	fn write(&mut self, depth: usize, index: LeftNibbleSlice, value: Index) {
		debug_assert!(depth <= index.len());
		// TODO audit if index should be owned???
	//fn write(&mut self, depth: usize, mut position: IndexPosition, index: Index) {
		self.insert(index_tree_key(depth, &index.truncate(depth)), value);
	}
	fn remove(&mut self, depth: usize, index: LeftNibbleSlice) {
		// TODO audit if owned index possible
		debug_assert!(depth <= index.len());
	//fn remove(&mut self, depth: usize, index: IndexPosition) {
		self.remove(&index_tree_key(depth, &index.truncate(depth))[..]);
	}
	fn iter<'a>(&'a self, depth: usize, parent_depth: usize, group: LeftNibbleSlice) -> IndexBackendIter<'a> {
		debug_assert!(parent_depth <= group.len());
	//fn iter<'a>(&'a self, depth: usize, depth_base: usize, change: &[u8]) -> IndexBackendIter<'a> {
		// TODO consider util function for it as this code will be duplicated in any ordered db.
		self.iter_from(depth, parent_depth, group.truncate(parent_depth))
	}
	fn iter_from<'a>(&'a self, depth: usize, parent_depth: usize, start_index: LeftNibbleSlice) -> IndexBackendIter<'a> {
		let l_size = crate::rstd::mem::size_of::<u32>();
		let start = index_tree_key(depth, &start_index);
		let mut end = index_tree_key(depth, &start_index.truncate(parent_depth));
		while end.last() == Some(&u8::max_value()) {
			end.pop();
		}
		end.last_mut().map(|v| *v += 1);
		
		let range = self.range(start..end);
		Box::new(range.into_iter().map(move |(k, ix)| {
			let k = LeftNibbleSlice::new_len(&k[l_size..], depth);
			((&k).into(), ix.clone())
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
#[derive(Clone)]
pub struct DepthIndexes(smallvec::SmallVec<[u32; 16]>);

impl crate::rstd::ops::Deref for DepthIndexes {
	type Target = smallvec::SmallVec<[u32; 16]>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

struct IndexBranch {
	depth: usize,
	has_value: bool,
	// If a branch was untouched/undeleted.
	processed_branch: bool,
	// Is valid when we know of:
	// - two untouched/undeleted branch index
	// - a value and a untouch/undeleted branch index
	// Undeleted check can be done over value and change
	// iterator without additional read ahead (deleted
	// value is not part of the iterator.
	valid: bool,
}

/// Branches used when iterating.
/// TODO The latest could be part of the iteration to avoid redundant key
/// comparison in `iter_build`.
struct StackBranches(smallvec::SmallVec<[IndexBranch; 10]>);

impl crate::rstd::ops::Deref for StackBranches {
	type Target = smallvec::SmallVec<[IndexBranch; 10]>;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl crate::rstd::ops::DerefMut for StackBranches {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}


impl StackBranches {
	fn current_branch_depth(&self) -> Option<usize> {
		self.0.last().map(|i| i.depth)
	}
}

impl DepthIndexes {
	/// Instantiate a new depth indexes configuration.
	pub fn new(indexes: &[u32]) -> Self {
		DepthIndexes(indexes.into())
	}

	// TODO rename this or other next_depth + maybe not pub.
	pub fn next_depth2(&self, current_depth: Option<usize>) -> Option<usize> {
		if let Some(current) = current_depth {
			for i in self.0.iter() {
				let i = *i as usize;
				if i > current {
					return Some(i);
				}
			}
		} else {
			return self.0.get(0).map(|i| *i as usize);
		}
		None
	}

	/// Returns next_index depth to register, starting at a given included depth.
	///
	/// We use an index position to be able to define indexing only for given prefixes
	/// (needed only for unbalanced trie).
	/// TODO this is not really efficient and use on every node
	/// TODO put in IdenxesConf trait.
	pub fn next_depth(&self, prefix_len: usize, nodekey_len: usize) -> Option<usize> {
		// No index 0 as they are ignored. TODO remove this??
/*		let depth = if position.len() == 0 {
			1
		} else{
			position.len()
		};*/
		for i in self.0.iter() {
			let i = *i as usize;
			if i >= prefix_len  {
				if i <= nodekey_len {
					return Some(i)
				} else {
					return None;
				}
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

/// Enum containing either a value or an index, mostly for internal use
/// (see `RootIndexIterator` and `iter_build::trie_visit_with_indexes`).
#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone)]
pub enum IndexOrValue2<B> {
	/// Contains depth as number of bit and the encoded value of the node for this index.
	/// Also an optional `Change of value is attached`.
	Index(Index),
	/// Value node value, from change set.
	Value(B),
	/// Value node value, from existing values.
	StoredValue(Vec<u8>),
}

impl<B> IndexOrValue<B> {
	/// Access the index branch depth (including the partial key).
	pub fn exact_index(&self) -> Option<bool> {
		match self {
			IndexOrValue::Index(index) => Some(index.on_index),
			_ => None,
		}
	}
}

impl<B> IndexOrValue2<B> {
	/// Access the index branch depth (including the partial key).
	pub fn exact_index(&self) -> Option<bool> {
		match self {
			IndexOrValue2::Index(index) => Some(index.on_index),
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

/// Iterator over index and value for root calculation
/// and index update.
/// TODO remove V as type param?? (already Vec<u8> in
/// StoredValue), or put V in kvbackend.
pub struct RootIndexIterator2<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	values: &'a KB,
	indexes: &'a IB,
	indexes_conf: &'a DepthIndexes,
	changes_iter: Peekable<ID>,
	index_iter: Vec<StackedIndex2<'a>>,
	current_value_iter: Option<Peekable<KVBackendIter<'a>>>,
	stack_branches: StackBranches,
	state: Next2,
	next_item: Option<(NibbleVec, IndexOrValue2<V>)>,
	// TODO make deleted optional and directly delete
	// in backend if no container
	deleted_indexes: &'a mut Vec<(usize, NibbleVec)>,
	deleted_values: &'a mut Vec<Vec<u8>>,
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
	next_index: Option<(NibbleVec, Index)>,
	conf_index_depth: usize,
}

struct StackedIndex2<'a> {
	iter: Peekable<IndexBackendIter<'a>>,
	conf_index_depth: usize,
}

impl<'a, KB, IB, V, ID> RootIndexIterator2<'a, KB, IB, V, ID>
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
		deleted_indexes: &'a mut Vec<(usize, NibbleVec)>,
		deleted_values: &'a mut Vec<Vec<u8>>,
	) -> Self {
		let current_value_iter = Some(Iterator::peekable(values.iter_from(&[])));
		let mut iter = RootIndexIterator2 {
			values,
			indexes,
			indexes_conf,
			changes_iter: Iterator::peekable(changes_iter),
			current_value_iter,
			index_iter: Vec::new(),
			stack_branches: StackBranches(Default::default()),
			state: Next2::None,
			next_item: None,
			deleted_indexes,
			deleted_values,
		};

		// always stack first level of indexes.
		iter.init_stack_index();

		assert!(iter.feed_next_item().is_none());
	
		iter
	}
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
		unimplemented!();
/*		let mut last_sub = false;
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
			Element::Index  => unimplemented!("TODO reimpl")/*if let Some((kv, depth)) = self.buffed_next_index().map(|kv| (kv.0.clone(), kv.1.actual_depth)) {
				if self.try_stack_on_index(&kv, depth) {
					self.stack_index(); // TODO if false continue with returning index
					return self.next();
				} else {
					false
				}
			} else {
				false
			}*/,
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
		}*/
	}
}

impl<'a, KB, IB, V, ID> Iterator for RootIndexIterator2<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	type Item = (NibbleVec, IndexOrValue2<V>);

	fn next(&mut self) -> Option<Self::Item> {
		if self.next_item.is_some() {
			self.feed_next_item()
		} else {
			None
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
		unimplemented!("TODO remove");
/*		let mut base = NibbleVec::new();
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
		self.sub_iterator = Some(sub_iter);*/
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
		unimplemented!();
		/*
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
		*/
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
				unimplemented!("TODO we just need to compare over current index depth");
				//let common_depth = crate::rstd::cmp::min(common_depth, next_index.1.actual_depth);
				//common_depth > next_index.1.actual_depth
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
		unimplemented!()
			//self.index_iter.last().and_then(|item| item.next_index.as_ref())
	}

	// TODO factor with next_sub_index
	fn next_index(&mut self) -> Option<(Vec<u8>, IndexOrValue<V>)> {
		unimplemented!("TODO");
/*		//self.previous_touched_index_depth = None;
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
					// TODO here out of range occurs for base_depth during synch
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
i*/
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
		unimplemented!()
/*		let end_iter = &self.end_iter;
		if let Some(index_iter) = self.index_iter.as_mut() {
			index_iter.next_index = index_iter.iter.next()
				.filter(|kv| end_iter.as_ref().map(|end| {
					&kv.0 < end
				}).unwrap_or(true));
		}*/
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
		unimplemented!("TODO");
/*		// TODO this is a copy of root 'next_index' function, could factor later.
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
*/
	}
}

// TODO rem pub
pub enum Next {
	Ascend,
	Descend,
	Value,
	Index(Option<usize>),
}

// TODO probaly only need StackNextIndex.
enum Next2 {
	None,
	// Iter index up to next branch with change (then go down index or change).
	IterToBranch,
	IterToChange,
	// Iter until index, then stack child index
	IterToStackIndex,
	// next is index.
	NextIndex,
}

impl<'a, KB, IB, V, ID> RootIndexIterator2<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{

	/// calculate and store next item, return previous
	/// value for next item (except on first call, returns
	/// `None` when iteration is finished).
	fn feed_next_item(&mut self) -> Option<(NibbleVec, IndexOrValue2<V>)> {
		let mut new_next = self.get_next_item();
		// TODO invalidate Index and sub iterate its value when:
		// TODO add in next_item wether a Value did overwrite and existing StoredValue
		// - check this index partial is not splitted:
		//	- previous (last next_item) is a not overwrite StoredValue: assert common less than common ix - next ix,
		//		feed next then subiterate
		// - check deleted parent branch
		//	- check previous is Value or Stored value that overwrite existing one, then not deleted
		//	- check ix with previous less than ix with next index: then we are on a new branch and
		//	should look ahead all deleted and value, up to next peekable then if common with next
		//	peekable is less than branch common: branch is deleted -> feed next and subiterate.
		if let Some((key_index1, new_next)) = new_next.as_ref() {
			if let Some((key_index2, _)) = self.next_item.as_ref() {
				let common_depth = key_index1.as_slice().common_length(&key_index2.as_slice());
				let mut do_stack = true;
				loop {
					if let Some(mut prev_branch) = self.stack_branches.last_mut() {
						if common_depth == prev_branch.depth {
							if prev_branch.processed_branch || prev_branch.has_value {
								prev_branch.valid = true;
							}
							prev_branch.processed_branch = true;
							do_stack = false;
							break;
						}
						if common_depth > prev_branch.depth {
							// stack over
							break;
						}
					} else {
						break;
					}
					let _ = self.stack_branches.pop();
				}
				if do_stack {
					self.stack_branches.push(IndexBranch {
						depth: common_depth,
						has_value: false, // TODO from previous values
						processed_branch: false,
						valid: false,
					});
				}
			}/* else {
				// unstack index
				self.index_iter.pop();
			}*/

			if matches!(new_next, IndexOrValue2::Index(_)) {
				let sibling_depth = if let Some(next_index) = self.index_iter.last_mut().and_then(|i| i.iter.peek()) {
					let common_depth = key_index1.as_slice().common_length(&next_index.0.as_slice());
					common_depth + 1
				} else {
					// just skip after index: this could be default (not the pop), TODO check perf as default

					self.index_iter.pop().map(|i| i.conf_index_depth)
//					self.index_iter.get(self.index_iter.len() - 1)
						.expect("Get a value from an iterator")
					// TODO rem iter here or wait for a value that is bellow index??
/* from parent ix: ko					let previous_index_depth = if self.index_iter.len() > 1 {
						self.index_iter.get(self.index_iter.len() - 2).map(|i| i.conf_index_depth)
					} else {
						None
					};
					previous_index_depth.unwrap_or(0) + 1*/
				};
				let mut key_index: NibbleVec = (&key_index1.as_slice().truncate(sibling_depth)).into();
				if key_index.next_sibling() {
					self.current_value_iter = Some(Iterator::peekable(self.values.iter_from(key_index.padded_buffer().as_slice())));
				} else {
					self.current_value_iter = None;
				}
			}
		}
	//	unimplemented!("branch update");
	//					(Ordering::Less, _commons, _is_prefix) => {
		crate::rstd::mem::replace(&mut self.next_item, new_next)
	}

	fn get_next_item(&mut self) -> Option<(NibbleVec, IndexOrValue2<V>)> {
		let cmp_change_index = if let Some(next_change) = self.changes_iter.peek() {
			let key_change = next_change.0.as_ref();
			if let Some(next_index) = self.index_iter.last_mut().and_then(|i| i.iter.peek()) {
				Some(LeftNibbleSlice::new(key_change).cmp_common_and_starts_with(&(&next_index.0).into()))
			} else {
				None
			}
		} else {
			None
		};
		match cmp_change_index {
			None => {
				if self.changes_iter.peek().is_some() {
					// change and value only
					return self.change_or_value(None);
				} else if self.index_iter.last_mut().and_then(|i| i.iter.peek()).is_some() {
					// index and value only
					return self.index_or_value();
				} else {
					// only value iterator in scope, just iterate it
					return self.current_value_iter.as_mut().and_then(|iter| iter.next()).map(|kv| (
						// TODO have NibbleVec without copy (at least from backing))
						(&LeftNibbleSlice::new(kv.0.as_slice())).into(),
						IndexOrValue2::StoredValue(kv.1),
					));
				}
			},
			Some((Ordering::Less, _commons, _is_prefix)) => {
				// change less than index 
				return self.change_or_value(None);
			},
			Some((_, _commons, false)) => {
				// change more than index: return index
				return self.index_or_value();
			},
			Some((_, _commons, true)) => {
				// change bellow or equal index
				// index skip
				let _ = self.index_iter.last_mut().and_then(|i| i.iter.next());
				// Stack index
				self.stack_index();
				// recurse
				return self.get_next_item();
			},
		}
	}

	fn change_or_value(&mut self, limit: Option<&LeftNibbleSlice>) -> Option<(NibbleVec, IndexOrValue2<V>)> {
		let mut do_change = false;
		let mut do_value = false;
		loop {
			if let Some(next_value) = self.current_value_iter.as_mut().and_then(|iter| iter.peek()) {
				if let Some(next_change) = self.changes_iter.peek() {
					match next_change.0.as_slice().cmp(next_value.0.as_ref()) {
						Ordering::Less => {
							// could be <= but we already check against index and this bound is usually index.
							if limit.map(|limit| limit < &LeftNibbleSlice::new(next_change.0.as_slice()))
								.unwrap_or(false) {
								return None;
							}
							if next_change.1.is_some() {
								do_change = true;
								break;
							} else {
								// delete nothing.
								let _ = self.changes_iter.next();
							}
						},
						Ordering::Greater => {
							// could be <= but we already check against index and this bound is usually index.
							if limit.map(|limit| limit < &LeftNibbleSlice::new(next_value.0.as_slice()))
								.unwrap_or(false) {
								return None;
							}
							do_value = true;
							break;
						},
						Ordering::Equal => {
							// could be <= but we already check against index and this bound is usually index.
							if limit.map(|limit| limit < &LeftNibbleSlice::new(next_value.0.as_slice()))
								.unwrap_or(false) {
								return None;
							}

							let deleted_values = &mut self.deleted_values;
							self.current_value_iter.as_mut().and_then(|iter| iter.next()).map(|next_value|
								deleted_values.push(next_value.0)
							);
							if next_change.1.is_some() {
								do_change = true;
								break;
							} else {
								// advance
								let _ = self.changes_iter.next();
							}
						},
					}
				} else {
					do_value = true;
					break;
				}
			} else {
				if let Some(next_change) = self.changes_iter.peek() {
					if limit.map(|limit| limit < &LeftNibbleSlice::new(next_change.0.as_slice()))
						.unwrap_or(false) {
						return None;
					}
					if next_change.1.is_some() {
						do_change = true;
						break;
					} else {
						// delete nothing.
						let _ = self.changes_iter.next();
					}
				} else {
					return None;
				}
			}
		}

		if do_change {
			return self.changes_iter.next().map(|kv| (
				// TODO have NibbleVec without copy from vec
				(&LeftNibbleSlice::new(kv.0.as_slice())).into(),
				IndexOrValue2::Value(kv.1.expect("Checked above")),
			));
		}
		if do_value {
			return self.current_value_iter.as_mut().and_then(|iter| iter.next()).map(|next_value| (
				// TODO have NibbleVec without copy from vec
				(&LeftNibbleSlice::new(next_value.0.as_slice())).into(),
				IndexOrValue2::StoredValue(next_value.1),
			));
		}

		None
	}

	fn index_or_value(&mut self) -> Option<(NibbleVec, IndexOrValue2<V>)> {
		let (do_value, do_index) = if let Some(next_value) = self.current_value_iter.as_mut().and_then(|iter| iter.peek()) {
			if let Some(next_index) = self.index_iter.last_mut().and_then(|i| i.iter.peek()) {
				// TODO use simple cmp if no need for commons and prefix in the future
				match LeftNibbleSlice::new(next_value.0.as_ref()).cmp_common_and_starts_with(&(&next_index.0).into()) {
					(Ordering::Equal, _commons, _is_prefix)
					| (Ordering::Greater, _commons, _is_prefix) => {
						// index
						(false, true)
						},
					(Ordering::Less, _commons, _is_prefix) => {
						// value
						(true, false)
					},
				}
			} else {
				// value
				(true, false)
			}
		} else {
			// index
			(false, true) 
		};
		if do_value {
			return self.current_value_iter.as_mut().and_then(|iter| iter.next()).map(|next_value| (
				// TODO have NibbleVec without copy from vec
				(&LeftNibbleSlice::new(next_value.0.as_slice())).into(),
				IndexOrValue2::StoredValue(next_value.1),
			));
		}
		if do_index {
			return self.index_iter.last_mut().and_then(|i| i.iter.next()).map(|next_index| (
				next_index.0,
				IndexOrValue2::Index(next_index.1),
			));
		}
		None
	}

	fn init_stack_index(&mut self) {
		// stack very first index
		if let Some(conf_index_depth) = self.indexes_conf.next_depth2(None) {
			let iter = self.indexes.iter(conf_index_depth, 0, LeftNibbleSlice::new(&[]));
			let iter = Iterator::peekable(iter);
			self.index_iter.push(StackedIndex2{conf_index_depth, iter});
		}
		//self.init_state();
	}

	// TODO remove result ??
	fn stack_index(&mut self) -> bool {
		if let Some(next_change) = self.changes_iter.peek() {
			let current_index_depth = self.index_iter.last().map(|i| i.conf_index_depth);
			if let Some(conf_index_depth) = self.indexes_conf.next_depth2(current_index_depth) {
				let iter = self.indexes.iter(
					conf_index_depth,
					current_index_depth.unwrap_or(0),
					LeftNibbleSlice::new(next_change.0.as_slice()),
				);
				let mut iter = Iterator::peekable(iter);
				if iter.peek().is_some() {
					self.index_iter.push(StackedIndex2{conf_index_depth, iter});
					return true;
				}
			}
		}
		false
	}

	// 
	fn init_state(&mut self) {
		if let Some(next_change) = self.changes_iter.peek() {
			let key_change = next_change.0.as_ref();
			if let Some(next_index) = self.index_iter.last_mut().and_then(|i| i.iter.peek()) {
				match LeftNibbleSlice::new(key_change).cmp_common_and_starts_with(&(&next_index.0).into()) {
					(Ordering::Less, _commons, _is_prefix) => {
						// iterate until index (even if not a prefix this will branch to prefix of index and
						// require change).
						self.state = Next2::IterToChange;
					},
					(_, _commons, false) => {
						// iterate until index without values.
						// TODO paded buffer on &mut that return slice.
						let next_index = next_index.0.clone().padded_buffer();
						// TODO setting to None should be more efficient or setting on state change.
						// or TODO iter_from required to be lazy (only costy of first call to peek/next.
						self.current_value_iter = Some(Iterator::peekable(self.values.iter_from(next_index.as_slice())));
						self.state = Next2::NextIndex;
					},
					(_, _commons, true) => {
						// iterate then stack index.
						self.state = Next2::IterToStackIndex;
					},
				}
			}

			unimplemented!("TODO feed next_item and then feed first branch");

/*			assert!(self.value_or_change().is_none());
			debug_assert!(self.next_item.is_some());*/
		}
	}

	fn stack_next_index(&mut self, next_change: &[u8]) {
		if let Some(next_change) = self.changes_iter.peek() {
			let current_index_depth = self.index_iter.last().map(|i| i.conf_index_depth);
			if let Some(conf_index_depth) = self.indexes_conf.next_depth2(current_index_depth) {
				if let Some(next_branch_depth) = self.stack_branches.current_branch_depth() {
					let iter = self.indexes.iter(
						conf_index_depth,
						current_index_depth.unwrap_or(0),
						LeftNibbleSlice::new_len(next_change.0.as_slice(), next_branch_depth),
					);
					let iter = Iterator::peekable(iter);
					self.index_iter.push(StackedIndex2{conf_index_depth, iter});
					self.state = Next2::IterToBranch;
				}
			}
		}
	}

	// When reading value we skip the deleted ones.
	fn peek_value_iter(&mut self, bound: Option<&[u8]>) -> Option<&(Vec<u8>, Vec<u8>)> {
		unimplemented!()
/*		// TODO save deleted values as deleted indexes.
		if let Some(next_value) = self.current_value_iter.peek() {
			if let Some(next_change) = self.changes_iter.peek() {
				if next_change.1.is_none() {
					let key_change = ;
					match next_change.0.as_ref().cmp(next_value.0.as_ref()) {
						Ordering::Less => {
							let _ = self.changes_iter.next();
							return self.peek_value_iter(bound);
						},
						Ordering::Greater => {
						},
						Ordering::Eq => {
						},
					}
				}
			}
		}*/
	}
	/*
	fn value_or_change(&mut self, next_change: &[u8]) -> Option<(NibbleVec, IndexOrValue2<V>)> {
		let value = if let Some(next_change) = self.changes_iter.peek() {
			if let Some(next_value) = self.current_value_iter.peek() {
				match next_value.0.cmp(next_change.0) {
					Ordering::Equal => {
						if change.is_some() {
						} else {
						}
					},
					Ordering::Greater
					Ordering::Less
				}
			}
		} else {
		}

		std::mem::replace(&mut self.next_item, value)
	}*/

	fn current_index_depth(&self) -> Option<usize> {
		self.index_iter.last().map(|i| i.conf_index_depth)
	}

	fn iter_value_skip_to(&mut self, key: &[u8]) {
		self.current_value_iter = Some(Iterator::peekable(self.values.iter_from(key)));
	}
}
