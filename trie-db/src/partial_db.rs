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
		KVBackend::iter(&**self) 
	}
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
		KVBackend::iter_from(self.as_ref(), start) 
	}
}

impl KVBackend for Arc<dyn KVBackend + Send + Sync> {
	fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
		KVBackend::read(self.as_ref(), key) 
	}
	fn write(&mut self, _key: &[u8], _value: &[u8]) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn remove(&mut self, _key: &[u8]) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn iter<'a>(&'a self) -> KVBackendIter<'a> {
		KVBackend::iter(&**self) 
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
	fn write(&mut self, _depth: usize, _index: LeftNibbleSlice, _value: Index) {
		unimplemented!("TODO split trait with mut and non mut");
	}
	fn remove(&mut self, _depth: usize, _index: LeftNibbleSlice) {
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
pub enum Item<B> {
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

impl<B> Item<B> {
	/// Access the index branch depth (including the partial key).
	pub fn exact_index(&self) -> Option<bool> {
		match self {
			Item::Index(index) => Some(index.on_index),
			_ => None,
		}
	}
}

struct NextCommonUnchanged {
	kind: NextCommonKind, // TODO may be useless (if Option<usize> for depth)
	prev_kind: NextCommonKind, // TODO may be useless
	value_deleted_from: Option<usize>,
	depth: usize,
}

#[derive(Clone, Copy)]
enum NextCommonKind {
	Index,
	Value,
	None,
}

impl Default for NextCommonUnchanged {
	fn default() -> Self {
		NextCommonUnchanged {
			kind: NextCommonKind::None,
			prev_kind: NextCommonKind::None,
			value_deleted_from: None,
			depth: 0,
		}
	}
}

impl NextCommonUnchanged {
	fn depth(&self) -> Option<usize> {
		if let NextCommonKind::None = self.kind {
			None
		} else {
			Some(self.depth)
		}
	}
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
	// Common with next item prior to change.
	// Needs update on value iteration (including skipped value)
	// or index iteration (not skipped index).
	next_common_unchanged: NextCommonUnchanged,
	// Common with next inedx from value insert (not value change).
	// when it differs with `next_common_unchanged`, this
	// indicates either inserted in partial (>), or every prior
	// child deleted (<).
	next_commmon_insert_index: Option<usize>,
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
	depth_with_previous: usize,
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
			next_common_unchanged: NextCommonUnchanged::default(),
			next_commmon_insert_index: None,
			deleted_indexes,
			deleted_values,
		};

		// always stack first level of indexes.
		iter.init_stack_index();

//		assert!(iter.feed_next_item().is_none());

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

impl<'a, KB, IB, V, ID> Iterator for RootIndexIterator2<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: Iterator<Item = (Vec<u8>, Option<V>)>,
{
	type Item = (NibbleVec, Item<V>);

	fn next(&mut self) -> Option<Self::Item> {
		self.get_next_item()
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

	/// consume all following delete of value.
	fn consume_next_deletes(&mut self) {
		// TODO avoid calling this twice (in get_next and here).
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
			None => (),
			Some((Ordering::Less, _commons, _is_prefix)) => (),
			Some((_, _commons, false)) => {
				// change more than index.
				return;
			},
			Some((_, _commons, true)) => {
				// change bellow or equal index
				// index skip
				self.skip_index();
				// Stack index
				self.stack_index();
				// recurse
				return self.consume_next_deletes();
			},
		}

		// TODO update next_common_unchanged calls
		if self.consume_changed_value() {
			return self.consume_next_deletes(); // TODO is probably useless (called in advance value)
		}
	}

	fn update_next_common_unchanged(&mut self, current_key: &LeftNibbleSlice, prev_index: bool, is_delete: bool) {
		self.next_common_unchanged = self.next_common_unchanged(current_key, is_delete);
	}

	// TODO remove prev_index param
	fn next_common_unchanged(&mut self, current_key: &LeftNibbleSlice, is_delete: bool) -> NextCommonUnchanged {
		// TODO this need to be reuse in get_next_item to avoid some key comparison. (actually probably
		// the other way).
		// + TODO maybe we already got peek index against peek value -> do memoize it
		// until next is call

		// against index
		let index_diff = self.index_iter.last_mut().and_then(|i| i.iter.peek()).map(|next_index| {
			current_key.cmp_common_and_starts_with(&(&next_index.0).into()).1
		});

		// against value
		let value_diff = self.current_value_iter.as_mut().and_then(|iter| iter.peek()).map(|next_value| {
			current_key.cmp_common_and_starts_with(&LeftNibbleSlice::new(next_value.0.as_slice()).into()).1
		});

		let (next_value, depth_next) = match (value_diff, index_diff) {
			(Some(val), Some(index)) if val > index => {
				(false, index)
			},
			(_, Some(depth)) => {
				(false, depth)
			},
			(Some(depth), _) => {
				(false, depth)
				/*let can_be_empty = is_delete && match self.next_common_unchanged {
					NextCommonUnchanged::Index { depth, .. } => {
						debug_assert!(depth_next >= depth); // if false, when < then return true (new branch)
						depth_next != depth // can only be empty if first index at depth
					},
					NextCommonUnchanged::Value { depth, can_be_empty, .. } => {
						debug_assert!(depth_next >= depth); // if false, when < then return true (new branch)
						can_be_empty || depth_next == depth
					},
					NextCommonUnchanged::None => true,
				};*/
			},
			(None, None) => {
				let mut res = NextCommonUnchanged::default();
				res.prev_kind = self.next_common_unchanged.kind;
				res.value_deleted_from = self.next_common_unchanged.value_deleted_from;
				return res;
			},
		};

		let value_deleted_from = if !is_delete {
			None
		} else if let Some(prev) = self.next_common_unchanged.value_deleted_from.clone() {
			Some(crate::rstd::cmp::min(prev, depth_next))
		} else {
			is_delete.then(|| depth_next)
		};

		NextCommonUnchanged {
			prev_kind: self.next_common_unchanged.kind,
			kind: if next_value {
				NextCommonKind::Value
			} else {
				NextCommonKind::Index
			},
			value_deleted_from,
			depth: depth_next,
		}
	}

	fn update_next_commmon_insert_index(&mut self, current_key: &LeftNibbleSlice) {
		// TODO from memoize previously peeked change and peeked index
		self.next_commmon_insert_index = self.index_iter.last_mut().and_then(|i| i.iter.peek()).map(|next_index| {
			current_key.cmp_common_and_starts_with(&(&next_index.0).into()).1
		});
	}

	fn advance_value(&mut self) -> Option<(NibbleVec, Item<V>)> {
		self.current_value_iter.as_mut().and_then(|iter| iter.next()).map(|kv| {
			let key = LeftNibbleSlice::new(kv.0.as_slice());
			self.update_next_common_unchanged(&key, false, false);
			self.consume_next_deletes();
			// TODO have NibbleVec without copy (at least from backing))
			((&key).into(), Item::StoredValue(kv.1))
		})
	}

	// return true if a value was skipped (deleted or modified).
	fn skip_value(&mut self, deleted: bool) -> Option<NibbleVec> {
		self.current_value_iter.as_mut().and_then(|iter| iter.next()).map(|kv| {
			let key = LeftNibbleSlice::new(kv.0.as_slice());
			self.update_next_common_unchanged(&key, false, deleted);
			self.consume_next_deletes();
			(&key).into()
		})
	}

	fn advance_index(&mut self) -> Option<Option<(NibbleVec, Item<V>)>> {
		let depth_ix = self.index_iter.last().map(|i| i.conf_index_depth).unwrap_or(0);
		let next_commmon_insert_index = self.next_commmon_insert_index.take();
		let result = if let Some(next_index) = self.index_iter.last_mut()
			.and_then(|i| i.iter.next()) {


			// Done in order to ensure next change follow index.
			self.consume_next_deletes(); // TODO is it noops? (we only advance index)

			let sibling_depth = depth_ix;
			// advance value iter. TODO try to remove this instanciation.
			let mut key_index: NibbleVec = (&next_index.0.as_slice().truncate(sibling_depth)).into();
			let previous_value_iter = self.current_value_iter.take();
			self.current_value_iter = if key_index.next_sibling() {
				Some(Iterator::peekable(self.values.iter_from(key_index.padded_buffer().as_slice())))
			} else {
				None
			};

			// TODO lighter method?? (only need depth)
			let next_common_depth = self.next_common_unchanged(&next_index.0.as_slice(), false).depth(); // TODO factor


			let parent_branch_depth = match (self.next_common_unchanged.depth(), next_common_depth) {
				(Some(a), Some(b)) => Some(crate::rstd::cmp::max(a, b)),
				(Some(a), None)
				| (None, Some(a)) => Some(a),
				_ => None,
			};
			let mut do_skip = false;
			// check if change is inserted into prefix
			if let Some(ix) = next_commmon_insert_index.as_ref() {
				debug_assert!(ix <= &depth_ix); // TODO can remove from below condition??
				if parent_branch_depth.as_ref().map(|parent| ix > parent).unwrap_or(true)
					&& ix <= &depth_ix {
					do_skip = true;
				}
			}
			if !do_skip {
				if let Some(next_change) = self.changes_iter.peek() {
					let ix = next_index.0.as_slice().common_length(&LeftNibbleSlice::new(next_change.0.as_slice()));
					if parent_branch_depth.as_ref().map(|parent| &ix > parent).unwrap_or(true)
						&& ix <= depth_ix {
						do_skip = true;
					}
				}
			}
			if !do_skip {
				if let Some(parent_branch) = parent_branch_depth {
					// check previous allow skip.
					// Note that parent branch with value is a case of previous undeleted
					// common item.
					if self.next_common_unchanged.depth < parent_branch
						|| self.next_common_unchanged.value_deleted_from.map(|from| from <= parent_branch)
							.unwrap_or(false)
					{
						// check no next in branch (we did consume deletes above).
						if let Some(depth) = next_common_depth {
							// next is under branch (no next)
							do_skip = depth < parent_branch;
						} else {
							// no next
							do_skip = true;
						}
					}
				} else {
					// from root, ignore: any change is in partial or would have stack index before.
					unreachable!();
				}
			}
			if do_skip {
				self.current_value_iter = previous_value_iter;
				self.stack_index();
				return None;
			}

			// advance value iter.
			self.update_next_common_unchanged(&next_index.0.as_slice(), true, false);
			self.consume_next_deletes();

			Some((next_index.0, Item::Index(next_index.1)))
		} else {
			None
		};
		if result.is_some() && self.index_iter.last_mut().and_then(|i| i.iter.peek()).is_none() {
			let _ = self.index_iter.pop();
		}
		Some(result)
	}

	// TODO rename to delete_change?
	fn skip_change(&mut self) {
		if self.changes_iter.next().is_some() {
			self.consume_next_deletes();
		}
	}

	// TODO rename to return_change?
	fn advance_change(&mut self) -> Option<(NibbleVec, Item<V>)> {
		self.changes_iter.next().map(|kv| {
			let key = LeftNibbleSlice::new(kv.0.as_slice());
			self.update_next_commmon_insert_index(&key);
			self.consume_next_deletes();
			// TODO have NibbleVec without copy from vec
			((&key).into(),
				Item::Value(kv.1.expect("Checked above")))
		})
	}

	// TODO merge with stack?? probably, better semantic
	fn skip_index(&mut self) {
		// this next common is only to check if we should stack next index, which we did.
		self.next_commmon_insert_index = None;
		let _ = self.index_iter.last_mut().and_then(|i| i.iter.next());
	}

	fn get_next_item(&mut self) -> Option<(NibbleVec, Item<V>)> {
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
					return self.advance_value();
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
				self.skip_index();
				// Stack index
				self.stack_index();
				// recurse
				return self.get_next_item();
			},
		}
	}

	// TODO fact with changed_or_value?
	fn consume_changed_value(&mut self) -> bool {
		if let Some(next_change) = self.changes_iter.peek() {
			if next_change.1.is_none() {
				if let Some(next_value) = self.current_value_iter.as_mut().and_then(|iter| iter.peek()) {
					match next_change.0.as_slice().cmp(next_value.0.as_ref()) {
						Ordering::Less => {
							// delete nothing.
							self.skip_change();
							return true;
						},
						Ordering::Greater => {
							// got value
							return false;
						},
						Ordering::Equal => {
							// advance
							self.skip_change();
							if let Some(key) = self.skip_value(true) {
								self.deleted_values.push(key.padded_buffer_vec());
							}
							return true;
						},
					}
				} else {
					self.skip_change();
					// no next value, consume following delete.
					return true;
				}
			}
		}
		false
	}

	fn change_or_value(&mut self, limit: Option<&LeftNibbleSlice>) -> Option<(NibbleVec, Item<V>)> {
		let mut do_change = false;
		let mut do_value = false;
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
						} else {
							// delete nothing.
							self.skip_change();
						}
					},
					Ordering::Greater => {
						// could be <= but we already check against index and this bound is usually index.
						if limit.map(|limit| limit < &LeftNibbleSlice::new(next_value.0.as_slice()))
							.unwrap_or(false) {
							return None;
						}
						do_value = true;
					},
					Ordering::Equal => {
						// could be <= but we already check against index and this bound is usually index.
						if limit.map(|limit| limit < &LeftNibbleSlice::new(next_value.0.as_slice()))
							.unwrap_or(false) {
							return None;
						}

						do_change = next_change.1.is_some();
						if !do_change {
							// advance
							self.skip_change();
						}
						// TODO bad code: here we do not have delet: siwtch to assert next_change.1.is_some()?
						if self.skip_value(!do_change).is_some() && !do_change {
							unreachable!("Delete are done preentively with consume_next_deletes");
						}
					},
				}
			} else {
				do_value = true;
			}
		} else {
			if let Some(next_change) = self.changes_iter.peek() {
				if limit.map(|limit| limit < &LeftNibbleSlice::new(next_change.0.as_slice()))
					.unwrap_or(false) {
					return None;
				}
				if next_change.1.is_some() {
					do_change = true;
				} else {
					// delete nothing.
					self.skip_change();
				}
			} else {
				return None;
			}
		}

		if do_change {
			return self.advance_change();
		}
		if do_value {
			return self.advance_value();
		}

		self.get_next_item()
	}

	fn index_or_value(&mut self) -> Option<(NibbleVec, Item<V>)> {
		let (do_value, do_index) = if let Some(next_value) = self.current_value_iter.as_mut().and_then(|iter| iter.peek()) {
			if let Some(next_index) = self.index_iter.last_mut().and_then(|i| i.iter.peek()) {
				// TODO use simple cmp if no need for commons and prefix in the future
				match LeftNibbleSlice::new(next_value.0.as_ref()).cmp_common_and_starts_with(&(&next_index.0).into()) {
					(Ordering::Equal, _commons, _is_prefix)
					| (Ordering::Greater, _commons, _is_prefix) => {
						// index
						(false, true)
						},
					(Ordering::Less, commons, _is_prefix) => {
						self.index_iter.last_mut().map(|i| {
							i.depth_with_previous = commons;
						});
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
			return self.advance_value();
		}
		if do_index {
			if let Some(advance) = self.advance_index() {
				return advance;
			}
			// stack index: recurse
			return self.get_next_item();
		}
		None
	}

	fn init_stack_index(&mut self) {
		// stack very first index
		if let Some(conf_index_depth) = self.indexes_conf.next_depth2(None) {
			let iter = self.indexes.iter(conf_index_depth, 0, LeftNibbleSlice::new(&[]));
			let iter = Iterator::peekable(iter);
			let depth_with_previous = 0;
			self.index_iter.push(StackedIndex2{conf_index_depth, iter, depth_with_previous});
		}
	}

	// TODO remove result ??
	fn stack_index(&mut self) -> bool {
		if let Some(next_change) = self.changes_iter.peek() {
			let depth_with_previous = self.index_iter.last().map(|i| i.depth_with_previous).unwrap_or(0);
			let current_index_depth = self.index_iter.last().map(|i| i.conf_index_depth);
			if let Some(conf_index_depth) = self.indexes_conf.next_depth2(current_index_depth) {
				let iter = self.indexes.iter(
					conf_index_depth,
					current_index_depth.unwrap_or(0),
					LeftNibbleSlice::new(next_change.0.as_slice()),
				);
				let mut iter = Iterator::peekable(iter);
				if iter.peek().is_some() {
					self.index_iter.push(StackedIndex2{conf_index_depth, iter, depth_with_previous});
					return true;
				}
			}
		}
		false
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
	fn value_or_change(&mut self, next_change: &[u8]) -> Option<(NibbleVec, Item<V>)> {
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
