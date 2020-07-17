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
	pub is_leaf: bool,
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
		let odd = depth % 8;
		if odd != 0 {
			position.last_mut().map(|l| 
				*l = *l & !(255 >> odd)
			);
		}
		self.insert(index_tree_key_owned(depth, position).to_vec(), index);
	}
	fn remove(&mut self, depth: usize, mut index: IndexPosition) {
		let odd = depth % 8;
		if odd != 0 {
			index.last_mut().map(|l| 
				*l = *l & !(255 >> odd)
			);
		}
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
		Box::new(range.into_iter().map(|(k, ix)| (k.to_vec(), ix.clone())))
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
	/// TODO probably do not need the usize
	Index(Index),
	/// Value node value.
	Value(B),
}

/// Iterator over index and value for root calculation
/// and index update.
pub struct RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: IntoIterator<Item = (Vec<u8>, Option<V>)>,
{
	pub values: &'a KB,
	pub indexes: &'a IB,
	pub indexes_conf: &'a DepthIndexes,
	pub values_delta: ID,
	pub deleted_indexes: Vec<(usize, IndexPosition)>,
}

impl<'a, KB, IB, V, ID> Iterator for RootIndexIterator<'a, KB, IB, V, ID>
	where
		KB: KVBackend,
		IB: IndexBackend,
		V: AsRef<[u8]>,
		ID: IntoIterator<Item = (Vec<u8>, Option<V>)>,
{
	type Item = (Vec<u8>, IndexOrValue<V>);

	fn next(&mut self) -> Option<Self::Item> {

		None
	}
}
