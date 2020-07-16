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
pub type IndexBackendIter<'a> = Box<dyn Iterator<Item = (Vec<u8>, usize, Vec<u8>)> + 'a>;

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
	fn read(&self, depth: usize, index: &[u8]) -> Option<(usize, Vec<u8>)>;
	/// Insert an `encode_index` with and `actual_depth` at configured `depth` for a given `index`.
	fn write(&mut self, depth: usize, index: IndexPosition, actual_depth: usize, encoded_index: Vec<u8>);
	/// Remove any value at a key.
	fn remove(&mut self, depth: usize, index: IndexPosition);
	/// Iterate over the index from a key.
	fn iter<'a>(&'a self, depth: usize, from_index: &[u8]) -> IndexBackendIter<'a>;
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
impl IndexBackend for BTreeMap<Vec<u8>, (usize, Vec<u8>)> {
	fn read(&self, depth: usize, index: &[u8]) -> Option<(usize, Vec<u8>)> {
		self.get(&index_tree_key(depth, index)[..]).cloned()
	}
	fn write(&mut self, depth: usize, mut index: IndexPosition, actual_depth: usize, encoded_index: Vec<u8>) {
		let odd = depth % 8;
		if odd != 0 {
			index.last_mut().map(|l| 
				*l = *l & !(255 >> odd)
			);
		}
		self.insert(index_tree_key_owned(depth, index).to_vec(), (actual_depth, encoded_index));
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
		Box::new(range.into_iter().map(|(k, (i, v))| (k.to_vec(), *i, v.to_vec())))
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
	/// TODO this is not really efficient and use on every node
	pub fn next_depth(&self, depth: usize) -> Option<usize> {
		for i in self.0.iter() {
			let i = *i as usize;
			if i >= depth {
				return Some(i)
			}
		}
		None
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
}
