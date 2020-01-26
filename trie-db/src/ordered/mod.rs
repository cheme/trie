// Copyright 2020 Parity Technologies
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

//! This module contains implementation of trie/tree based on ordered sequential key only.

use hash_db::{HashDBRef, Prefix, EMPTY_PREFIX, Hasher, FixHash};
use super::{Result, DBValue};
use crate::rstd::marker::PhantomData;
use crate::rstd::vec::Vec;
pub mod key {
	/// Base type for key, TODO
	/// implementing on usize for now,
	/// then this will probably look like
	/// substrate simple arithmetic trait.
	/// -> need an implementation for [u8]
	/// (unbounded key length)
	pub trait OrderedKey: Ord {}

	impl<T: Ord> OrderedKey for T {}
}



pub mod meta {
	use hash_db::Hasher;

	/// Codec for the meta needed to get
	/// information for the tree.
	pub trait MetaCodec {
		/// If false we do not associate
		/// meta to the trie, their value
		/// should be trusted from an external
		/// source (eg inclusion in the header
		/// of a block like root), in this case
		/// the implementation of this trait should
		/// never be use. Then trie root is the binary_root.
		/// If true the code will be use to produce
		/// a root with those information and the
		/// root will be hash(meta ++ binary_root),
		/// so at minimal an additional round of
		/// hashing. Binary root is included in meta.
		///
		/// A node contaning this `meta ++ binary_root`
		/// is using a prefix of length 0.
		const ATTACH_TO_ROOT: bool;

		/// The buffer for this codec, this allows
		/// to use fix length buffer.
		type Buff: AsMut<[u8]> + AsRef<[u8]>;
		/// The hash to use if we need
		/// to associate meta.
		type Hash: Hasher;
		/// The actual meta to use.
		type Meta;
		/// Decode
		fn decode(input: &[u8]) -> Self::Meta;
		/// Encode TODO use write and stream trait?
		fn encode(meta: &Self::Meta) -> Vec<u8>;
	}

	/// Direct hasher as meta indicates there
	/// is no meta so we can use the root directly.
	impl<H: Hasher> MetaCodec for H {
		const ATTACH_TO_ROOT: bool = false;
		type Buff = [u8;0];
		type Hash = Self;
		type Meta = ();
		fn decode(_input: &[u8]) -> Self::Meta { () }
		fn encode(_meta: &Self::Meta) -> Vec<u8> { Vec::new() }
	}

}

#[derive(PartialEq, Eq, Debug)]
/// A binary trie with guaranties
/// of content being in a fix range
/// of sequential values.
pub struct SequenceBinaryTree<K: key::OrderedKey> {
	// Metadata (needs to be validated)

	/// global offset for index.
	offset: K,
	/// Nb deleted values at start.
	start: K,
	start_depth: usize,
	/// Nb deleted values at end.
	end: K,
	end_depth: usize,

	// depth of the full tree (maximum depth)
	depth: usize,
	// memmoïze 2^depth
	length: K,

	_ph: PhantomData<K>,
}

pub struct SequenceBinaryTreeDB<'a, K: key::OrderedKey, H: FixHash> {
	tree: &'a SequenceBinaryTree<K>,
	db: &'a dyn HashDBRef<H, DBValue>,
	root: &'a <H::Hasher as Hasher>::Out,
}

impl Default for SequenceBinaryTree<usize> {
	fn default() -> Self {
		SequenceBinaryTree {
			offset: 0,
			start: 0,
			start_depth: 0,
			end: 0,
			end_depth: 0,
			depth: 0,
			length: 0,
			_ph: PhantomData,
		}
	}
}

fn depth(nb: usize) -> usize {
	if nb == 0 {
		0
	} else {
		((0usize.leading_zeros() - (nb - 1).leading_zeros()) as usize) + 1
	}
}

fn right_at(value: usize, index: usize) -> bool {
	value & (1 << index) != 0
}


impl SequenceBinaryTree<usize> {
	pub fn new(offset: usize, start: usize, number: usize) -> Self {
		let len = start + number;
		let length = if len == 0 {
			0
		} else {
			len.next_power_of_two()
		};
		let end = length - start - number;
		let start_depth = depth(start);
		let end_depth = depth(end);
		let depth = depth(length);
		SequenceBinaryTree {
			offset,
			start,
			start_depth,
			end,
			end_depth,
			depth,
			length,
			_ph: PhantomData,
		}
	}

	fn push(&mut self, mut nb: usize) {
		while nb > self.end {
			nb -= self.end;
			self.depth += 1;
			if self.length == 0 {
				self.length += 1;
				self.end = 1;
			} else {
				self.end = self.length;
				self.length *= 2;
			}

		}
		self.end -= nb;
		self.end_depth = depth(self.end);
	}

	fn depth_index(&self, index: usize) -> usize {
		if self.depth == 0 {
			// TODO this corner case is bad, should switch depth indexing.
			// and boolean empty.
			return 0;
		} else if self.depth == 1 {
			// one elt first index too
			return 0;
		};
		let mut depth = self.depth - 2;
		let mut result = self.depth;
		// TODO this iteration could be probably optimize with bit logic
		// but that is probably not big gain.
		while index & (1 << depth) != 0
			|| self.end & (1 << depth) != 0
		{
			if self.end & (1 << depth) != 0 {
				result -= 1;
			}
			if depth == 0 {
				break;
			} else {
				depth -= 1;
			}
		}

		result
	}

	fn pop(&mut self, nb: usize) {
		unimplemented!("update max depth");
	}

	fn pop_front(&mut self, nb: usize) {
		unimplemented!("update max depth");
		// TODO if start = max_depth_length / 2 -> max_depth - 1
	}

	fn max_depth_length(end: &usize) -> usize {
		// 2^x = max_depth_length
		unimplemented!()
	}


	fn front_depth(index: usize) -> usize {
		unimplemented!("for index between end and max len");
	}

	fn tail_depth(index: usize) -> usize {
		unimplemented!("for index between end and max len");
	}
}



// prefix scheme, the prefix use to avoid conflict of hash in a single trie is build
// upon indexed key of the leftmost child with the depth of the prefix and then the compact encoding.
// Therefore it can be use to iterate if there is only a single state for the trie.
//
// prefix scheme: not two node with same prefix ++ hash.
// meta & root cannot happen.
// level 1 can happen: just prefix 0 or 1
// level 2 with level 1 can happen but only on different prefix
// level 3 with level 1
//
// no offset and the depth of , therefore compact encoding is rather suitable for it.
// We use a compact
//
// NOTE that changing trie changes depth (existing k at depth 2 moving to depth 4), therefore the scheme is rather broken
// as we cannot address the nodes anymore.
// Therefore we should prefix over the index(key) as compact. For intermediattory key it will be
// the leftmost key index. TODO make test to check no collision and write asumption that to create
// collision we need inline values of length == to hash (to target previous 2 values hash eg for 3
// nodes trie: hash(v1,v2) = h1, v3 = h1 but this implies h1 of length of hash and this means that
// we hash the value (with inline hash of length being strictly the hash length this can be use: 
// CONCLUSION even if we cannot run inline values of length of the H::Out (more should be fine as
// it implies a second round of hashing) -> can be avoided with custom encoder.
// Inline value less than size hash are a problem on the other hand: when close to size hash we can
// find collision rather easilly, but that does not work because leftmost index is 3 for v3 and 1
// for h1 so seems rather safe. If removed from start (offset), then it is not written so safe to
// except V1 del then V2 become h(v1,v2) and then v3 = v2 does break but prefix do not move : v2 is
// still 2 and v3 is still 3 so fine to.
// Could add a value bool to the prefix or the compact encoding scheme to indicate that it is a
// terminal value -> no value are stored outside? -> seems good to iterate (same for terminal node
// with a inline value -> 4 info here : intermediate, first value, second value, both value (the
// three lasts being the same (first in fact). This lead to possible iteration by.
// For partial storage we can use same approach for a few level of intermediate (this will bound
// key size for fix prefix, then last value is reserved for compact encoding of level next which
// should really never happen).
//
// NOTE inline value does not make sense, api should only use hash, additional api could store
// access values from terminal hash.
// Prefix wise, we could store in same db with key as prefix. Also if we want to inline value,
// then the value just need to be extract from terminal hash instead. (terminal hash marker
// and value describe above is still interesting).

#[cfg(test)]
mod test {
	use keccak_hasher::FixKeccakHasher;

	//type Tree = super::SequenceBinaryTree<usize, FixKeccakHasher>;
	type Tree = super::SequenceBinaryTree<usize>;

	#[test]
	fn test_max_depth() {
		let values = [
			(0, 0),
			(1, 1),
			(2, 2),
			(3, 3),
			(4, 3),
			(5, 4),
			(8, 4),
			(9, 5),
			(16, 5),
			(17, 6),
			(32, 6),
		];
		let mut tree = Tree::default();
		let mut prev = 0;
		for (nb, depth) in values.iter().cloned() {
			let inc = nb - prev;
			prev = nb;
			tree.push(inc);
			assert_eq!(tree.depth, depth);
			let tree2 = Tree::new(0, 0, nb);
			assert_eq!(tree2.depth, depth);
			assert_eq!(tree, tree2);
		}
	}

	#[test]
	fn test_depth_index() {
		// 8 trie
		let tree = Tree::new(0, 0, 7);
		assert_eq!(tree.depth_index(3), 4);
		assert_eq!(tree.depth_index(4), 4);
		assert_eq!(tree.depth_index(6), 3);
		let tree = Tree::new(0, 0, 6);
		assert_eq!(tree.depth_index(0), 4);
		assert_eq!(tree.depth_index(3), 4);
		assert_eq!(tree.depth_index(4), 3);
		assert_eq!(tree.depth_index(5), 3);
		let tree = Tree::new(0, 0, 5);
		assert_eq!(tree.depth_index(3), 4);
		assert_eq!(tree.depth_index(4), 2);
		// 16 trie
		let tree = Tree::new(0, 0, 12);
		assert_eq!(tree.depth_index(7), 5);
		assert_eq!(tree.depth_index(8), 4);
		assert_eq!(tree.depth_index(11), 4);
		let tree = Tree::new(0, 0, 11);
		assert_eq!(tree.depth_index(7), 5);
		assert_eq!(tree.depth_index(8), 4);
		assert_eq!(tree.depth_index(9), 4);
		assert_eq!(tree.depth_index(10), 3);
		let tree = Tree::new(0, 0, 10);
		assert_eq!(tree.depth_index(7), 5);
		assert_eq!(tree.depth_index(8), 3);
		assert_eq!(tree.depth_index(9), 3);
		let tree = Tree::new(0, 0, 9);
		assert_eq!(tree.depth_index(7), 5);
		assert_eq!(tree.depth_index(8), 2);
		// 32 trie TODO
	}

}
