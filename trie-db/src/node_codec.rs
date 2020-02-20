// Copyright 2017, 2018 Parity Technologies
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

//! Generic trait for trie node encoding/decoding. Takes a `hash_db::Hasher`
//! to parametrize the hashes used in the codec.

use crate::MaybeDebug;
use crate::node::{Node, NodePlan};
use crate::ChildReference;

use crate::rstd::{borrow::Borrow, Error, hash, vec::Vec, EmptyIter, ops::Range, marker::PhantomData};

/// Representation of a nible slice (right aligned).
/// It contains a right aligned padded first byte (first pair element is the number of nibbles
/// (0 to max nb nibble - 1), second pair element is the padded nibble), and a slice over
/// the remaining bytes.
pub type Partial<'a> = ((u8, u8), &'a[u8]);

/// Trait for trie node encoding/decoding.
pub trait NodeCodec: Sized {
	/// Codec error type.
	type Error: Error;

	/// Output type of encoded node hasher.
	type HashOut: AsRef<[u8]> + AsMut<[u8]> + Default + MaybeDebug + PartialEq + Eq
		+ hash::Hash + Send + Sync + Clone + Copy;

	/// Get the hashed null node.
	fn hashed_null_node() -> Self::HashOut;

	/// Decode bytes to a `NodePlan`. Returns `Self::E` on failure.
	fn decode_plan(data: &[u8]) -> Result<NodePlan, Self::Error>;

	/// Decode bytes to a `Node`. Returns `Self::E` on failure.
	fn decode(data: &[u8]) -> Result<Node, Self::Error> {
		Ok(Self::decode_plan(data)?.build(data))
	}

	/// Check if the provided bytes correspond to the codecs "empty" node.
	fn is_empty_node(data: &[u8]) -> bool;

	/// Returns an encoded empty node.
	fn empty_node() -> &'static [u8];

	/// Returns an encoded leaf node
	fn leaf_node(partial: Partial, value: &[u8]) -> Vec<u8>;

	/// Returns an encoded extension node
	/// Note that number_nibble is the number of element of the iterator
	/// it can possibly be obtain by `Iterator` `size_hint`, but
	/// for simplicity it is used directly as a parameter.
	fn extension_node(
		partial: impl Iterator<Item = u8>,
		number_nibble: usize,
		child_ref: ChildReference<Self::HashOut>,
	) -> Vec<u8>;

	/// Returns an encoded branch node.
	/// Takes an iterator yielding `ChildReference<Self::HashOut>` and an optional value.
	fn branch_node(
		children: impl Iterator<Item = impl Borrow<Option<ChildReference<Self::HashOut>>>>,
		value: Option<&[u8]>,
		register_children: Option<&mut [Option<Range<usize>>]>,
	) -> Vec<u8>;

	/// Returns an encoded branch node with a possible partial path.
	/// `number_nibble` is the partial path length as in `extension_node`.
	fn branch_node_nibbled(
		partial: impl Iterator<Item = u8>,
		number_nibble: usize,
		children: impl Iterator<Item = impl Borrow<Option<ChildReference<Self::HashOut>>>>,
		value: Option<&[u8]>,
		register_children: Option<&mut [Option<Range<usize>>]>,
	) -> Vec<u8>;
}

use ordered_trie::{HashDBComplex, HasherComplex};
use hash_db::{HashDB, Prefix, HashDBRef, Hasher};

pub trait HashDBComplexDyn<H: Hasher, T>: Send + Sync + HashDB<H, T> {
	/// Insert a datum item into the DB and return the datum's hash for a later lookup. Insertions
	/// are counted and the equivalent number of `remove()`s must be performed before the data
	/// is considered dead.
	///
	/// TODO warn semantic of children differs from HashDBComplex (in HashDBComplex it is the
	/// children of the binary hasher, here it is the children of the patricia merkle trie).
	fn insert_complex(
		&mut self,
		prefix: Prefix,
		value: &[u8],
		children: &[Option<Range<usize>>],
	) -> H::Out;
}

impl<H: HasherComplex, T, C: HashDBComplex<H, T>> HashDBComplexDyn<H, T> for C {
	fn insert_complex(
		&mut self,
		prefix: Prefix,
		value: &[u8],
		children: &[Option<Range<usize>>],
	) -> H::Out {

		// TODO factor this with iter_build (just use the trait)
		let nb_children = children.iter().filter(|v| v.is_some()).count();
		let children = ComplexLayoutIterValues::new(
			nb_children,
			children.iter().filter_map(|v| v.as_ref()),
			value,
		);

		<C as HashDBComplex<H, T>>::insert_complex(
			self,
			prefix,
			value,
			nb_children,
			children,
			EmptyIter::default(),
			false,
		)
	}
}

impl<'a, H: Hasher, T> HashDBRef<H, T> for &'a dyn HashDBComplexDyn<H, T> {
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T> {
		self.as_hash_db().get(key, prefix)
	}

	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool {
		self.as_hash_db().contains(key, prefix)
	}
}

impl<'a, H: Hasher, T> HashDBRef<H, T> for &'a mut dyn HashDBComplexDyn<H, T> {
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T> {
		self.as_hash_db().get(key, prefix)
	}

	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool {
		self.as_hash_db().contains(key, prefix)
	}
}

// TODO this using a buffer is bad (we should switch
// binary hasher to use slice as input (or be able to))
pub struct ComplexLayoutIterValues<'a, HO, I> {
	nb_children: usize, 
	children: I, 
	node: &'a [u8],
	_ph: PhantomData<HO>,
}
/*
code snippet for children iter:
ComplexLayoutIterValues::new(nb_children, children, value)
				.map(|(is_defined, v)| {
					debug_assert!(is_defined);
					v
				});
code snippet for proof
			let iter = ComplexLayoutIterValues::new(nb_children, children, value)
				.zip(iter_key)
				.filter_map(|((is_defined, hash), key)| if is_defined {
					Some((key, hash))
				} else {
					None
				});
*/	

impl<'a, HO: Default, I> ComplexLayoutIterValues<'a, HO, I> {
	pub fn new(nb_children: usize, children: I, node: &'a[u8]) -> Self {
		ComplexLayoutIterValues {
			nb_children,
			children,
			node,
			_ph: PhantomData,
		}
	}
}

impl<'a, HO: AsMut<[u8]> + Default, I: Iterator<Item = &'a Range<usize>>> Iterator for ComplexLayoutIterValues<'a, HO, I> {
	type Item = Option<HO>;

	fn next(&mut self) -> Option<Self::Item> {
		if let Some(range) = self.children.next() {
			let range_len = range.len();
			if range_len == 0 {
				// this is for undefined proof hash
				return None;
			}
			let mut dest = HO::default();
			dest.as_mut()[..range_len].copy_from_slice(&self.node[range.clone()]);
			/* inherent to default?? TODO consider doing it, but if refacto
			 * this will be part of hasher (when run with slice as input)
			 * for i in range_len..dest.len() {
				dest[i] = 0;
			}*/
			// TODO the input iterator is HO but could really be &HO, would need some
			// change on trie_root to.
			Some(Some(dest))
		} else {
			None
		}
	}
}
