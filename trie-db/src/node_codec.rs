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

use crate::{
	node::{Node, NodePlan, Value},
	rstd::{borrow::Borrow, hash, vec::Vec, Error},
	ChildReference, MaybeDebug, NibbleOps,
};

use crate::rstd::{borrow::Borrow, hash, vec::Vec, Error};

/// Representation of a nible slice (right aligned).
/// It contains a right aligned padded first byte (first pair element is the number of nibbles
/// (0 to max nb nibble - 1), second pair element is the padded nibble), and a slice over
/// the remaining bytes.
pub type Partial<'a> = ((u8, u8), &'a [u8]);

/// Trait for trie node encoding/decoding.
/// Uses a type parameter to allow registering
/// positions without colling decode plan.
pub trait NodeCodec<const N: usize>: Sized {
	/// Escape header byte sequence to indicate next node is a
	/// branch or leaf with hash of value, followed by the value node.
	const ESCAPE_HEADER: Option<u8> = None;

	/// Codec error type.
	type Error: Error;

	type Nibble: NibbleOps;

	/// Output type of encoded node hasher.
	type HashOut: AsRef<[u8]>
		+ AsMut<[u8]>
		+ Default
		+ MaybeDebug
		+ PartialEq
		+ Eq
		+ hash::Hash
		+ Send
		+ Sync
		+ Clone
		+ Copy;

	/// Get the hashed null node.
	fn hashed_null_node() -> Self::HashOut;

	/// Decode bytes to a `NodePlan`. Returns `Self::E` on failure.
	fn decode_plan(data: &[u8]) -> Result<NodePlan<Self::Nibble, N>, Self::Error>;

	/// Decode bytes to a `Node`. Returns `Self::E` on failure.
	fn decode<'a>(data: &'a [u8]) -> Result<Node<'a, Self::Nibble, N>, Self::Error> {
		Ok(Self::decode_plan(data)?.build(data))
	}

	/// Check if the provided bytes correspond to the codecs "empty" node.
	fn is_empty_node(data: &[u8]) -> bool;

	/// Returns an encoded empty node.
	fn empty_node() -> &'static [u8];

	/// Returns an encoded leaf node
	///
	/// Note that number_nibble is the number of element of the iterator
	/// it can possibly be obtain by `Iterator` `size_hint`, but
	/// for simplicity it is used directly as a parameter.
	fn leaf_node(partial: impl Iterator<Item = u8>, number_nibble: usize, value: Value) -> Vec<u8>;

	/// Returns an encoded extension node
	///
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
		value: Option<Value>,
	) -> Vec<u8>;

	/// Returns an encoded branch node with a possible partial path.
	/// `number_nibble` is the partial path length as in `extension_node`.
	fn branch_node_nibbled(
		partial: impl Iterator<Item = u8>,
		number_nibble: usize,
		children: impl Iterator<Item = impl Borrow<Option<ChildReference<Self::HashOut>>>>,
		value: Option<Value>,
	) -> Vec<u8>;
}

/// Bitmap encoder for the number of children nodes.
pub trait BitMap: Sized {
	/// length to encode the bitmap
	const ENCODED_LEN: usize;
	/// Codec error type.
	type Error: Error;

	/// Codec buffer to use.
	type Buffer: AsRef<[u8]> + AsMut<[u8]> + Default;

	/// Decode bitmap from its encoded full slice.
	fn decode(data: &[u8]) -> Result<Self, Self::Error>;

	/// Return wether the bitmap registered a value for a branch
	/// child index.
	fn value_at(&self, i: usize) -> bool;

	/// Encode bitmap, output slice must be of right length.
	fn encode<I: Iterator<Item = bool>>(has_children: I, output: &mut [u8]);
}
