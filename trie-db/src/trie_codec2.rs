// Copyright 2019, 2021 Parity Technologies
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

//! Compact encoding/decoding functions for partial Merkle-Patricia tries.
//!
//! A partial trie is a subset of the nodes in a complete trie, which can still be used to
//! perform authenticated lookups on a subset of keys. A naive encoding is the set of encoded nodes
//! in the partial trie. This, however, includes redundant hashes of other nodes in the partial
//! trie which could be computed directly.
//! The compact encoding focus on content only by encoding an ordered sequence of key
//! construction info, present values in the trie and preimage of missing content in the
//! partial trie.

use crate::{
	nibble_ops::NIBBLE_LENGTH,
	node::{NodeHandlePlan, NodePlan, OwnedNode, ValuePlan},
	rstd::{rc::Rc, vec::Vec},
	CError, CompactDecoderError, DBValue, NibbleVec, Result, TrieDB,
	TrieDBNodeIterator, TrieError, TrieHash, TrieLayout,
};
use codec::{Decode, Encode};
use hash_db::{HashDB, Prefix};

/// Representation of each encoded action
/// for building the proof.
/// TODO ref variant for encoding ??
/// TODO V as parameter is a bit useless, here to help adapt
/// to iter_build, but likely the iter_build api should be
/// change instead.
#[derive(Encode, Decode, Debug)]
pub(crate) enum Op<H, V> {
	// key content followed by a mask for last byte.
	// If mask erase some content the content need to
	// be set at 0 (or error).
	// Two consecutive `KeyPush` are invalid.
	KeyPush(Vec<u8>, u8), // TODO could use BackingByteVec (but Vec for new as it scale encode)
	// Last call to pop is implicit (up to root), defining
	// one will result in an error.
	// Two consecutive `KeyPush` are invalid.
	// TODO should be compact encoding of number.
	KeyPop(u16),
	// u8 is child index, shorthand for key push one nibble followed by key pop.
	HashChild(Enc<H>, u8),
	// All value variant are only after a `KeyPush` or at first position.
	HashValue(Enc<H>),
	Value(V),
	// Only to build old value
	// representation if trie layout
	// MAX_INLINE_VALUE was switched.
	ValueForceInline(V),
	// Same
	ValueForceHashed(V),
	// This is not strictly necessary, only if the proof is not sized, otherwhise if we know the
	// stream will end it can be skipped.
	EndProof,
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Enc<H>(pub H);

impl<H: AsRef<[u8]>> Encode for Enc<H> {
	fn size_hint(&self) -> usize {
		self.0.as_ref().len()
	}

	fn encoded_size(&self) -> usize {
		self.0.as_ref().len()
	}

	fn encode_to<T: codec::Output + ?Sized>(&self, dest: &mut T) {
		dest.write(self.0.as_ref())
	}
}

impl<H: AsMut<[u8]> + Default> Decode for Enc<H> {
	fn decode<I: codec::Input>(input: &mut I) -> core::result::Result<Self, codec::Error> {
		let mut dest = H::default();
		input.read(dest.as_mut())?;
		Ok(Enc(dest))
	}
}

/// Detached value if included does write a reserved header,
/// followed by node encoded with 0 length value and the value
/// as a standalone vec.
fn detached_value<L: TrieLayout>(
	value: &ValuePlan,
	node_data: &[u8],
	node_prefix: Prefix,
	val_fetcher: &TrieDBNodeIterator<L>,
) -> Option<Vec<u8>> {
	let fetched;
	match value {
		ValuePlan::Node(hash_plan) => {
			if let Ok(value) = val_fetcher.fetch_value(&node_data[hash_plan.clone()], node_prefix) {
				fetched = value;
			} else {
				return None
			}
		},
		_ => return None,
	}
	Some(fetched)
}

/// Generates a compact representation of the partial trie stored in the given DB. The encoding
/// is a vector of mutated trie nodes with those child references omitted. The mutated trie nodes
/// are listed in pre-order traversal order so that the full nodes can be efficiently
/// reconstructed recursively.
///
/// This function makes the assumption that all child references in an inline trie node are inline
/// references.
pub fn encode_compact<L>(db: &TrieDB<L>) -> Result<Vec<u8>, TrieHash<L>, CError<L>>
where
	L: TrieLayout,
{
	let mut output = Vec::new();

	// TrieDBNodeIterator guarantees that:
	// - It yields at least one node.
	// - The first node yielded is the root node with an empty prefix and is not inline.
	// - The prefixes yielded are in strictly increasing lexographic order.
	let mut iter = TrieDBNodeIterator::new(db)?;

	let mut at = NibbleVec::new();
	let mut cursor_depth = 0;

	let mut stack =
		Vec::<(Rc<OwnedNode<DBValue>>, [Option<NodeHandlePlan>; NIBBLE_LENGTH], usize)>::new();

	let key_pop = |cursor_depth: &mut usize, depth, output: &mut _| {
		let op = Op::<TrieHash<L>, DBValue>::KeyPop((*cursor_depth - depth) as u16);
		println!("{:?}", op);
		op.encode_to(output);
		*cursor_depth = depth;
	};

	let key_push = |cursor_depth: &mut usize, prefix: &NibbleVec, output: &mut _| {
		let descend = prefix.right_of(*cursor_depth);
		let op = Op::<TrieHash<L>, DBValue>::KeyPush(descend.0, descend.1);
		println!("{:?}", op);
		op.encode_to(output);
		*cursor_depth = prefix.len();
	};

	while let Some(item) = iter.next() {
		//		println!("{:?}", item);
		match item {
			Ok((mut prefix, _node_hash, node)) => {
				let common_depth = crate::nibble_ops::biggest_depth(at.inner(), prefix.inner());
				let common_depth = if common_depth > at.len() {
					at.len()
				} else if common_depth > prefix.len() {
					prefix.len()
				} else {
					common_depth
				};

				if common_depth < at.len() {
					if at.len() > cursor_depth {
						key_push(&mut cursor_depth, &at, &mut output);
					}

					while stack.last().map(|s| s.2 > common_depth).unwrap_or(false) {
						if let Some((node, children, depth)) = stack.pop() {
							let mut pop_written = (cursor_depth - depth) == 0;
							let mut ix = 0;
							while ix < NIBBLE_LENGTH as usize {
								if let Some(NodeHandlePlan::Hash(plan)) = &children[ix] {
									if !pop_written {
										key_pop(&mut cursor_depth, depth, &mut output);
										pop_written = true;
									}
									// inline are ignored: will be processed by next iterator calls
									let mut hash: TrieHash<L> = Default::default();
									hash.as_mut().copy_from_slice(&node.data()[plan.clone()]);
									let op = Op::<_, DBValue>::HashChild(Enc(hash), ix as u8);
									println!("{:?}", op);
									op.encode_to(&mut output);
								}
								ix += 1;
							}
						}
					}
					if cursor_depth > common_depth {
						key_pop(&mut cursor_depth, common_depth, &mut output);
					}
				}

				if let Some((_node, children, _depth)) = stack.last_mut() {
					let at_child = prefix.at(prefix.len() - 1) as usize;
					debug_assert!(children[at_child].is_some());
					children[at_child] = None;
				}

				let node_plan = node.node_plan();

				match &node_plan {
					NodePlan::Leaf { partial, .. } | NodePlan::NibbledBranch { partial, .. } => {
						let partial = partial.build(node.data());
						prefix.append_optional_slice_and_nibble(Some(&partial), None);
					},
					_ => (),
				}
				if let Some(value) = match &node_plan {
					// would just need to stack empty array of child.
					NodePlan::Extension { .. } => unimplemented!(),
					NodePlan::Leaf { value, .. } => Some(value.clone()),
					NodePlan::NibbledBranch { value, .. } | NodePlan::Branch { value, .. } =>
						value.clone(),
					_ => None,
				} {
					if common_depth < cursor_depth {
						key_pop(&mut cursor_depth, common_depth, &mut output);
					}
					key_push(&mut cursor_depth, &prefix, &mut output);

					let op = match &value {
						ValuePlan::Inline(range) => {
							let value = node.data()[range.clone()].to_vec();
							if Some(value.len() as u32) >= L::MAX_INLINE_VALUE {
								// should be a external node, force it
								Op::<TrieHash<L>, DBValue>::ValueForceInline(value)
							} else {
								Op::<TrieHash<L>, DBValue>::Value(value)
							}
						},
						ValuePlan::Node(hash) => {
							if let Some(value) =
								detached_value(&value, node.data(), prefix.as_prefix(), &iter)
							{
								if Some(value.len() as u32) < L::MAX_INLINE_VALUE {
									// should be inline, may result from change of threshold value
									Op::<TrieHash<L>, DBValue>::ValueForceHashed(value)
								} else {
									Op::<TrieHash<L>, DBValue>::Value(value)
								}
							} else {
								let hash_bytes = &node.data()[hash.clone()];
								let mut hash = TrieHash::<L>::default();
								hash.as_mut().copy_from_slice(hash_bytes);
								Op::<TrieHash<L>, DBValue>::HashValue(Enc(hash))
							}
						},
					};
					println!("{:?}", op);
					op.encode_to(&mut output);
				}
				match node_plan {
					// would just need to stack empty array of child.
					NodePlan::Extension { .. } => unimplemented!(),
					NodePlan::Empty { .. } => (),
					NodePlan::Leaf { .. } => (),
					NodePlan::NibbledBranch { children, .. } |
					NodePlan::Branch { children, .. } => {
						// node is rc
						stack.push((node.clone(), children.clone(), prefix.len()));
						at = prefix;
					},
				}
			},

			Err(err) => match *err {
				// If we hit an IncompleteDatabaseError, just ignore it and continue encoding the
				// incomplete trie. This encoding must support partial tries, which can be used for
				// space-efficient storage proofs.
				TrieError::IncompleteDatabase(_) => {},
				_ => return Err(err),
			},
		}
	}

	if at.len() > cursor_depth {
		key_push(&mut cursor_depth, &at, &mut output);
	}

	while let Some((node, children, depth)) = stack.pop() {
		let mut pop_written = (cursor_depth - depth) == 0;
		let mut ix = 0;
		while ix < NIBBLE_LENGTH as usize {
			if let Some(NodeHandlePlan::Hash(plan)) = &children[ix] {
				if !pop_written {
					key_pop(&mut cursor_depth, depth, &mut output);
					pop_written = true;
				}
				// inline are ignored: will be processed by next iterator calls
				let mut hash: TrieHash<L> = Default::default();
				hash.as_mut().copy_from_slice(&node.data()[plan.clone()]);
				let op = Op::<_, DBValue>::HashChild(Enc(hash), ix as u8);
				println!("{:?}", op);
				op.encode_to(&mut output);
			}
			ix += 1;
		}
	}

	// TODO make it optional from parameter
	let op = Op::<TrieHash<L>, DBValue>::EndProof;
	println!("{:?}", op);
	op.encode_to(&mut output);

	Ok(output)
}

/// Reconstructs a partial trie DB from a compact representation. The encoding is a vector of
/// mutated trie nodes with those child references omitted. The decode function reads them in order
/// from the given slice, reconstructing the full nodes and inserting them into the given `HashDB`.
/// It stops after fully constructing one partial trie and returns the root hash and the number of
/// nodes read. If an error occurs during decoding, there are no guarantees about which entries
/// were or were not added to the DB.
///
/// The number of nodes read may be fewer than the total number of items in `encoded`. This allows
/// one to concatenate multiple compact encodings together and still reconstruct them all.
///
/// This function assumes that all child references in an inline trie node are inline references.
pub fn decode_compact<L, DB>(
	db: &mut DB,
	mut encoded: &[u8],
) -> Result<(TrieHash<L>, usize), TrieHash<L>, CError<L>>
where
	L: TrieLayout,
	DB: HashDB<L::Hash, DBValue>,
{
	let read = &mut encoded;
	let ops_iter = core::iter::from_fn(move || {
		if read.len() != 0 {
			Some(
				Op::<TrieHash<L>, DBValue>::decode(read)
					.map_err(|_e| CompactDecoderError::DecodingFailure.into()),
			)
		} else {
			None
		}
	});
	let mut trie_builder = crate::iter_build::TrieBuilder::<L, DB>::new(db);
	crate::iter_build::trie_visit_compact::<L, _, _, _>(ops_iter, &mut trie_builder)?;

	if let Some(root) = trie_builder.root {
		Ok((root, 0))
	} else {
		Err(CompactDecoderError::DecodingFailure.into())
	}
}

/// Variant of 'decode_compact' that accept an iterator of encoded nodes as input.
/// TODO this does not make any sense as an api, we just need decode from
/// input with input being io read.
pub fn decode_compact_from_iter<'a, L, DB, I>(
	db: &mut DB,
	encoded: I,
) -> Result<(TrieHash<L>, usize), TrieHash<L>, CError<L>>
where
	L: TrieLayout,
	DB: HashDB<L::Hash, DBValue>,
	I: IntoIterator<Item = &'a [u8]>,
{
let ops_iter = encoded.into_iter().map(|mut buf|
				Op::<TrieHash<L>, DBValue>::decode(&mut buf)
					.map_err(|_e| CompactDecoderError::DecodingFailure.into()),
	);
	let mut trie_builder = crate::iter_build::TrieBuilder::<L, DB>::new(db);
	crate::iter_build::trie_visit_compact::<L, _, _, _>(ops_iter, &mut trie_builder)?;

	if let Some(root) = trie_builder.root {
		Ok((root, 0))
	} else {
		Err(CompactDecoderError::DecodingFailure.into())
	}
}
