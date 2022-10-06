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
	node::{Node, NodeHandle, NodeHandlePlan, NodePlan, OwnedNode, ValuePlan},
	rstd::{boxed::Box, convert::TryInto, marker::PhantomData, rc::Rc, result, vec, vec::Vec},
	CError, ChildReference, DBValue, NibbleVec, NodeCodec, Result, TrieDB, TrieDBNodeIterator,
	TrieError, TrieHash, TrieLayout,
};
use codec::{Decode, Encode};
use hash_db::{HashDB, Prefix};

/// Representation of each encoded action
/// for building the proof.
/// TODO ref variant for encoding ??
#[derive(Encode, Decode, Debug)]
enum Op<H> {
	// key content followed by a mask for last byte.
	// If mask erase some content the content need to
	// be set at 0 (or error).
	KeyPush(Vec<u8>, u8), // TODO could use BackingByteVec (but Vec for new as it scale encode)
	// Last call to pop is implicit (up to root), defining
	// one will result in an error.
	// TODO should be compact encoding of number.
	KeyPop(u16),
	// u8 is child index, shorthand for key push one nibble followed by key pop.
	HashChild(Enc<H>, u8),
	HashValue(Enc<H>),
	Value(DBValue),
	// Only to build old value
	// representation if trie layout
	// MAX_INLINE_VALUE was switched.
	ValueForceInline(DBValue),
	// Same
	ValueForceHashed(DBValue),
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
	fn decode<I: codec::Input>(input: &mut I) -> std::result::Result<Self, codec::Error> {
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

fn delta_prefix(new: &NibbleVec, old: &NibbleVec) -> (Vec<u8>, u8) {
	// no check for common prefix this is part of iterator property.
	new.right_of(old.len())
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
	let mut prev = NibbleVec::new();

	let mut stack =
		Vec::<(usize, Rc<OwnedNode<DBValue>>, [Option<NodeHandlePlan>; NIBBLE_LENGTH])>::new();
	while let Some(item) = iter.next() {
		match item {
			Ok((prefix, _node_hash, node)) => {
				if prefix.len() > at.len() {
					let descend = delta_prefix(&prefix, &at);
					let at_descend = prefix.at(at.len());
					if let Some((ix, node, children)) = stack.last_mut() {
						while *ix < at_descend as usize {
							if let Some(NodeHandlePlan::Hash(plan)) = &children[*ix] {
								// inline are ignored: will be processed by next iterator calls
								let mut hash: TrieHash<L> = Default::default();
								hash.as_mut().copy_from_slice(&node.data()[plan.clone()]);
								let op = Op::HashChild(Enc(hash), *ix as u8);
								println!("{:?}", op);
								op.encode_to(&mut output);
							}
							*ix += 1;
						}
						*ix += 1;
					}
					// TODO Note that we should have variant of iterator (or just a next fn)
					// that pass key delta, not build key.
					let op = Op::<TrieHash<L>>::KeyPush(descend.0, descend.1);
					println!("{:?}", op);
					op.encode_to(&mut output);
				} else {
					let op = Op::<TrieHash<L>>::KeyPop((at.len() - prefix.len()) as u16);
					println!("{:?}", op);
					op.encode_to(&mut output);
					if let Some((mut ix, node, children)) = stack.pop() {
						while ix < NIBBLE_LENGTH as usize {
							if let Some(NodeHandlePlan::Hash(plan)) = &children[ix] {
								// inline are ignored: will be processed by next iterator calls
								let mut hash: TrieHash<L> = Default::default();
								hash.as_mut().copy_from_slice(&node.data()[plan.clone()]);
								let op = Op::HashChild(Enc(hash), ix as u8);
								println!("{:?}", op);
								op.encode_to(&mut output);
							}
							ix += 1;
						}
					}
				}
				let node_plan = node.node_plan();
				if let Some(value) = match &node_plan {
					// would just need to stack empty array of child.
					NodePlan::Extension { .. } => unimplemented!(),
					NodePlan::Leaf { value, .. } => Some(value.clone()),
					NodePlan::NibbledBranch { value, .. } | NodePlan::Branch { value, .. } =>
						value.clone(),
					_ => None,
				} {
					let op = match &value {
						ValuePlan::Inline(range) => {
							let value = node.data()[range.clone()].to_vec();
							if Some(value.len() as u32) >= L::MAX_INLINE_VALUE {
								// should be a external node, force it
								Op::<TrieHash<L>>::ValueForceInline(value)
							} else {
								Op::<TrieHash<L>>::Value(value)
							}
						},
						ValuePlan::Node(hash) => {
							if let Some(value) =
								detached_value(&value, node.data(), prefix.as_prefix(), &iter)
							{
								if Some(value.len() as u32) < L::MAX_INLINE_VALUE {
									// should be inline, may result from change of threshold value
									Op::<TrieHash<L>>::ValueForceHashed(value)
								} else {
									Op::<TrieHash<L>>::Value(value)
								}
							} else {
								let hash_bytes = &node.data()[hash.clone()];
								let mut hash = TrieHash::<L>::default();
								hash.as_mut().copy_from_slice(hash_bytes);
								Op::<TrieHash<L>>::HashValue(Enc(hash))
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
					NodePlan::Leaf { .. } => {
						// TODO pop could sometime just be infered (eg if there is hash in parent
						// branch.
						let op = Op::<TrieHash<L>>::KeyPop((prefix.len() - at.len()) as u16);
						println!("{:?}", op);
						op.encode_to(&mut output);
					},
					NodePlan::NibbledBranch { children, .. } |
					NodePlan::Branch { children, .. } => {
						// node is rc
						stack.push((0, node.clone(), children.clone()));
						at = prefix;
					},
				}
				/*	match node.node_plan() {
					NodePlan::Empty => (),
					NodePlan::Leaf { value, .. } => {},
					NodePlan::Extension { .. } => (),
					NodePlan::NibbledBranch { value: Some(value), .. } |
					NodePlan::Branch { value: Some(value), .. } => {},
					NodePlan::NibbledBranch { value: None, .. } |
					NodePlan::Branch { value: None, .. } => (NIBBLE_LENGTH, None),
				};*/
			},

			/*
							stack.push(EncoderStackEntry {
								prefix,
								node,
								child_index: 0,
								omit_children: vec![false; children_len],
								omit_value: detached_value.is_some(),
								output_index: output.len(),
								_marker: PhantomData::default(),
							});
							// Insert a placeholder into output which will be replaced when this new entry is
							// popped from the stack.
							output.push(Vec::new());
							if let Some(value) = detached_value {
								output.push(value);
							}
						},
			*/
			Err(err) => match *err {
				// If we hit an IncompleteDatabaseError, just ignore it and continue encoding the
				// incomplete trie. This encoding must support partial tries, which can be used for
				// space-efficient storage proofs.
				TrieError::IncompleteDatabase(_) => {},
				_ => return Err(err),
			},
		}
	}

	while let Some((mut ix, node, children)) = stack.pop() {
		while ix < NIBBLE_LENGTH as usize {
			if let Some(NodeHandlePlan::Hash(plan)) = &children[ix] {
				// inline are ignored: will be processed by next iterator calls
				let mut hash: TrieHash<L> = Default::default();
				hash.as_mut().copy_from_slice(&node.data()[plan.clone()]);
				let op = Op::HashChild(Enc(hash), ix as u8);
				println!("{:?}", op);
				op.encode_to(&mut output);
			}
			ix += 1;
		}
	}

	Ok(output)
}

struct DecoderStackEntry<'a, C: NodeCodec> {
	node: Node<'a>,
	/// The next entry in the stack is a child of the preceding entry at this index. For branch
	/// nodes, the index is in [0, NIBBLE_LENGTH] and for extension nodes, the index is in [0, 1].
	child_index: usize,
	/// The reconstructed child references.
	children: Vec<Option<ChildReference<C::HashOut>>>,
	/// A value attached as a node. The node will need to use its hash as value.
	attached_value: Option<&'a [u8]>,
	_marker: PhantomData<C>,
}

impl<'a, C: NodeCodec> DecoderStackEntry<'a, C> {
	/// Advance the child index until either it exceeds the number of children or the child is
	/// marked as omitted. Omitted children are indicated by an empty inline reference. For each
	/// child that is passed over and not omitted, copy over the child reference from the node to
	/// this entries `children` list.
	///
	/// Returns true if the child index is past the last child, meaning the `children` references
	/// list is complete. If this returns true and the entry is an extension node, then
	/// `children[0]` is guaranteed to be Some.
	fn advance_child_index(&mut self) -> Result<bool, C::HashOut, C::Error> {
		match self.node {
			Node::Extension(_, child) if self.child_index == 0 => {
				match child {
					NodeHandle::Inline(data) if data.is_empty() => return Ok(false),
					_ => {
						let child_ref = child.try_into().map_err(|hash| {
							Box::new(TrieError::InvalidHash(C::HashOut::default(), hash))
						})?;
						self.children[self.child_index] = Some(child_ref);
					},
				}
				self.child_index += 1;
			},
			Node::Branch(children, _) | Node::NibbledBranch(_, children, _) => {
				while self.child_index < NIBBLE_LENGTH {
					match children[self.child_index] {
						Some(NodeHandle::Inline(data)) if data.is_empty() => return Ok(false),
						Some(child) => {
							let child_ref = child.try_into().map_err(|hash| {
								Box::new(TrieError::InvalidHash(C::HashOut::default(), hash))
							})?;
							self.children[self.child_index] = Some(child_ref);
						},
						None => {},
					}
					self.child_index += 1;
				}
			},
			_ => {},
		}
		Ok(true)
	}

	/// Push the partial key of this entry's node (including the branch nibble) to the given
	/// prefix.
	fn push_to_prefix(&self, prefix: &mut NibbleVec) {
		match self.node {
			Node::Empty => {},
			Node::Leaf(partial, _) | Node::Extension(partial, _) => {
				prefix.append_partial(partial.right());
			},
			Node::Branch(_, _) => {
				prefix.push(self.child_index as u8);
			},
			Node::NibbledBranch(partial, _, _) => {
				prefix.append_partial(partial.right());
				prefix.push(self.child_index as u8);
			},
		}
	}

	/// Pop the partial key of this entry's node (including the branch nibble) from the given
	/// prefix.
	fn pop_from_prefix(&self, prefix: &mut NibbleVec) {
		match self.node {
			Node::Empty => {},
			Node::Leaf(partial, _) | Node::Extension(partial, _) => {
				prefix.drop_lasts(partial.len());
			},
			Node::Branch(_, _) => {
				prefix.pop();
			},
			Node::NibbledBranch(partial, _, _) => {
				prefix.pop();
				prefix.drop_lasts(partial.len());
			},
		}
	}

	/// Reconstruct the encoded full trie node from the node and the entry's child references.
	///
	/// Preconditions:
	/// - if node is an extension node, then `children[0]` is Some.
	fn encode_node(self, attached_hash: Option<&[u8]>) -> Vec<u8> {
		let attached_hash = attached_hash.map(|h| crate::node::Value::Node(h));
		match self.node {
			Node::Empty => C::empty_node().to_vec(),
			Node::Leaf(partial, value) =>
				C::leaf_node(partial.right_iter(), partial.len(), attached_hash.unwrap_or(value)),
			Node::Extension(partial, _) => C::extension_node(
				partial.right_iter(),
				partial.len(),
				self.children[0].expect("required by method precondition; qed"),
			),
			Node::Branch(_, value) => C::branch_node(
				self.children.into_iter(),
				if attached_hash.is_some() { attached_hash } else { value },
			),
			Node::NibbledBranch(partial, _, value) => C::branch_node_nibbled(
				partial.right_iter(),
				partial.len(),
				self.children.iter(),
				if attached_hash.is_some() { attached_hash } else { value },
			),
		}
	}
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
//
/// This function makes the assumption that all child references in an inline trie node are inline
/// references.
pub fn decode_compact<L, DB>(
	db: &mut DB,
	encoded: &[u8],
) -> Result<(TrieHash<L>, usize), TrieHash<L>, CError<L>>
where
	L: TrieLayout,
	DB: HashDB<L::Hash, DBValue>,
{
	unimplemented!()
	//	decode_compact_from_iter::<L, DB, _>(db, encoded.iter().map(Vec::as_slice))
}

/// Variant of 'decode_compact' that accept an iterator of encoded nodes as input.
pub fn decode_compact_from_iter<'a, L, DB, I>(
	db: &mut DB,
	encoded: I,
) -> Result<(TrieHash<L>, usize), TrieHash<L>, CError<L>>
where
	L: TrieLayout,
	DB: HashDB<L::Hash, DBValue>,
	I: IntoIterator<Item = &'a [u8]>,
{
	// The stack of nodes through a path in the trie. Each entry is a child node of the preceding
	// entry.
	let mut stack: Vec<DecoderStackEntry<L::Codec>> = Vec::new();

	// The prefix of the next item to be read from the slice of encoded items.
	let mut prefix = NibbleVec::new();

	let mut iter = encoded.into_iter().enumerate();
	while let Some((i, encoded_node)) = iter.next() {
		let mut attached_node = 0;
		if let Some(header) = L::Codec::ESCAPE_HEADER {
			if encoded_node.starts_with(&[header]) {
				attached_node = 1;
			}
		}
		let node = L::Codec::decode(&encoded_node[attached_node..])
			.map_err(|err| Box::new(TrieError::DecoderError(<TrieHash<L>>::default(), err)))?;

		let children_len = match node {
			Node::Empty | Node::Leaf(..) => 0,
			Node::Extension(..) => 1,
			Node::Branch(..) | Node::NibbledBranch(..) => NIBBLE_LENGTH,
		};
		let mut last_entry = DecoderStackEntry {
			node,
			child_index: 0,
			children: vec![None; children_len],
			attached_value: None,
			_marker: PhantomData::default(),
		};

		if attached_node > 0 {
			// Read value
			if let Some((_, fetched_value)) = iter.next() {
				last_entry.attached_value = Some(fetched_value);
			} else {
				return Err(Box::new(TrieError::IncompleteDatabase(<TrieHash<L>>::default())))
			}
		}

		loop {
			if !last_entry.advance_child_index()? {
				last_entry.push_to_prefix(&mut prefix);
				stack.push(last_entry);
				break
			}

			// Since `advance_child_index` returned true, the preconditions for `encode_node` are
			// satisfied.
			let hash = last_entry
				.attached_value
				.as_ref()
				.map(|value| db.insert(prefix.as_prefix(), value));
			let node_data = last_entry.encode_node(hash.as_ref().map(|h| h.as_ref()));
			let node_hash = db.insert(prefix.as_prefix(), node_data.as_ref());

			if let Some(entry) = stack.pop() {
				last_entry = entry;
				last_entry.pop_from_prefix(&mut prefix);
				last_entry.children[last_entry.child_index] = Some(ChildReference::Hash(node_hash));
				last_entry.child_index += 1;
			} else {
				return Ok((node_hash, i + 1))
			}
		}
	}

	Err(Box::new(TrieError::IncompleteDatabase(<TrieHash<L>>::default())))
}
