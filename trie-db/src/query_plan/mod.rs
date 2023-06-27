// Copyright 2023, 2023 Parity Technologies
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

//! Iterate on multiple values following a specific query plan.
//! Can be use on a trie db to register a proof, or
//! be use on a proof directly.
//! When use on a proof, process can be interrupted, checked and
//! restore (at the cost of additional hashes in proof).
//!
//! Because nodes are guaranted to be accessed only once and recorded
//! in proof, we do not use a cache (or only the node caching part).

use core::marker::PhantomData;

use crate::{
	content_proof::Op,
	nibble::{nibble_ops, nibble_ops::NIBBLE_LENGTH, LeftNibbleSlice, NibbleSlice},
	node::{NodeHandle, NodePlan, OwnedNode, Value},
	node_codec::NodeCodec,
	proof::VerifyError,
	rstd::{
		borrow::{Borrow, Cow},
		boxed::Box,
		cmp::*,
		result::Result,
	},
	CError, ChildReference, DBValue, NibbleVec, Trie, TrieDB, TrieHash, TrieLayout,
};
use hash_db::Hasher;
pub use record::{record_query_plan, HaltedStateRecord, Recorder};
pub use verify::verify_query_plan_iter;
use verify::HaltedStateCheckNode;
pub use verify_content::verify_query_plan_iter_content;
use verify_content::HaltedStateCheckContent;

mod record;
mod verify;
mod verify_content;

/// Item to query, in memory.
#[derive(Default, Clone, Debug)]
pub struct InMemQueryPlanItem {
	key: Vec<u8>,
	hash_only: bool,
	//	hash_only: bool, TODO implement
	as_prefix: bool,
}

impl InMemQueryPlanItem {
	/// Create new item.
	pub fn new(key: Vec<u8>, hash_only: bool, as_prefix: bool) -> Self {
		Self { key, hash_only, as_prefix }
	}
	/// Get ref.
	pub fn as_ref(&self) -> QueryPlanItem {
		QueryPlanItem { key: &self.key, hash_only: self.hash_only, as_prefix: self.as_prefix }
	}
}

/// Item to query.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryPlanItem<'a> {
	pub key: &'a [u8],
	pub hash_only: bool,
	pub as_prefix: bool,
}

impl<'a> QueryPlanItem<'a> {
	fn before(&self, other: &Self) -> (bool, usize) {
		let (common_depth, ordering) = nibble_ops::biggest_depth_and_order(&self.key, &other.key);

		(
			match ordering {
				Ordering::Less => {
					if self.as_prefix {
						// do not allow querying content inside a prefix
						!other.key.starts_with(self.key)
					} else {
						true
					}
				},
				Ordering::Greater | Ordering::Equal => false,
			},
			common_depth,
		)
	}

	fn to_owned(&self) -> InMemQueryPlanItem {
		InMemQueryPlanItem {
			key: self.key.to_vec(),
			hash_only: self.hash_only,
			as_prefix: self.as_prefix,
		}
	}
}

/// Query plan in memory.
#[derive(Clone, Debug)]
pub struct InMemQueryPlan {
	pub items: Vec<InMemQueryPlanItem>,
	pub kind: ProofKind,
	// TODO rem
	pub ignore_unordered: bool,
}

/// Iterator as type of mapped slice iter is very noisy.
pub struct QueryPlanItemIter<'a>(&'a Vec<InMemQueryPlanItem>, usize);

impl<'a> Iterator for QueryPlanItemIter<'a> {
	type Item = QueryPlanItem<'a>;

	fn next(&mut self) -> Option<Self::Item> {
		if self.1 >= self.0.len() {
			return None
		}
		self.1 += 1;
		Some(self.0[self.1 - 1].as_ref())
	}
}

impl InMemQueryPlan {
	/// Get ref.
	pub fn as_ref(&self) -> QueryPlan<QueryPlanItemIter> {
		QueryPlan {
			items: QueryPlanItemIter(&self.items, 0),
			kind: self.kind,
			ignore_unordered: self.ignore_unordered,
			_ph: PhantomData,
		}
	}
}

/// Query plan.
pub struct QueryPlan<'a, I> {
	pub items: I,
	pub ignore_unordered: bool,
	pub kind: ProofKind,
	pub _ph: PhantomData<&'a ()>,
}

/// Different proof support.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProofKind {
	/// Proof is a sequence of fully encoded node, this is not
	/// size efficient but allows streaming a proof better, since
	/// the consumer can halt at the first invalid node.
	FullNodes,

	/// Proof got its accessed hash and value removed (same scheme
	/// as in trie_codec.rs ordering is same as full node, from
	/// root and in lexicographic order).
	///
	/// Checking this proof requires to read it fully. When stopping
	/// recording, the partial proof stays valid, but it will
	/// contains hashes that would not be needed if creating the
	/// proof at once.
	CompactNodes,

	/// Content oriented proof, no nodes are written, just a
	/// sequence of accessed by lexicographical order as described
	/// in content_proof::Op.
	/// As with compact node, checking validity of proof need to
	/// load the full proof or up to the halt point.
	CompactContent,
}

/*
impl ProofKind {
	// Do we need to record child hash and inline value individually.
	fn record_inline(&self) -> bool {
		match self {
			ProofKind::FullNodes | ProofKind::CompactNodes => false,
			ProofKind::CompactContent => true,
		}
	}
}
*/

#[derive(Default, Clone, Copy)]
struct Bitmap(u16);

pub(crate) trait BitmapAccess: Copy {
	fn at(&self, i: usize) -> bool;
}

impl BitmapAccess for Bitmap {
	fn at(&self, i: usize) -> bool {
		self.0 & (1u16 << i) != 0
	}
}

impl<'a> BitmapAccess for &'a [bool] {
	fn at(&self, i: usize) -> bool {
		self[i]
	}
}

impl Bitmap {
	fn set(&mut self, i: usize, v: bool) {
		if v {
			self.0 |= 1u16 << i
		} else {
			self.0 &= !(1u16 << i)
		}
	}
}

#[derive(Clone)]
struct StackedNodeRecord {
	/// Node in memory content.
	node: OwnedNode<DBValue>,
	/// Flags indicating whether each child is omitted (accessed) in the encoded node.
	/// For some encoding, it also record if the child has already been written.
	accessed_children_node: Bitmap,
	/// Skip value if value node is after.
	accessed_value_node: bool,
	/// Depth of node in nibbles (actual depth of an attached value (post partial)).
	depth: usize,
	/// Next descended child, can also be use to get node position in parent
	/// (this minus one).
	next_descended_child: u8,
	/// Is the node inline.
	is_inline: bool,
}

/// Allows sending proof recording as it is produced.
pub trait RecorderOutput {
	/// Append bytes.
	fn write_bytes(&mut self, bytes: &[u8]);

	/// Bytes buf len.
	fn buf_len(&self) -> usize;

	/// Append a delimited sequence of bytes (usually a node).
	fn write_entry(&mut self, bytes: Cow<[u8]>);
}

/// Simple in memory recorder.
/// Depending on type of proof, nodes or buffer should
/// be used.
/// Sequence is guaranteed by pushing buffer in nodes
/// every time a node is written.
#[derive(Default)]
pub struct InMemoryRecorder {
	pub nodes: Vec<DBValue>,
	pub buffer: Vec<u8>,
}

impl RecorderOutput for InMemoryRecorder {
	fn write_bytes(&mut self, bytes: &[u8]) {
		self.buffer.extend_from_slice(bytes)
	}

	fn buf_len(&self) -> usize {
		self.buffer.len()
	}

	fn write_entry(&mut self, bytes: Cow<[u8]>) {
		if !self.buffer.is_empty() {
			self.nodes.push(core::mem::take(&mut self.buffer));
		}
		self.nodes.push(bytes.into_owned());
	}
}

/// Limits to size proof to record.
struct Limits {
	remaining_node: Option<usize>,
	remaining_size: Option<usize>,
	kind: ProofKind,
}

impl Limits {
	#[must_use]
	fn add_node(&mut self, size: usize, hash_size: usize, is_root: bool) -> bool {
		let mut res = false;
		match self.kind {
			ProofKind::CompactNodes | ProofKind::FullNodes => {
				if let Some(rem_size) = self.remaining_size.as_mut() {
					if let ProofKind::CompactNodes = self.kind {
						if !is_root {
							// remove a parent hash
							*rem_size += hash_size;
						}
					}
					if *rem_size >= size {
						*rem_size -= size;
					} else {
						*rem_size = 0;
						res = true;
					}
				}
				if let Some(rem_node) = self.remaining_node.as_mut() {
					if *rem_node > 1 {
						*rem_node -= 1;
					} else {
						*rem_node = 0;
						res = true;
					}
				}
			},
			ProofKind::CompactContent => {
				// everything is counted as a node.
				if let Some(rem_size) = self.remaining_size.as_mut() {
					if *rem_size >= size {
						*rem_size -= size;
					} else {
						*rem_size = 0;
						res = true;
					}
				}
				if let Some(rem_node) = self.remaining_node.as_mut() {
					if *rem_node > 1 {
						*rem_node -= 1;
					} else {
						*rem_node = 0;
						res = true;
					}
				}
			},
		}
		res
	}

	#[must_use]
	fn add_value(&mut self, size: usize, hash_size: usize) -> bool {
		let mut res = false;
		match self.kind {
			ProofKind::CompactNodes | ProofKind::FullNodes => {
				if let Some(rem_size) = self.remaining_size.as_mut() {
					if let ProofKind::CompactNodes = self.kind {
						// remove a parent value hash
						*rem_size += hash_size;
					}
					if *rem_size >= size {
						*rem_size -= size;
					} else {
						*rem_size = 0;
						res = true;
					}
				}
				if let Some(rem_node) = self.remaining_node.as_mut() {
					if *rem_node > 1 {
						*rem_node -= 1;
					} else {
						*rem_node = 0;
						res = true;
					}
				}
			},
			ProofKind::CompactContent => {
				unreachable!()
			},
		}
		res
	}
}

/// When process is halted keep execution state
/// to restore later.
pub enum HaltedStateCheck<'a, L: TrieLayout, C, D: SplitFirst> {
	Node(HaltedStateCheckNode<'a, L, C, D>),
	Content(HaltedStateCheckContent<'a, L, C>),
}

#[derive(Eq, PartialEq)]
enum TryStackChildResult {
	/// If there is no child to stack.
	NotStacked,
	/// Same indicating it is a branch so in case of iteration
	/// we will attempt next child.
	NotStackedBranch,
	/// Nothing stacked, this is a next child attempt that allows
	/// suspending proof registering of proof check iteration.
	Halted,
	/// Child stacked and matched of the full partial key.
	StackedFull,
	/// Child stacked but part of partial key is into.
	/// If prefix query plan item, this is part of the prefix.
	StackedInto,
	/// Child stacked but part of partial key is after.
	/// Indicate that the query plan item need to be switched.
	/// Next query plan item could still be using this stacked node (as any stacked variant).
	StackedAfter,
}

#[derive(Eq, PartialEq)]
enum ReadProofState {
	/// Iteration not started.
	NotStarted,
	/// Iterating.
	Running,
	/// Switch next item.
	SwitchQueryPlan,
	/// Proof read.
	PlanConsumed,
	/// Proof read.
	Halted,
	/// Iteration finished.
	Finished,
}

struct StackedNodeCheck<L: TrieLayout, D: SplitFirst> {
	node: ItemStackNode<D>,
	children: Vec<Option<ChildReference<TrieHash<L>>>>,
	attached_value_hash: Option<TrieHash<L>>,
	depth: usize,
	next_descended_child: u8,
}

#[derive(Clone)]
enum ValueSet<H, V> {
	None,
	Standard(V),
	HashOnly(H),
	//	ForceInline(V),
	//	ForceHashed(V),
}

impl<H, V> ValueSet<H, V> {
	fn as_ref(&self) -> Option<&V> {
		match self {
			ValueSet::Standard(v) => Some(v),
			//ValueSet::ForceInline(v) | ValueSet::ForceHashed(v) => Some(v),
			ValueSet::HashOnly(..) | ValueSet::None => None,
		}
	}
}

impl<H, V> From<Op<H, V>> for ValueSet<H, V> {
	fn from(op: Op<H, V>) -> Self {
		match op {
			Op::HashValue(hash) => ValueSet::HashOnly(hash),
			Op::Value(value) => ValueSet::Standard(value),
			//Op::ValueForceInline(value) => ValueSet::ForceInline(value),
			//Op::ValueForceHashed(value) => ValueSet::ForceHashed(value),
			_ => ValueSet::None,
		}
	}
}

struct ItemContentStack<L: TrieLayout> {
	children: Vec<Option<ChildReference<TrieHash<L>>>>,
	value: ValueSet<TrieHash<L>, Vec<u8>>,
	depth: usize,
}

impl<L: TrieLayout, D: SplitFirst> Clone for StackedNodeCheck<L, D> {
	fn clone(&self) -> Self {
		StackedNodeCheck {
			node: self.node.clone(),
			children: self.children.clone(),
			attached_value_hash: self.attached_value_hash,
			depth: self.depth,
			next_descended_child: self.next_descended_child,
		}
	}
}

impl<L: TrieLayout> Clone for ItemContentStack<L> {
	fn clone(&self) -> Self {
		ItemContentStack {
			children: self.children.clone(),
			value: self.value.clone(),
			depth: self.depth,
		}
	}
}

#[derive(Clone)]
enum ItemStackNode<D: SplitFirst> {
	Inline(OwnedNode<Vec<u8>>),
	Node(OwnedNode<D>),
}

impl<L: TrieLayout, D: SplitFirst> From<(ItemStackNode<D>, bool)> for StackedNodeCheck<L, D> {
	fn from((node, is_compact): (ItemStackNode<D>, bool)) -> Self {
		let children = if !is_compact {
			Vec::new()
		} else {
			match &node {
				ItemStackNode::Inline(_) => Vec::new(),
				ItemStackNode::Node(node) => match node.node_plan() {
					NodePlan::Empty | NodePlan::Leaf { .. } => Vec::new(),
					NodePlan::Extension { child, .. } => {
						let mut result: Vec<Option<ChildReference<TrieHash<L>>>> = vec![None; 1];
						let node_data = node.data();
						match child.build(node_data) {
							NodeHandle::Inline(data) if data.is_empty() => (),
							child => {
								use std::convert::TryInto;
								let child_ref =
									child.try_into().expect("TODO proper error and not using From");

								result[0] = Some(child_ref);
							},
						}
						result
					},
					NodePlan::Branch { children, .. } |
					NodePlan::NibbledBranch { children, .. } => {
						let mut i = 0;
						let mut result: Vec<Option<ChildReference<TrieHash<L>>>> =
							vec![None; NIBBLE_LENGTH];
						let node_data = node.data();
						while i < NIBBLE_LENGTH {
							match children[i].as_ref().map(|c| c.build(node_data)) {
								Some(NodeHandle::Inline(data)) if data.is_empty() => (),
								Some(child) => {
									use std::convert::TryInto;
									let child_ref = child
										.try_into()
										.expect("TODO proper error and not using From");

									result[i] = Some(child_ref);
								},
								None => {},
							}
							i += 1;
						}
						result
					},
				},
			}
		};

		StackedNodeCheck {
			node,
			depth: 0,
			next_descended_child: 0,
			children,
			attached_value_hash: None,
		}
	}
}

impl<L: TrieLayout, D: SplitFirst> StackedNodeCheck<L, D> {
	fn data(&self) -> &[u8] {
		match &self.node {
			ItemStackNode::Inline(n) => n.data(),
			ItemStackNode::Node(n) => n.data(),
		}
	}

	fn node_plan(&self) -> &NodePlan {
		match &self.node {
			ItemStackNode::Inline(n) => n.node_plan(),
			ItemStackNode::Node(n) => n.node_plan(),
		}
	}
}

fn verify_hash<L: TrieLayout>(
	data: &[u8],
	expected: &[u8],
) -> Result<(), VerifyError<TrieHash<L>, CError<L>>> {
	let checked_hash = L::Hash::hash(data);
	if checked_hash.as_ref() != expected {
		let mut error_hash = TrieHash::<L>::default();
		error_hash.as_mut().copy_from_slice(expected);
		Err(VerifyError::HashMismatch(error_hash))
	} else {
		Ok(())
	}
}

/// Byte array where we can remove first item.
/// This is only needed for the ESCAPE_HEADER of COMPACT which
/// itself is not strictly needed (we can know if escaped from
/// query plan).
pub trait SplitFirst: Borrow<[u8]> + Clone {
	fn split_first(&mut self);
}

impl SplitFirst for Vec<u8> {
	fn split_first(&mut self) {
		*self = self.split_off(1);
	}
}

impl<'a> SplitFirst for &'a [u8] {
	fn split_first(&mut self) {
		*self = &self[1..];
	}
}

/// Content return on success when reading proof.
pub enum ReadProofItem<'a, L: TrieLayout, C, D: SplitFirst> {
	/// Successfull read of proof, not all content read.
	Halted(Box<HaltedStateCheck<'a, L, C, D>>),
	/// Seen value and key in proof.
	/// We only return content matching the query plan.
	/// TODO should be possible to return &Vec<u8>
	Value(Cow<'a, [u8]>, Vec<u8>),
	/// Seen hash of value and key in proof.
	/// We only return content matching the query plan.
	Hash(Cow<'a, [u8]>, TrieHash<L>),
	/// No value seen for a key in the input query plan.
	NoValue(&'a [u8]),
	/// Seen fully covered prefix in proof, this is only
	/// return when we read the proof with the query input (otherwhise
	/// we would need to indicate every child without a hash as a prefix).
	/// TODO unused implement
	StartPrefix(Vec<u8>),
	/// End of a previously start prefix.
	/// TODO unused implement
	EndPrefix,
}