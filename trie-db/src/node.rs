// Copyright 2017, 2021 Parity Technologies
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

use crate::{
	nibble::{self, nibble_ops, NibbleSlice},
	node_codec::NodeCodec,
	Bytes, CError, Result, TrieError, TrieHash, TrieLayout,
};
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};
use hash_db::Hasher;

use crate::rstd::{borrow::Borrow, ops::Range};

/// Partial node key type: offset and owned value of a nibbleslice.
/// Offset is applied on first byte of array (bytes are right aligned).
pub type NodeKey = (usize, nibble::BackingByteVec);

/// Get dynamic size for `NodeKey`.
pub fn node_key_dyn_size(key: &NodeKey) -> usize {
	if key.1.len() > crate::nibble::BACKING_BYTE_VEC_SMALLVEC_SIZE {
		key.1.len()
	} else {
		0
	}
}

/// A reference to a trie node which may be stored within another trie node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHandle<'a> {
	Hash(&'a [u8]),
	Inline(&'a [u8]),
}

impl NodeHandle<'_> {
	/// Converts this node handle into a [`NodeHandleOwned`].
	pub fn to_owned_handle<L: TrieLayout>(
		&self,
	) -> Result<crate::triedbmut::NodeHandle<L>, TrieHash<L>, CError<L>> {
		use crate::triedbmut::NodeHandle as NodeHandleOwned;
		match self {
			Self::Hash(h) => decode_hash::<L::Hash>(h)
				.ok_or_else(|| Box::new(TrieError::InvalidHash(Default::default(), h.to_vec())))
				.map(NodeHandleOwned::Hash),
			Self::Inline(i) => match L::Codec::decode(i) {
				Ok(node) => Ok(NodeHandleOwned::Inline(Box::new(node.to_owned_node()?))),
				Err(e) => Err(Box::new(TrieError::DecoderError(Default::default(), e))),
			},
		}
	}
}

/// Read a hash from a slice into a Hasher output. Returns None if the slice is the wrong length.
pub fn decode_hash<H: Hasher>(data: &[u8]) -> Option<H::Out> {
	if data.len() != H::LENGTH {
		return None
	}
	let mut hash = H::Out::default();
	hash.as_mut().copy_from_slice(data);
	Some(hash)
}

/// Value representation in `Node`.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Value<'a> {
	/// Value byte slice as stored in a trie node.
	Inline(&'a [u8]),
	/// Hash byte slice as stored in a trie node.
	Node(&'a [u8]),
}

impl<'a> Value<'a> {
	pub(crate) fn new_inline(value: &'a [u8], threshold: Option<u32>) -> Option<Self> {
		if let Some(threshold) = threshold {
			if value.len() >= threshold as usize {
				return None
			} else {
				Some(Value::Inline(value))
			}
		} else {
			Some(Value::Inline(value))
		}
	}

	pub fn to_owned_value<L: TrieLayout>(&self) -> crate::triedbmut::Value<L> {
		use crate::triedbmut::Value as ValueOwned;
		match self {
			Self::Inline(data) => ValueOwned::Inline(Bytes::from(*data), None),
			Self::Node(hash) => {
				let mut res = TrieHash::<L>::default();
				res.as_mut().copy_from_slice(hash);

				ValueOwned::Node(res)
			},
		}
	}
}

/// Type of node in the trie and essential information thereof.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Node<'a> {
	/// Null trie node; could be an empty root or an empty branch entry.
	Empty,
	/// Leaf node; has key slice and value. Value may not be empty.
	Leaf(NibbleSlice<'a>, Value<'a>),
	/// Extension node; has key slice and node data. Data may not be null.
	Extension(NibbleSlice<'a>, NodeHandle<'a>),
	/// Branch node; has slice of child nodes (each possibly null)
	/// and an optional immediate node data.
	Branch([Option<NodeHandle<'a>>; nibble_ops::NIBBLE_LENGTH], Option<Value<'a>>),
	/// Branch node with support for a nibble (when extension nodes are not used).
	NibbledBranch(
		NibbleSlice<'a>,
		[Option<NodeHandle<'a>>; nibble_ops::NIBBLE_LENGTH],
		Option<Value<'a>>,
	),
}

impl Node<'_> {
	/// Converts this node into a [`NodeOwned`].
	pub fn to_owned_node<L: TrieLayout>(
		&self,
	) -> Result<crate::triedbmut::Node<L>, TrieHash<L>, CError<L>> {
		use crate::triedbmut::Node as NodeOwned;
		match self {
			Self::Empty => Ok(NodeOwned::Empty),
			Self::Leaf(n, d) => Ok(NodeOwned::Leaf((*n).into(), d.to_owned_value())),
			Self::Extension(n, h) => Ok(NodeOwned::Extension((*n).into(), h.to_owned_handle()?)),
			Self::Branch(childs, data) => {
				let mut childs_owned = [(); nibble_ops::NIBBLE_LENGTH].map(|_| None);
				childs
					.iter()
					.enumerate()
					.map(|(i, c)| {
						childs_owned[i] = c.as_ref().map(|c| c.to_owned_handle()).transpose()?;
						Ok(())
					})
					.collect::<Result<_, _, _>>()?;

				Ok(NodeOwned::Branch(
					Box::new(childs_owned),
					data.as_ref().map(|d| d.to_owned_value()),
				))
			},
			Self::NibbledBranch(n, childs, data) => {
				let mut childs_owned = [(); nibble_ops::NIBBLE_LENGTH].map(|_| None);
				childs
					.iter()
					.enumerate()
					.map(|(i, c)| {
						childs_owned[i] = c.as_ref().map(|c| c.to_owned_handle()).transpose()?;
						Ok(())
					})
					.collect::<Result<_, _, _>>()?;

				Ok(NodeOwned::NibbledBranch(
					(*n).into(),
					Box::new(childs_owned),
					data.as_ref().map(|d| d.to_owned_value()),
				))
			},
		}
	}
}

/// Node that represents a value.
///
/// This variant is only constructed when working with a [`crate::TrieCache`]. It is only
/// used to cache a raw value.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct CachedValueOwned<H>(pub Bytes, pub H);

/// A `NodeHandlePlan` is a decoding plan for constructing a `NodeHandle` from an encoded trie
/// node. This is used as a substructure of `NodePlan`. See `NodePlan` for details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeHandlePlan {
	Hash(Range<usize>),
	Inline(Range<usize>),
}

impl NodeHandlePlan {
	/// Build a node handle by decoding a byte slice according to the node handle plan. It is the
	/// responsibility of the caller to ensure that the node plan was created for the argument
	/// data, otherwise the call may decode incorrectly or panic.
	pub fn build<'a, 'b>(&'a self, data: &'b [u8]) -> NodeHandle<'b> {
		match self {
			NodeHandlePlan::Hash(range) => NodeHandle::Hash(&data[range.clone()]),
			NodeHandlePlan::Inline(range) => NodeHandle::Inline(&data[range.clone()]),
		}
	}
}

/// A `NibbleSlicePlan` is a blueprint for decoding a nibble slice from a byte slice. The
/// `NibbleSlicePlan` is created by parsing a byte slice and can be reused multiple times.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct NibbleSlicePlan {
	bytes: Range<usize>,
	offset: usize,
}

impl NibbleSlicePlan {
	/// Construct a nibble slice decode plan.
	pub fn new(bytes: Range<usize>, offset: usize) -> Self {
		NibbleSlicePlan { bytes, offset }
	}

	/// Returns the nibble length of the slice.
	pub fn len(&self) -> usize {
		(self.bytes.end - self.bytes.start) * nibble_ops::NIBBLE_PER_BYTE - self.offset
	}

	/// Build a nibble slice by decoding a byte slice according to the plan. It is the
	/// responsibility of the caller to ensure that the node plan was created for the argument
	/// data, otherwise the call may decode incorrectly or panic.
	pub fn build<'a, 'b>(&'a self, data: &'b [u8]) -> NibbleSlice<'b> {
		NibbleSlice::new_offset(&data[self.bytes.clone()], self.offset)
	}
}

/// Plan for value representation in `NodePlan`.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum ValuePlan {
	/// Range for byte representation in encoded node.
	Inline(Range<usize>),
	/// Range for hash in encoded node and original
	/// value size.
	Node(Range<usize>),
}

impl ValuePlan {
	/// Build a value slice by decoding a byte slice according to the plan.
	pub fn build<'a, 'b>(&'a self, data: &'b [u8]) -> Value<'b> {
		match self {
			ValuePlan::Inline(range) => Value::Inline(&data[range.clone()]),
			ValuePlan::Node(range) => Value::Node(&data[range.clone()]),
		}
	}
}

/// A `NodePlan` is a blueprint for decoding a node from a byte slice. The `NodePlan` is created
/// by parsing an encoded node and can be reused multiple times. This is useful as a `Node` borrows
/// from a byte slice and this struct does not.
///
/// The enum values mirror those of `Node` except that instead of byte slices, this struct stores
/// ranges that can be used to index into a large byte slice.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum NodePlan {
	/// Null trie node; could be an empty root or an empty branch entry.
	Empty,
	/// Leaf node; has a partial key plan and value.
	Leaf { partial: NibbleSlicePlan, value: ValuePlan },
	/// Extension node; has a partial key plan and child data.
	Extension { partial: NibbleSlicePlan, child: NodeHandlePlan },
	/// Branch node; has slice of child nodes (each possibly null)
	/// and an optional immediate node data.
	Branch {
		value: Option<ValuePlan>,
		children: [Option<NodeHandlePlan>; nibble_ops::NIBBLE_LENGTH],
	},
	/// Branch node with support for a nibble (when extension nodes are not used).
	NibbledBranch {
		partial: NibbleSlicePlan,
		value: Option<ValuePlan>,
		children: [Option<NodeHandlePlan>; nibble_ops::NIBBLE_LENGTH],
	},
}

impl NodePlan {
	/// Build a node by decoding a byte slice according to the node plan. It is the responsibility
	/// of the caller to ensure that the node plan was created for the argument data, otherwise the
	/// call may decode incorrectly or panic.
	pub fn build<'a, 'b>(&'a self, data: &'b [u8]) -> Node<'b> {
		match self {
			NodePlan::Empty => Node::Empty,
			NodePlan::Leaf { partial, value } => Node::Leaf(partial.build(data), value.build(data)),
			NodePlan::Extension { partial, child } =>
				Node::Extension(partial.build(data), child.build(data)),
			NodePlan::Branch { value, children } => {
				let mut child_slices = [None; nibble_ops::NIBBLE_LENGTH];
				for i in 0..nibble_ops::NIBBLE_LENGTH {
					child_slices[i] = children[i].as_ref().map(|child| child.build(data));
				}
				Node::Branch(child_slices, value.as_ref().map(|v| v.build(data)))
			},
			NodePlan::NibbledBranch { partial, value, children } => {
				let mut child_slices = [None; nibble_ops::NIBBLE_LENGTH];
				for i in 0..nibble_ops::NIBBLE_LENGTH {
					child_slices[i] = children[i].as_ref().map(|child| child.build(data));
				}
				Node::NibbledBranch(
					partial.build(data),
					child_slices,
					value.as_ref().map(|v| v.build(data)),
				)
			},
		}
	}

	/// Access value plan from node plan, return `None` for
	/// node that cannot contain a `ValuePlan`.
	pub fn value_plan(&self) -> Option<&ValuePlan> {
		match self {
			NodePlan::Extension { .. } | NodePlan::Empty => None,
			NodePlan::Leaf { value, .. } => Some(value),
			NodePlan::Branch { value, .. } | NodePlan::NibbledBranch { value, .. } =>
				value.as_ref(),
		}
	}

	/// Mutable ccess value plan from node plan, return `None` for
	/// node that cannot contain a `ValuePlan`.
	pub fn value_plan_mut(&mut self) -> Option<&mut ValuePlan> {
		match self {
			NodePlan::Extension { .. } | NodePlan::Empty => None,
			NodePlan::Leaf { value, .. } => Some(value),
			NodePlan::Branch { value, .. } | NodePlan::NibbledBranch { value, .. } =>
				value.as_mut(),
		}
	}
}

/// An `OwnedNode` is an owned type from which a `Node` can be constructed which borrows data from
/// the `OwnedNode`. This is useful for trie iterators.
#[cfg_attr(feature = "std", derive(Debug))]
#[derive(PartialEq, Eq)]
pub struct OwnedNode<D: Borrow<[u8]>> {
	data: D,
	plan: NodePlan,
}

impl<D: Borrow<[u8]>> OwnedNode<D> {
	/// Construct an `OwnedNode` by decoding an owned data source according to some codec.
	pub fn new<C: NodeCodec>(data: D) -> core::result::Result<Self, C::Error> {
		let plan = C::decode_plan(data.borrow())?;
		Ok(OwnedNode { data, plan })
	}

	/// Returns a reference to the backing data.
	pub fn data(&self) -> &[u8] {
		self.data.borrow()
	}

	/// Returns a reference to the node decode plan.
	pub fn node_plan(&self) -> &NodePlan {
		&self.plan
	}

	/// Returns a mutable reference to the node decode plan.
	pub fn node_plan_mut(&mut self) -> &mut NodePlan {
		&mut self.plan
	}

	/// Construct a `Node` by borrowing data from this struct.
	pub fn node(&self) -> Node {
		self.plan.build(self.data.borrow())
	}
}
