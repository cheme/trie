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
	nibble::{self, NibbleOps, NibbleSlice, NibbleVec},
	node_codec::NodeCodec,
	Bytes, CError, ChildReference, Result, TrieError, TrieHash, TrieLayout,
};
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};
use hash_db::Hasher;

use crate::nibble::{ChildIndex, ChildSliceIndex};

use crate::rstd::{borrow::Borrow, marker::PhantomData, mem, ops::Range};

/// Partial node key type: offset and owned value of a nibbleslice.
/// Offset is applied on first byte of array (bytes are right aligned).
pub type NodeKey = (usize, nibble::BackingByteVec);

#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
/// Alias to branch children slice, it is equivalent to '&[&[u8]]'.
/// Reason for using it is https://github.com/rust-lang/rust/issues/43408.
pub struct BranchChildrenSlice<'a, I> {
	index: I,
	data: &'a [u8],
}

impl<'a, I: ChildSliceIndex> BranchChildrenSlice<'a, I> {
	/// Similar to `Index` but returns a copied value.
	pub fn at(&self, index: usize) -> Option<NodeHandle<'a>> {
		self.index.slice_at(index, self.data)
	}

	/// Iterator over children node handles.
	pub fn iter(&'a self) -> impl Iterator<Item = Option<NodeHandle<'a>>> {
		self.index.iter(self.data)
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
	pub fn to_owned_handle<L: TrieLayout<N>, const N: usize>(
		&self,
	) -> Result<NodeHandleOwned<TrieHash<L, N>, N>, TrieHash<L, N>, CError<L, N>> {
		match self {
			Self::Hash(h) => decode_hash::<L::Hash>(h)
				.ok_or_else(|| Box::new(TrieError::InvalidHash(Default::default(), h.to_vec())))
				.map(NodeHandleOwned::Hash),
			Self::Inline(i) => match L::Codec::decode(i) {
				Ok(node) => Ok(NodeHandleOwned::Inline(Box::new(node.to_owned_node::<L>()?))),
				Err(e) => Err(Box::new(TrieError::DecoderError(Default::default(), e))),
			},
		}
	}
}

/// Owned version of [`NodeHandleOwned`].
/// TODO param L
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum NodeHandleOwned<H, const N: usize> {
	Hash(H),
	Inline(Box<NodeOwned<H, N>>),
}

impl<H, const N: usize> NodeHandleOwned<H, N>
where
	H: Default + AsRef<[u8]> + AsMut<[u8]> + Copy,
{
	/// Returns `self` as a [`ChildReference`].
	///
	/// # Panic
	///
	/// This function panics if `self == Self::Inline(_)` and the inline node encoded length is
	/// greater then the length of the hash.
	fn as_child_reference<C: NodeCodec<N, HashOut = H>>(&self) -> ChildReference<H> {
		match self {
			NodeHandleOwned::Hash(h) => ChildReference::Hash(*h),
			NodeHandleOwned::Inline(n) => {
				let encoded = n.to_encoded::<C>();
				let mut store = H::default();
				assert!(store.as_ref().len() > encoded.len(), "Invalid inline node handle");

				store.as_mut()[..encoded.len()].copy_from_slice(&encoded);
				ChildReference::Inline(store, encoded.len())
			},
		}
	}
}

impl<H, const N: usize> NodeHandleOwned<H, N> {
	/// Returns `self` as inline node.
	pub fn as_inline(&self) -> Option<&NodeOwned<H, N>> {
		match self {
			Self::Hash(_) => None,
			Self::Inline(node) => Some(&*node),
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

	pub fn to_owned_value<L: TrieLayout<N>, const N: usize>(&self) -> ValueOwned<TrieHash<L, N>> {
		match self {
			Self::Inline(data) => ValueOwned::Inline(Bytes::from(*data), L::Hash::hash(data)),
			Self::Node(hash) => {
				let mut res = TrieHash::<L, N>::default();
				res.as_mut().copy_from_slice(hash);

				ValueOwned::Node(res)
			},
		}
	}
}

/// Owned value representation in `Node`.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum ValueOwned<H> {
	/// Value bytes as stored in a trie node and its hash.
	Inline(Bytes, H),
	/// Hash byte slice as stored in a trie node.
	Node(H),
}

impl<H: AsRef<[u8]> + Copy> ValueOwned<H> {
	/// Returns self as [`Value`].
	pub fn as_value(&self) -> Value {
		match self {
			Self::Inline(data, _) => Value::Inline(&data),
			Self::Node(hash) => Value::Node(hash.as_ref()),
		}
	}

	/// Returns the hash of the data stored in self.
	pub fn data_hash(&self) -> Option<H> {
		match self {
			Self::Inline(_, hash) => Some(*hash),
			Self::Node(hash) => Some(*hash),
		}
	}
}

impl<H> ValueOwned<H> {
	/// Returns the data stored in self.
	pub fn data(&self) -> Option<&Bytes> {
		match self {
			Self::Inline(data, _) => Some(data),
			Self::Node(_) => None,
		}
	}
}

/// Type of node in the trie and essential information thereof.
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Node<'a, const N: usize> {
	/// Null trie node; could be an empty root or an empty branch entry.
	Empty,
	/// Leaf node; has key slice and value. Value may not be empty.
	Leaf(NibbleSlice<'a, N>, Value<'a>),
	/// Extension node; has key slice and node data. Data may not be null.
	Extension(NibbleSlice<'a, N>, NodeHandle<'a>),
	/// Branch node; has slice of child nodes (each possibly null)
	/// and an optional immediate node data.
	Branch([Option<NodeHandle<'a>>; N], Option<Value<'a>>),
	/// Branch node with support for a nibble (when extension nodes are not used).
	NibbledBranch(NibbleSlice<'a, N>, [Option<NodeHandle<'a>>; N], Option<Value<'a>>),
}

impl<const N: usize> Node<'_, N> {
	/// Converts this node into a [`NodeOwned`].
	pub fn to_owned_node<L: TrieLayout<N>>(
		&self,
	) -> Result<NodeOwned<TrieHash<L, N>, N>, TrieHash<L, N>, CError<L, N>> {
		match self {
			Self::Empty => Ok(NodeOwned::Empty),
			Self::Leaf(n, d) => Ok(NodeOwned::Leaf((*n).into(), d.to_owned_value::<L, N>())),
			Self::Extension(n, h) =>
				Ok(NodeOwned::Extension((*n).into(), h.to_owned_handle::<L, N>()?)),
			Self::Branch(childs, data) => {
				let mut childs_owned = [(); N].map(|_| None);
				childs
					.iter()
					.enumerate()
					.map(|(i, c)| {
						childs_owned[i] =
							c.as_ref().map(|c| c.to_owned_handle::<L, N>()).transpose()?;
						Ok(())
					})
					.collect::<Result<_, _, _>>()?;

				Ok(NodeOwned::Branch(
					childs_owned,
					data.as_ref().map(|d| d.to_owned_value::<L, N>()),
				))
			},
			Self::NibbledBranch(n, childs, data) => {
				let mut childs_owned = [(); N].map(|_| None);
				childs
					.iter()
					.enumerate()
					.map(|(i, c)| {
						childs_owned[i] =
							c.as_ref().map(|c| c.to_owned_handle::<L, N>()).transpose()?;
						Ok(())
					})
					.collect::<Result<_, _, _>>()?;

				Ok(NodeOwned::NibbledBranch(
					(*n).into(),
					childs_owned,
					data.as_ref().map(|d| d.to_owned_value::<L, N>()),
				))
			},
		}
	}
}

/// Owned version of [`Node`].
/// TODO NodeOwned<L, N>
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum NodeOwned<H, const N: usize> {
	/// Null trie node; could be an empty root or an empty branch entry.
	Empty,
	/// Leaf node; has key slice and value. Value may not be empty.
	Leaf(NibbleVec<N>, ValueOwned<H>),
	/// Extension node; has key slice and node data. Data may not be null.
	Extension(NibbleVec<N>, NodeHandleOwned<H, N>),
	/// Branch node; has slice of child nodes (each possibly null)
	/// and an optional immediate node data.
	Branch([Option<NodeHandleOwned<H, N>>; N], Option<ValueOwned<H>>),
	/// Branch node with support for a nibble (when extension nodes are not used).
	NibbledBranch(NibbleVec<N>, [Option<NodeHandleOwned<H, N>>; N], Option<ValueOwned<H>>),
	/// Node that represents a value.
	///
	/// This variant is only constructed when working with a [`crate::TrieCache`]. It is only
	/// used to cache a raw value.
	Value(Bytes, H),
}

impl<H, const N: usize> NodeOwned<H, N>
where
	H: Default + AsRef<[u8]> + AsMut<[u8]> + Copy,
{
	/// Convert to its encoded format.
	pub fn to_encoded<C>(&self) -> Vec<u8>
	where
		C: NodeCodec<N, HashOut = H>,
	{
		match self {
			Self::Empty => C::empty_node().to_vec(),
			Self::Leaf(partial, value) =>
				C::leaf_node(partial.right_iter(), partial.len(), value.as_value()),
			Self::Extension(partial, child) => C::extension_node(
				partial.right_iter(),
				partial.len(),
				child.as_child_reference::<C>(),
			),
			Self::Branch(children, value) => C::branch_node(
				children.iter().map(|child| child.as_ref().map(|c| c.as_child_reference::<C>())),
				value.as_ref().map(|v| v.as_value()),
			),
			Self::NibbledBranch(partial, children, value) => C::branch_node_nibbled(
				partial.right_iter(),
				partial.len(),
				children.iter().map(|child| child.as_ref().map(|c| c.as_child_reference::<C>())),
				value.as_ref().map(|v| v.as_value()),
			),
			Self::Value(data, _) => data.to_vec(),
		}
	}

	/// Returns an iterator over all existing children with their optional nibble.
	pub fn child_iter(&self) -> impl Iterator<Item = (Option<u8>, &NodeHandleOwned<H, N>)> {
		enum ChildIter<'a, H, const N: usize> {
			Empty,
			Single(&'a NodeHandleOwned<H, N>, bool),
			Array(&'a [Option<NodeHandleOwned<H, N>>; N], usize),
		}

		impl<'a, H, const N: usize> Iterator for ChildIter<'a, H, N> {
			type Item = (Option<u8>, &'a NodeHandleOwned<H, N>);

			fn next(&mut self) -> Option<Self::Item> {
				loop {
					match self {
						Self::Empty => break None,
						Self::Single(child, returned) =>
							break if *returned {
								None
							} else {
								*returned = true;
								Some((None, child))
							},
						Self::Array(childs, index) =>
							if *index >= childs.len() {
								break None
							} else {
								*index += 1;

								// Ignore non-existing childs.
								if let Some(ref child) = childs[*index - 1] {
									break Some((Some(*index as u8 - 1), child))
								}
							},
					}
				}
			}
		}

		match self {
			Self::Leaf(_, _) | Self::Empty | Self::Value(_, _) => ChildIter::Empty,
			Self::Extension(_, child) => ChildIter::Single(child, false),
			Self::Branch(children, _) | Self::NibbledBranch(_, children, _) =>
				ChildIter::Array(children, 0),
		}
	}

	/// Returns the hash of the data attached to this node.
	pub fn data_hash(&self) -> Option<H> {
		match &self {
			Self::Empty => None,
			Self::Leaf(_, value) => value.data_hash(),
			Self::Extension(_, _) => None,
			Self::Branch(_, value) => value.as_ref().and_then(|v| v.data_hash()),
			Self::NibbledBranch(_, _, value) => value.as_ref().and_then(|v| v.data_hash()),
			Self::Value(_, hash) => Some(*hash),
		}
	}
}

impl<H, const N: usize> NodeOwned<H, N> {
	/// Returns the data attached to this node.
	pub fn data(&self) -> Option<&Bytes> {
		match &self {
			Self::Empty => None,
			Self::Leaf(_, value) => value.data(),
			Self::Extension(_, _) => None,
			Self::Branch(_, value) => value.as_ref().and_then(|v| v.data()),
			Self::NibbledBranch(_, _, value) => value.as_ref().and_then(|v| v.data()),
			Self::Value(data, _) => Some(data),
		}
	}

	/// Returns the partial key of this node.
	pub fn partial_key(&self) -> Option<&NibbleVec<N>> {
		match self {
			Self::Branch(_, _) | Self::Value(_, _) | Self::Empty => None,
			Self::Extension(partial, _) |
			Self::Leaf(partial, _) |
			Self::NibbledBranch(partial, _, _) => Some(partial),
		}
	}

	/// Returns the size in bytes of this node in memory.
	///
	/// This also includes the size of any inline child nodes.
	pub fn size_in_bytes(&self) -> usize {
		let self_size = mem::size_of::<Self>();

		fn childs_size<'a, H: 'a, const N: usize>(
			childs: impl Iterator<Item = &'a Option<NodeHandleOwned<H, N>>>,
		) -> usize {
			// If a `child` isn't an inline node, its size is already taken account for by
			// `self_size`.
			childs
				.filter_map(|c: &'a Option<NodeHandleOwned<H, N>>| c.as_ref())
				.map(|c| c.as_inline().map_or(0, |n| n.size_in_bytes()))
				.sum()
		}

		// As `self_size` only represents the static size of `Self`, we also need
		// to add the size of any dynamically allocated data.
		let dynamic_size = match self {
			Self::Empty => 0,
			Self::Leaf(nibbles, value) =>
				nibbles.inner().len() + value.data().map_or(0, |b| b.len()),
			Self::Value(bytes, _) => bytes.len(),
			Self::Extension(nibbles, child) => {
				// If the `child` isn't an inline node, its size is already taken account for by
				// `self_size`.
				nibbles.inner().len() + child.as_inline().map_or(0, |n| n.size_in_bytes())
			},
			Self::Branch(childs, value) =>
				childs_size(childs.iter()) +
					value.as_ref().and_then(|v| v.data()).map_or(0, |b| b.len()),
			Self::NibbledBranch(nibbles, childs, value) =>
				nibbles.inner().len() +
					childs_size(childs.iter()) +
					value.as_ref().and_then(|v| v.data()).map_or(0, |b| b.len()),
		};

		self_size + dynamic_size
	}
}

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
pub struct NibbleSlicePlan<const N: usize> {
	bytes: Range<usize>,
	offset: usize,
}

impl<const N: usize> NibbleSlicePlan<N> {
	/// Construct a nibble slice decode plan.
	pub fn new(bytes: Range<usize>, offset: usize) -> Self {
		NibbleSlicePlan { bytes, offset }
	}

	/// Returns the nibble length of the slice.
	pub fn len(&self) -> usize {
		(self.bytes.end - self.bytes.start) * NibbleOps::<N>::nibble_per_byte() - self.offset
	}

	/// Build a nibble slice by decoding a byte slice according to the plan. It is the
	/// responsibility of the caller to ensure that the node plan was created for the argument
	/// data, otherwise the call may decode incorrectly or panic.
	pub fn build<'a, 'b>(&'a self, data: &'b [u8]) -> NibbleSlice<'b, N> {
		NibbleSlice::new_offset(&data[self.bytes.clone()], self.offset)
	}
}

/// TODO try non public
#[derive(Eq, PartialEq, Clone)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct BranchChildrenNodePlan<I> {
	index: I,
}

impl<I: ChildIndex<NodeHandlePlan>> BranchChildrenNodePlan<I> {
	/// Similar to `Index` but return a copied value.
	pub fn at(&self, index: usize) -> Option<NodeHandlePlan> {
		self.index.at(index).cloned()
	}

	/// Build from sequence of content.
	pub fn new(nodes: impl Iterator<Item = Option<NodeHandlePlan>>) -> Self {
		let index = ChildIndex::from_iter(nodes);
		BranchChildrenNodePlan { index }
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
pub enum NodePlan<const N: usize> {
	/// Null trie node; could be an empty root or an empty branch entry.
	Empty,
	/// Leaf node; has a partial key plan and value.
	Leaf { partial: NibbleSlicePlan<N>, value: ValuePlan },
	/// Extension node; has a partial key plan and child data.
	Extension { partial: NibbleSlicePlan<N>, child: NodeHandlePlan },
	/// Branch node; has slice of child nodes (each possibly null)
	/// and an optional immediate node data.
	Branch { value: Option<ValuePlan>, children: [Option<NodeHandlePlan>; N] },
	/// Branch node with support for a nibble (when extension nodes are not used).
	NibbledBranch {
		partial: NibbleSlicePlan<N>,
		value: Option<ValuePlan>,
		children: [Option<NodeHandlePlan>; N],
	},
}

impl<const N: usize> NodePlan<N> {
	/// Build a node by decoding a byte slice according to the node plan. It is the responsibility
	/// of the caller to ensure that the node plan was created for the argument data, otherwise the
	/// call may decode incorrectly or panic.
	pub fn build<'a, 'b>(&'a self, data: &'b [u8]) -> Node<'b, N> {
		match self {
			NodePlan::Empty => Node::Empty,
			NodePlan::Leaf { partial, value } => Node::Leaf(partial.build(data), value.build(data)),
			NodePlan::Extension { partial, child } =>
				Node::Extension(partial.build(data), child.build(data)),
			NodePlan::Branch { value, children } => {
				let mut child_slices = [None; N];
				for i in 0..N {
					child_slices[i] = children[i].as_ref().map(|child| child.build(data));
				}
				Node::Branch(child_slices, value.as_ref().map(|v| v.build(data)))
			},
			NodePlan::NibbledBranch { partial, value, children } => {
				let mut child_slices = [None; N];
				for i in 0..N {
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
pub struct OwnedNode<D: Borrow<[u8]>, const N: usize> {
	data: D,
	plan: NodePlan<N>,
}

impl<D: Borrow<[u8]>, const N: usize> OwnedNode<D, N> {
	/// Construct an `OwnedNode` by decoding an owned data source according to some codec.
	pub fn new<C: NodeCodec<N>>(data: D) -> core::result::Result<Self, C::Error> {
		let plan = C::decode_plan(data.borrow())?;
		Ok(OwnedNode { data, plan })
	}

	/// Returns a reference to the backing data.
	pub fn data(&self) -> &[u8] {
		self.data.borrow()
	}

	/// Returns a reference to the node decode plan.
	pub fn node_plan(&self) -> &NodePlan<N> {
		&self.plan
	}

	/// Returns a mutable reference to the node decode plan.
	pub fn node_plan_mut(&mut self) -> &mut NodePlan<N> {
		&mut self.plan
	}

	/// Construct a `Node` by borrowing data from this struct.
	pub fn node(&self) -> Node<N> {
		self.plan.build(self.data.borrow())
	}
}
