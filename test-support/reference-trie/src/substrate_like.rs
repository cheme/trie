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

//! Codec and layout configuration similar to upstream default substrate one.

use super::{CodecError as Error, NodeCodec as NodeCodecT, *};
use trie_db::node::{Node, NodeHandle, Value};

/// No extension trie with no hashed value.
pub struct HashedValueNoExt;

/// No extension trie which stores value above a static size
/// as external node.
pub struct HashedValueNoExtThreshold;

impl TrieLayout for HashedValueNoExt {
	const USE_EXTENSION: bool = false;
	const ALLOW_EMPTY: bool = false;
	const MAX_INLINE_VALUE: Option<u32> = None;

	type Hash = RefHasher;
	type Codec = ReferenceNodeCodecNoExtMeta<RefHasher>;
}

impl TrieLayout for HashedValueNoExtThreshold {
	const USE_EXTENSION: bool = false;
	const ALLOW_EMPTY: bool = false;
	const MAX_INLINE_VALUE: Option<u32> = Some(1);

	type Hash = RefHasher;
	type Codec = ReferenceNodeCodecNoExtMeta<RefHasher>;
}

/// Constants specific to encoding with external value node support.
pub mod trie_constants {
	const FIRST_PREFIX: u8 = 0b_00 << 6;
	pub const NIBBLE_SIZE_BOUND: usize = u16::max_value() as usize;
	pub const LEAF_PREFIX_MASK: u8 = 0b_01 << 6;
	pub const BRANCH_WITHOUT_MASK: u8 = 0b_10 << 6;
	pub const BRANCH_WITH_MASK: u8 = 0b_11 << 6;
	pub const EMPTY_TRIE: u8 = FIRST_PREFIX | (0b_00 << 4);
	pub const ALT_HASHING_LEAF_PREFIX_MASK: u8 = FIRST_PREFIX | (0b_1 << 5);
	pub const ALT_HASHING_BRANCH_WITH_MASK: u8 = FIRST_PREFIX | (0b_01 << 4);
	pub const ESCAPE_COMPACT_HEADER: u8 = EMPTY_TRIE | 0b_00_01;
}

#[derive(Default, Clone)]
pub struct NodeCodec<H>(PhantomData<H>);

impl<H: Hasher> NodeCodec<H> {
	fn decode_plan_inner_hashed(data: &[u8]) -> Result<NodePlan, Error> {
		let mut input = ByteSliceInput::new(data);

		let header = NodeHeader::decode(&mut input)?;
		let contains_hash = header.contains_hash_of_value();

		let branch_has_value = if let NodeHeader::Branch(has_value, _) = &header {
			*has_value
		} else {
			// alt_hash_branch
			true
		};

		match header {
			NodeHeader::Null => Ok(NodePlan::Empty),
			NodeHeader::HashedValueBranch(nibble_count) | NodeHeader::Branch(_, nibble_count) => {
				let padding = nibble_count % nibble_ops::NIBBLE_PER_BYTE != 0;
				// check that the padding is valid (if any)
				if padding && nibble_ops::pad_left(data[input.offset]) != 0 {
					return Err(CodecError::from("Bad format"))
				}
				let partial = input.take(
					(nibble_count + (nibble_ops::NIBBLE_PER_BYTE - 1)) /
						nibble_ops::NIBBLE_PER_BYTE,
				)?;
				let partial_padding = nibble_ops::number_padding(nibble_count);
				let bitmap_range = input.take(BITMAP_LENGTH)?;
				let bitmap = Bitmap::decode(&data[bitmap_range])?;
				let value = if branch_has_value {
					Some(if contains_hash {
						ValuePlan::Node(input.take(H::LENGTH)?)
					} else {
						let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
						ValuePlan::Inline(input.take(count)?)
					})
				} else {
					None
				};
				let mut children = [
					None, None, None, None, None, None, None, None, None, None, None, None, None,
					None, None, None,
				];
				for i in 0..nibble_ops::NIBBLE_LENGTH {
					if bitmap.value_at(i) {
						let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
						let range = input.take(count)?;
						children[i] = Some(if count == H::LENGTH {
							NodeHandlePlan::Hash(range)
						} else {
							NodeHandlePlan::Inline(range)
						});
					}
				}
				Ok(NodePlan::NibbledBranch {
					partial: NibbleSlicePlan::new(partial, partial_padding),
					value,
					children,
				})
			},
			NodeHeader::HashedValueLeaf(nibble_count) | NodeHeader::Leaf(nibble_count) => {
				let padding = nibble_count % nibble_ops::NIBBLE_PER_BYTE != 0;
				// check that the padding is valid (if any)
				if padding && nibble_ops::pad_left(data[input.offset]) != 0 {
					return Err(CodecError::from("Bad format"))
				}
				let partial = input.take(
					(nibble_count + (nibble_ops::NIBBLE_PER_BYTE - 1)) /
						nibble_ops::NIBBLE_PER_BYTE,
				)?;
				let partial_padding = nibble_ops::number_padding(nibble_count);
				let value = if contains_hash {
					ValuePlan::Node(input.take(H::LENGTH)?)
				} else {
					let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
					ValuePlan::Inline(input.take(count)?)
				};

				Ok(NodePlan::Leaf {
					partial: NibbleSlicePlan::new(partial, partial_padding),
					value,
				})
			},
		}
	}
}

impl<H> NodeCodecT for NodeCodec<H>
where
	H: Hasher,
{
	const ESCAPE_HEADER: Option<u8> = Some(trie_constants::ESCAPE_COMPACT_HEADER);
	type Error = Error;
	type HashOut = H::Out;

	fn hashed_null_node() -> <H as Hasher>::Out {
		H::hash(<Self as NodeCodecT>::empty_node())
	}

	fn decode_plan(data: &[u8]) -> Result<NodePlan, Self::Error> {
		Self::decode_plan_inner_hashed(data)
	}

	fn is_empty_node(data: &[u8]) -> bool {
		data == <Self as NodeCodecT>::empty_node()
	}

	fn empty_node() -> &'static [u8] {
		&[trie_constants::EMPTY_TRIE]
	}

	fn leaf_node(partial: impl Iterator<Item = u8>, number_nibble: usize, value: Value) -> Vec<u8> {
		let contains_hash = matches!(&value, Value::Node(..));
		let mut output = if contains_hash {
			partial_from_iterator_encode(partial, number_nibble, NodeKind::HashedValueLeaf)
		} else {
			partial_from_iterator_encode(partial, number_nibble, NodeKind::Leaf)
		};
		match value {
			Value::Inline(value) => {
				Compact(value.len() as u32).encode_to(&mut output);
				output.extend_from_slice(value);
			},
			Value::Node(hash) => {
				debug_assert!(hash.len() == H::LENGTH);
				output.extend_from_slice(hash);
			},
		}
		output
	}

	fn extension_node(
		_partial: impl Iterator<Item = u8>,
		_nbnibble: usize,
		_child: ChildReference<<H as Hasher>::Out>,
	) -> Vec<u8> {
		unreachable!("Codec without extension.")
	}

	fn branch_node(
		_children: impl Iterator<Item = impl Borrow<Option<ChildReference<<H as Hasher>::Out>>>>,
		_maybe_value: Option<Value>,
	) -> Vec<u8> {
		unreachable!("Codec without extension.")
	}

	fn branch_node_nibbled(
		partial: impl Iterator<Item = u8>,
		number_nibble: usize,
		children: impl Iterator<Item = impl Borrow<Option<ChildReference<<H as Hasher>::Out>>>>,
		value: Option<Value>,
	) -> Vec<u8> {
		let contains_hash = matches!(&value, Some(Value::Node(..)));
		let mut output = match (&value, contains_hash) {
			(&None, _) =>
				partial_from_iterator_encode(partial, number_nibble, NodeKind::BranchNoValue),
			(_, false) =>
				partial_from_iterator_encode(partial, number_nibble, NodeKind::BranchWithValue),
			(_, true) =>
				partial_from_iterator_encode(partial, number_nibble, NodeKind::HashedValueBranch),
		};

		let bitmap_index = output.len();
		let mut bitmap: [u8; BITMAP_LENGTH] = [0; BITMAP_LENGTH];
		(0..BITMAP_LENGTH).for_each(|_| output.push(0));
		match value {
			Some(Value::Inline(value)) => {
				Compact(value.len() as u32).encode_to(&mut output);
				output.extend_from_slice(value);
			},
			Some(Value::Node(hash)) => {
				debug_assert!(hash.len() == H::LENGTH);
				output.extend_from_slice(hash);
			},
			None => (),
		}
		Bitmap::encode(
			children.map(|maybe_child| match maybe_child.borrow() {
				Some(ChildReference::Hash(h)) => {
					h.as_ref().encode_to(&mut output);
					true
				},
				&Some(ChildReference::Inline(inline_data, len)) => {
					inline_data.as_ref()[..len].encode_to(&mut output);
					true
				},
				None => false,
			}),
			bitmap.as_mut(),
		);
		output[bitmap_index..bitmap_index + BITMAP_LENGTH]
			.copy_from_slice(&bitmap[..BITMAP_LENGTH]);
		output
	}
}

// utils

/// Encode and allocate node type header (type and size), and partial value.
/// It uses an iterator over encoded partial bytes as input.
fn partial_from_iterator_encode<I: Iterator<Item = u8>>(
	partial: I,
	nibble_count: usize,
	node_kind: NodeKind,
) -> Vec<u8> {
	let nibble_count = std::cmp::min(trie_constants::NIBBLE_SIZE_BOUND, nibble_count);

	let mut output = Vec::with_capacity(4 + (nibble_count / nibble_ops::NIBBLE_PER_BYTE));
	match node_kind {
		NodeKind::Leaf => NodeHeader::Leaf(nibble_count).encode_to(&mut output),
		NodeKind::BranchWithValue => NodeHeader::Branch(true, nibble_count).encode_to(&mut output),
		NodeKind::BranchNoValue => NodeHeader::Branch(false, nibble_count).encode_to(&mut output),
		NodeKind::HashedValueLeaf =>
			NodeHeader::HashedValueLeaf(nibble_count).encode_to(&mut output),
		NodeKind::HashedValueBranch =>
			NodeHeader::HashedValueBranch(nibble_count).encode_to(&mut output),
	};
	output.extend(partial);
	output
}

/// A node header.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum NodeHeader {
	Null,
	// contains wether there is a value and nibble count
	Branch(bool, usize),
	// contains nibble count
	Leaf(usize),
	// contains nibble count.
	HashedValueBranch(usize),
	// contains nibble count.
	HashedValueLeaf(usize),
}

impl NodeHeader {
	fn contains_hash_of_value(&self) -> bool {
		match self {
			NodeHeader::HashedValueBranch(_) | NodeHeader::HashedValueLeaf(_) => true,
			_ => false,
		}
	}
}

/// NodeHeader without content
pub(crate) enum NodeKind {
	Leaf,
	BranchNoValue,
	BranchWithValue,
	HashedValueLeaf,
	HashedValueBranch,
}

impl Encode for NodeHeader {
	fn encode_to<T: Output + ?Sized>(&self, output: &mut T) {
		match self {
			NodeHeader::Null => output.push_byte(trie_constants::EMPTY_TRIE),
			NodeHeader::Branch(true, nibble_count) =>
				encode_size_and_prefix(*nibble_count, trie_constants::BRANCH_WITH_MASK, 2, output),
			NodeHeader::Branch(false, nibble_count) => encode_size_and_prefix(
				*nibble_count,
				trie_constants::BRANCH_WITHOUT_MASK,
				2,
				output,
			),
			NodeHeader::Leaf(nibble_count) =>
				encode_size_and_prefix(*nibble_count, trie_constants::LEAF_PREFIX_MASK, 2, output),
			NodeHeader::HashedValueBranch(nibble_count) => encode_size_and_prefix(
				*nibble_count,
				trie_constants::ALT_HASHING_BRANCH_WITH_MASK,
				4,
				output,
			),
			NodeHeader::HashedValueLeaf(nibble_count) => encode_size_and_prefix(
				*nibble_count,
				trie_constants::ALT_HASHING_LEAF_PREFIX_MASK,
				3,
				output,
			),
		}
	}
}

impl parity_scale_codec::EncodeLike for NodeHeader {}

impl Decode for NodeHeader {
	fn decode<I: Input>(input: &mut I) -> Result<Self, Error> {
		let i = input.read_byte()?;
		if i == trie_constants::EMPTY_TRIE {
			return Ok(NodeHeader::Null)
		}
		match i & (0b11 << 6) {
			trie_constants::LEAF_PREFIX_MASK => Ok(NodeHeader::Leaf(decode_size(i, input, 2)?)),
			trie_constants::BRANCH_WITH_MASK =>
				Ok(NodeHeader::Branch(true, decode_size(i, input, 2)?)),
			trie_constants::BRANCH_WITHOUT_MASK =>
				Ok(NodeHeader::Branch(false, decode_size(i, input, 2)?)),
			trie_constants::EMPTY_TRIE => {
				if i & (0b111 << 5) == trie_constants::ALT_HASHING_LEAF_PREFIX_MASK {
					Ok(NodeHeader::HashedValueLeaf(decode_size(i, input, 3)?))
				} else if i & (0b1111 << 4) == trie_constants::ALT_HASHING_BRANCH_WITH_MASK {
					Ok(NodeHeader::HashedValueBranch(decode_size(i, input, 4)?))
				} else {
					// do not allow any special encoding
					Err("Unallowed encoding".into())
				}
			},
			_ => unreachable!(),
		}
	}
}

/// Returns an iterator over encoded bytes for node header and size.
/// Size encoding allows unlimited, length inefficient, representation, but
/// is bounded to 16 bit maximum value to avoid possible DOS.
pub(crate) fn size_and_prefix_iterator(
	size: usize,
	prefix: u8,
	prefix_mask: usize,
) -> impl Iterator<Item = u8> {
	let size = std::cmp::min(trie_constants::NIBBLE_SIZE_BOUND, size);

	let max_value = 255u8 >> prefix_mask;
	let l1 = std::cmp::min(max_value as usize - 1, size);
	let (first_byte, mut rem) = if size == l1 {
		(once(prefix + l1 as u8), 0)
	} else {
		(once(prefix + max_value as u8), size - l1)
	};
	let next_bytes = move || {
		if rem > 0 {
			if rem < 256 {
				let result = rem - 1;
				rem = 0;
				Some(result as u8)
			} else {
				rem = rem.saturating_sub(255);
				Some(255)
			}
		} else {
			None
		}
	};
	first_byte.chain(std::iter::from_fn(next_bytes))
}

/// Encodes size and prefix to a stream output (prefix on 2 first bit only).
fn encode_size_and_prefix<W>(size: usize, prefix: u8, prefix_mask: usize, out: &mut W)
where
	W: Output + ?Sized,
{
	for b in size_and_prefix_iterator(size, prefix, prefix_mask) {
		out.push_byte(b)
	}
}

/// Decode size only from stream input and header byte.
fn decode_size(first: u8, input: &mut impl Input, prefix_mask: usize) -> Result<usize, Error> {
	let max_value = 255u8 >> prefix_mask;
	let mut result = (first & max_value) as usize;
	if result < max_value as usize {
		return Ok(result)
	}
	result -= 1;
	while result <= trie_constants::NIBBLE_SIZE_BOUND {
		let n = input.read_byte()? as usize;
		if n < 255 {
			return Ok(result + n + 1)
		}
		result += 255;
	}
	Ok(trie_constants::NIBBLE_SIZE_BOUND)
}

/// Reference implementation of a `TrieStream` without extension.
#[derive(Default, Clone)]
pub struct ReferenceTrieStreamNoExt {
	/// Current node buffer.
	buffer: Vec<u8>,
}

/// Create a leaf/branch node, encoding a number of nibbles.
fn fuse_nibbles_node<'a>(nibbles: &'a [u8], kind: NodeKind) -> impl Iterator<Item = u8> + 'a {
	let size = std::cmp::min(trie_constants::NIBBLE_SIZE_BOUND, nibbles.len());

	let iter_start = match kind {
		NodeKind::Leaf => size_and_prefix_iterator(size, trie_constants::LEAF_PREFIX_MASK, 2),
		NodeKind::BranchNoValue =>
			size_and_prefix_iterator(size, trie_constants::BRANCH_WITHOUT_MASK, 2),
		NodeKind::BranchWithValue =>
			size_and_prefix_iterator(size, trie_constants::BRANCH_WITH_MASK, 2),
		NodeKind::HashedValueLeaf =>
			size_and_prefix_iterator(size, trie_constants::ALT_HASHING_LEAF_PREFIX_MASK, 3),
		NodeKind::HashedValueBranch =>
			size_and_prefix_iterator(size, trie_constants::ALT_HASHING_BRANCH_WITH_MASK, 4),
	};
	iter_start
		.chain(if nibbles.len() % 2 == 1 { Some(nibbles[0]) } else { None })
		.chain(nibbles[nibbles.len() % 2..].chunks(2).map(|ch| ch[0] << 4 | ch[1]))
}

use trie_root::Value as TrieStreamValue;
impl TrieStream for ReferenceTrieStreamNoExt {
	fn new() -> Self {
		Self { buffer: Vec::new() }
	}

	fn append_empty_data(&mut self) {
		self.buffer.push(trie_constants::EMPTY_TRIE);
	}

	fn append_leaf(&mut self, key: &[u8], value: TrieStreamValue) {
		let kind = match &value {
			TrieStreamValue::Inline(..) => NodeKind::Leaf,
			TrieStreamValue::Node(..) => NodeKind::HashedValueLeaf,
		};
		self.buffer.extend(fuse_nibbles_node(key, kind));
		match &value {
			TrieStreamValue::Inline(value) => {
				Compact(value.len() as u32).encode_to(&mut self.buffer);
				self.buffer.extend_from_slice(value);
			},
			TrieStreamValue::Node(hash) => {
				self.buffer.extend_from_slice(hash.as_slice());
			},
		};
	}

	fn begin_branch(
		&mut self,
		maybe_partial: Option<&[u8]>,
		maybe_value: Option<TrieStreamValue>,
		has_children: impl Iterator<Item = bool>,
	) {
		if let Some(partial) = maybe_partial {
			let kind = match &maybe_value {
				None => NodeKind::BranchNoValue,
				Some(TrieStreamValue::Inline(..)) => NodeKind::BranchWithValue,
				Some(TrieStreamValue::Node(..)) => NodeKind::HashedValueBranch,
			};

			self.buffer.extend(fuse_nibbles_node(partial, kind));
			let bm = branch_node_bit_mask(has_children);
			self.buffer.extend([bm.0, bm.1].iter());
		} else {
			unreachable!("trie stream codec only for no extension trie");
		}
		match maybe_value {
			None => (),
			Some(TrieStreamValue::Inline(value)) => {
				Compact(value.len() as u32).encode_to(&mut self.buffer);
				self.buffer.extend_from_slice(value);
			},
			Some(TrieStreamValue::Node(hash)) => {
				self.buffer.extend_from_slice(hash.as_slice());
			},
		}
	}

	fn append_extension(&mut self, _key: &[u8]) {
		unreachable!("trie stream codec only for no extension trie");
	}

	fn append_substream<H: Hasher>(&mut self, other: Self) {
		let data = other.out();
		match data.len() {
			0..=31 => data.encode_to(&mut self.buffer),
			_ => H::hash(&data).as_ref().encode_to(&mut self.buffer),
		}
	}

	fn out(self) -> Vec<u8> {
		self.buffer
	}
}

#[test]
fn show_empty() {
	use parity_scale_codec::Decode;
	let pb_proof: &[u8] = &[
		128, 69, 25, 20, 0, 0, 0, 0, 0, 0, 0, 66, 219, 180, 62, 0, 2, 0, 0, 0, 1, 0, 0, 0, 44, 1,
		234, 3, 0, 0, 10, 97, 63, 31, 17, 188, 162, 154, 67, 154, 97, 17, 109, 131, 112, 144, 211,
		70, 167, 125, 215, 130, 41, 175, 59, 206, 126, 136, 3, 5, 175, 209, 146, 123, 26, 195, 215,
		22, 117, 140, 74, 6, 242, 114, 200, 159, 116, 243, 103, 74, 234, 200, 55, 82, 165, 29, 229,
		226, 204, 201, 221, 6, 49, 42, 69, 44, 36, 38, 28, 87, 198, 143, 220, 133, 245, 134, 203,
		212, 95, 234, 61, 191, 97, 91, 10, 201, 53, 199, 90, 226, 45, 65, 66, 172, 252, 145, 253,
		150, 85, 195, 76, 5, 96, 220, 123, 221, 163, 84, 219, 179, 18, 31, 177, 239, 154, 186, 254,
		114, 49, 97, 52, 58, 1, 61, 152, 89, 179, 255, 153, 130, 147, 159, 133, 59, 50, 70, 26, 83,
		61, 179, 64, 122, 22, 163, 30, 114, 2, 23, 191, 249, 248, 94, 1, 213, 60, 174, 82, 88, 31,
		232, 104, 149, 96, 21, 120, 30, 156, 44, 3, 24, 238, 90, 140, 109, 45, 239, 151, 185, 249,
		54, 225, 147, 41, 62, 100, 179, 231, 214, 186, 251, 169, 68, 125, 33, 250, 200, 31, 67, 7,
		125, 75, 105, 7, 243, 128, 206, 202, 241, 192, 166, 27, 77, 3, 232, 237, 59, 133, 149, 218,
		82, 138, 170, 253, 137, 158, 199, 89, 27, 255, 162, 244, 140, 13, 212, 169, 118, 81, 138,
		173, 134, 3, 20, 226, 199, 95, 201, 93, 64, 245, 230, 80, 86, 69, 58, 227, 82, 78, 113, 78,
		97, 233, 179, 16, 5, 101, 174, 82, 76, 50, 97, 2, 248, 188, 23, 189, 62, 158, 115, 161,
		133, 184, 17, 106, 158, 150, 78, 102, 92, 186, 81, 29, 229, 60, 36, 194, 181, 244, 187, 55,
		247, 80, 103, 175, 210, 205, 73, 7, 179, 179, 61, 140, 62, 195, 157, 170, 204, 42, 72, 127,
		190, 233, 2, 119, 94, 245, 45, 24, 212, 163, 156, 163, 107, 178, 152, 155, 1, 97, 62, 225,
		193, 214, 198, 187, 234, 171, 235, 198, 17, 168, 19, 104, 233, 39, 85, 158, 91, 12, 0, 91,
		199, 31, 55, 42, 249, 161, 26, 248, 252, 56, 60, 30, 43, 94, 134, 126, 214, 119, 205, 201,
		208, 172, 172, 58, 155, 98, 155, 69, 17, 242, 124, 227, 62, 15, 246, 86, 96, 145, 143, 115,
		189, 200, 169, 70, 61, 31, 134, 89, 44, 229, 43, 46, 79, 127, 237, 153, 0, 43, 30, 194, 77,
		241, 219, 8, 6, 97, 117, 114, 97, 32, 129, 45, 83, 8, 0, 0, 0, 0, 5, 97, 117, 114, 97, 1,
		1, 226, 13, 152, 88, 5, 191, 3, 248, 173, 153, 30, 132, 109, 234, 13, 7, 205, 36, 212, 208,
		99, 222, 133, 23, 230, 207, 199, 144, 208, 36, 86, 86, 114, 146, 40, 148, 199, 241, 8, 136,
		181, 253, 192, 79, 149, 238, 81, 68, 121, 224, 74, 255, 93, 127, 14, 68, 192, 82, 32, 41,
		80, 100, 57, 140, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 44, 0, 232, 3, 0, 0, 26, 103,
		172, 149, 195, 62, 184, 87, 32, 58, 167, 48, 34, 91, 222, 120, 145, 143, 134, 251, 70, 58,
		213, 22, 122, 37, 226, 7, 195, 240, 175, 150, 152, 116, 39, 171, 160, 149, 136, 225, 131,
		183, 239, 7, 46, 246, 48, 72, 110, 36, 91, 107, 139, 159, 202, 167, 34, 211, 98, 94, 130,
		87, 178, 91, 232, 212, 166, 194, 104, 63, 91, 233, 77, 143, 84, 27, 150, 39, 221, 181, 66,
		137, 84, 129, 234, 153, 212, 104, 20, 33, 179, 48, 233, 9, 54, 192, 125, 134, 39, 34, 139,
		243, 222, 82, 217, 118, 89, 217, 3, 151, 154, 210, 135, 33, 176, 145, 236, 109, 24, 54, 68,
		101, 60, 230, 156, 80, 225, 25, 147, 129, 87, 105, 66, 17, 240, 43, 237, 194, 171, 22, 128,
		182, 178, 203, 74, 171, 86, 12, 34, 239, 217, 115, 63, 14, 219, 69, 224, 57, 194, 155, 174,
		115, 155, 183, 13, 56, 9, 175, 124, 221, 9, 72, 190, 252, 106, 62, 173, 88, 182, 81, 255,
		26, 138, 239, 171, 86, 97, 143, 139, 59, 90, 111, 62, 115, 35, 32, 179, 63, 56, 250, 169,
		6, 222, 169, 33, 138, 142, 154, 16, 61, 247, 56, 45, 155, 128, 100, 126, 74, 15, 0, 18,
		212, 169, 142, 164, 71, 3, 227, 177, 239, 232, 112, 40, 98, 202, 77, 245, 112, 178, 146,
		56, 28, 197, 30, 76, 106, 162, 249, 110, 76, 148, 15, 246, 25, 202, 34, 7, 244, 197, 185,
		195, 166, 208, 1, 165, 77, 154, 150, 188, 27, 14, 105, 233, 38, 107, 250, 75, 240, 248,
		186, 11, 98, 86, 202, 16, 194, 61, 30, 197, 11, 196, 237, 56, 191, 77, 97, 190, 131, 114,
		234, 86, 75, 229, 8, 92, 15, 234, 40, 248, 234, 253, 56, 113, 37, 233, 117, 52, 134, 100,
		8, 233, 2, 201, 231, 209, 151, 96, 7, 68, 75, 199, 26, 237, 146, 40, 137, 171, 66, 32, 216,
		10, 116, 249, 55, 234, 16, 133, 231, 192, 4, 223, 45, 72, 158, 206, 29, 240, 0, 254, 131,
		18, 78, 135, 148, 160, 215, 219, 43, 208, 211, 237, 70, 146, 210, 218, 169, 179, 5, 137,
		88, 98, 103, 168, 65, 19, 30, 230, 123, 120, 2, 73, 15, 43, 187, 90, 14, 77, 169, 102, 147,
		98, 29, 177, 15, 224, 175, 113, 144, 141, 140, 16, 73, 125, 25, 35, 253, 222, 193, 78, 217,
		231, 8, 8, 6, 97, 117, 114, 97, 32, 130, 45, 83, 8, 0, 0, 0, 0, 5, 97, 117, 114, 97, 1, 1,
		186, 31, 140, 233, 81, 207, 239, 239, 27, 62, 123, 201, 70, 139, 28, 66, 162, 72, 8, 76,
		186, 87, 6, 184, 49, 208, 114, 173, 174, 240, 71, 55, 149, 221, 45, 116, 34, 133, 33, 27,
		217, 0, 179, 122, 65, 206, 57, 50, 76, 11, 170, 187, 67, 127, 128, 135, 203, 78, 191, 22,
		239, 212, 213, 139, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 44, 0, 233, 3, 0, 0, 26, 103,
		172, 149, 195, 62, 184, 87, 32, 58, 167, 48, 34, 91, 222, 120, 145, 143, 134, 251, 70, 58,
		213, 22, 122, 37, 226, 7, 195, 240, 175, 150, 114, 114, 74, 41, 26, 38, 111, 213, 133, 159,
		153, 127, 161, 83, 255, 10, 99, 132, 25, 164, 40, 237, 155, 119, 202, 161, 151, 0, 177,
		162, 65, 91, 1, 221, 45, 52, 55, 138, 17, 103, 182, 178, 117, 225, 146, 227, 40, 109, 67,
		47, 174, 148, 185, 89, 172, 238, 152, 136, 245, 36, 176, 222, 33, 185, 109, 203, 152, 246,
		174, 253, 11, 11, 244, 145, 199, 114, 100, 225, 72, 205, 210, 159, 180, 20, 130, 245, 212,
		57, 6, 35, 57, 104, 228, 6, 125, 48, 161, 232, 69, 3, 22, 133, 48, 150, 12, 250, 32, 35,
		98, 188, 194, 184, 1, 250, 228, 183, 183, 82, 127, 160, 160, 132, 252, 147, 208, 67, 46,
		154, 40, 104, 209, 222, 98, 56, 155, 210, 111, 2, 241, 218, 1, 64, 222, 242, 7, 74, 57,
		227, 9, 232, 31, 43, 189, 184, 48, 250, 171, 62, 223, 28, 70, 24, 124, 237, 139, 157, 247,
		29, 2, 143, 154, 137, 194, 219, 188, 128, 100, 195, 121, 122, 180, 64, 255, 194, 149, 122,
		230, 73, 246, 90, 162, 137, 133, 50, 204, 155, 132, 144, 198, 183, 95, 108, 120, 37, 176,
		99, 14, 236, 99, 14, 28, 169, 222, 46, 40, 182, 250, 114, 166, 4, 131, 47, 50, 62, 220,
		198, 205, 164, 155, 81, 239, 190, 15, 209, 132, 185, 116, 7, 77, 18, 88, 202, 159, 21, 185,
		252, 207, 237, 195, 91, 125, 238, 110, 26, 185, 233, 95, 177, 5, 198, 64, 39, 139, 251, 33,
		231, 59, 153, 122, 145, 123, 64, 192, 128, 31, 248, 202, 8, 46, 121, 200, 13, 219, 58, 43,
		202, 45, 3, 233, 2, 82, 232, 177, 97, 179, 167, 189, 89, 85, 12, 88, 10, 79, 134, 62, 125,
		50, 249, 100, 255, 45, 148, 147, 158, 228, 213, 201, 25, 101, 5, 251, 218, 150, 233, 47, 0,
		38, 3, 141, 108, 179, 128, 85, 43, 78, 160, 38, 175, 77, 9, 64, 47, 199, 178, 176, 104,
		143, 227, 86, 78, 89, 177, 215, 83, 6, 60, 153, 59, 116, 69, 169, 213, 2, 129, 39, 248, 10,
		154, 194, 3, 118, 52, 20, 68, 135, 11, 124, 114, 160, 179, 84, 252, 12, 239, 234, 135, 249,
		137, 154, 169, 8, 6, 97, 117, 114, 97, 32, 130, 45, 83, 8, 0, 0, 0, 0, 5, 97, 117, 114, 97,
		1, 1, 18, 245, 87, 39, 215, 101, 20, 179, 26, 70, 127, 108, 151, 130, 182, 228, 245, 109,
		182, 113, 95, 182, 82, 116, 65, 73, 171, 61, 183, 193, 223, 89, 151, 164, 108, 62, 36, 48,
		253, 133, 63, 245, 169, 180, 81, 5, 243, 20, 236, 136, 126, 213, 248, 139, 23, 220, 194,
		183, 185, 184, 170, 163, 163, 128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 7, 189,
		150, 70, 98, 12, 0, 2, 0, 0, 128, 26, 103, 172, 149, 195, 62, 184, 87, 32, 58, 167, 48, 34,
		91, 222, 120, 145, 143, 134, 251, 70, 58, 213, 22, 122, 37, 226, 7, 195, 240, 175, 150, 65,
		1, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 34, 148, 8, 96, 18, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 192, 62, 212, 30, 94,
		22, 5, 103, 101, 188, 132, 97, 133, 16, 114, 201, 215, 56, 60, 221, 4, 42, 42, 64, 203, 15,
		146, 148, 113, 62, 4, 190, 205, 61, 140, 105, 112, 241, 221, 210, 95, 20, 197, 183, 129,
		246, 93, 117, 2, 65, 1, 63, 60, 12, 52, 190, 59, 149, 251, 180, 232, 29, 203, 74, 169, 195,
		235, 146, 114, 110, 225, 182, 255, 172, 224, 159, 167, 104, 53, 206, 224, 72, 159, 51, 140,
		101, 39, 170, 188, 129, 12, 239, 235, 125, 91, 145, 161, 50, 41, 248, 83, 173, 96, 175, 80,
		220, 37, 224, 115, 44, 76, 52, 18, 19, 219, 196, 153, 238, 13, 133, 55, 196, 251, 139, 206,
		61, 95, 59, 101, 247, 172, 176, 84, 157, 208, 42, 113, 162, 209, 80, 165, 222, 0, 128, 26,
		103, 172, 149, 195, 62, 184, 87, 32, 58, 167, 48, 34, 91, 222, 120, 145, 143, 134, 251, 70,
		58, 213, 22, 122, 37, 226, 7, 195, 240, 175, 150, 180, 85, 14, 242, 92, 253, 166, 239, 58,
		0, 0, 0, 0, 128, 225, 67, 242, 56, 3, 172, 80, 232, 246, 248, 230, 38, 149, 209, 206, 158,
		78, 29, 104, 170, 54, 193, 205, 44, 253, 21, 52, 2, 19, 243, 66, 62, 132, 94, 46, 223, 59,
		223, 56, 29, 235, 227, 49, 171, 116, 70, 173, 223, 220, 64, 0, 0, 138, 93, 120, 69, 99, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 136, 95, 7, 200, 117, 228, 207, 247, 65, 72, 228, 98, 143, 38, 75,
		151, 76, 128, 64, 121, 176, 198, 244, 4, 125, 124, 47, 0, 0, 0, 0, 0, 0, 0, 0, 21, 1, 128,
		0, 96, 128, 135, 252, 174, 99, 183, 143, 102, 23, 25, 202, 137, 126, 35, 216, 209, 13, 221,
		140, 239, 110, 75, 200, 7, 236, 195, 205, 2, 242, 184, 108, 107, 77, 128, 13, 148, 252,
		225, 180, 145, 1, 186, 101, 25, 219, 146, 247, 230, 195, 68, 100, 199, 20, 19, 153, 246,
		73, 120, 78, 150, 68, 123, 30, 12, 92, 103, 21, 1, 128, 1, 4, 128, 236, 1, 146, 215, 103,
		191, 226, 7, 167, 34, 106, 203, 118, 214, 179, 140, 82, 245, 156, 135, 24, 222, 65, 53, 86,
		228, 61, 247, 4, 26, 36, 103, 128, 12, 247, 18, 162, 99, 133, 82, 117, 52, 5, 99, 36, 14,
		45, 156, 53, 62, 103, 184, 40, 40, 149, 237, 178, 224, 40, 233, 60, 114, 234, 37, 32, 220,
		128, 1, 128, 72, 94, 140, 233, 97, 93, 224, 119, 90, 130, 248, 169, 77, 195, 210, 133, 161,
		4, 1, 128, 204, 20, 194, 206, 77, 73, 53, 117, 36, 56, 161, 147, 164, 73, 104, 116, 189,
		38, 199, 27, 115, 82, 17, 32, 157, 231, 152, 135, 233, 216, 94, 148, 29, 2, 128, 2, 164,
		128, 214, 39, 26, 76, 42, 3, 157, 222, 15, 165, 70, 220, 57, 158, 212, 142, 149, 168, 251,
		105, 137, 48, 66, 190, 238, 161, 132, 221, 29, 159, 59, 72, 128, 196, 21, 177, 37, 131, 22,
		108, 172, 244, 126, 220, 241, 139, 190, 57, 21, 104, 35, 93, 37, 135, 25, 79, 61, 65, 228,
		126, 222, 165, 96, 106, 15, 128, 216, 68, 244, 66, 196, 2, 208, 185, 88, 36, 149, 139, 232,
		73, 196, 118, 33, 45, 244, 28, 222, 59, 204, 206, 96, 103, 57, 77, 1, 225, 86, 153, 128,
		211, 56, 180, 136, 10, 140, 61, 211, 54, 168, 97, 31, 52, 91, 194, 15, 211, 211, 218, 0,
		108, 33, 68, 98, 32, 216, 169, 134, 58, 74, 203, 82, 188, 128, 4, 4, 84, 94, 165, 193, 177,
		154, 183, 160, 79, 83, 108, 81, 154, 202, 73, 131, 172, 16, 81, 165, 222, 0, 84, 94, 152,
		253, 190, 156, 230, 197, 88, 55, 87, 108, 96, 199, 175, 56, 80, 16, 5, 0, 0, 0, 161, 2,
		128, 4, 108, 128, 202, 70, 147, 40, 74, 186, 30, 48, 35, 98, 221, 211, 15, 121, 140, 13,
		211, 246, 25, 227, 248, 53, 120, 22, 129, 69, 201, 171, 214, 18, 225, 245, 128, 197, 116,
		89, 16, 181, 118, 12, 74, 79, 19, 89, 67, 229, 81, 149, 61, 61, 171, 185, 78, 82, 121, 81,
		77, 106, 182, 234, 241, 123, 145, 207, 2, 128, 143, 131, 240, 216, 230, 57, 158, 159, 157,
		16, 195, 200, 155, 97, 78, 49, 136, 72, 223, 183, 236, 234, 216, 155, 19, 122, 177, 254,
		47, 169, 104, 249, 128, 235, 86, 115, 99, 115, 106, 16, 175, 105, 146, 201, 188, 63, 134,
		34, 178, 180, 67, 143, 34, 33, 243, 104, 195, 255, 185, 9, 8, 107, 194, 224, 89, 128, 209,
		73, 88, 220, 60, 171, 224, 243, 65, 69, 101, 65, 247, 191, 164, 136, 199, 97, 180, 70, 242,
		32, 154, 205, 209, 12, 6, 12, 108, 38, 106, 0, 161, 2, 128, 5, 193, 128, 138, 81, 68, 108,
		215, 185, 168, 8, 130, 232, 204, 7, 45, 90, 134, 223, 118, 39, 245, 0, 58, 237, 21, 19, 64,
		88, 103, 32, 110, 164, 49, 215, 128, 0, 174, 20, 89, 189, 50, 193, 78, 240, 213, 202, 229,
		180, 164, 142, 17, 116, 72, 237, 167, 232, 61, 218, 25, 42, 243, 65, 30, 91, 95, 208, 225,
		128, 24, 229, 43, 127, 211, 11, 68, 100, 104, 149, 49, 110, 117, 125, 78, 82, 64, 171, 132,
		209, 168, 41, 88, 76, 227, 99, 45, 233, 171, 228, 158, 105, 128, 248, 102, 26, 157, 230,
		228, 6, 44, 198, 47, 227, 94, 59, 161, 207, 225, 22, 182, 79, 11, 131, 223, 176, 211, 104,
		0, 94, 119, 193, 43, 94, 80, 128, 125, 125, 42, 10, 249, 175, 67, 120, 244, 27, 94, 231, 8,
		129, 146, 214, 238, 226, 149, 95, 91, 5, 87, 247, 84, 203, 226, 115, 121, 87, 68, 197, 153,
		1, 128, 6, 8, 128, 136, 108, 24, 242, 32, 46, 169, 181, 189, 245, 9, 223, 57, 203, 44, 24,
		232, 185, 213, 227, 37, 47, 130, 107, 213, 115, 202, 5, 221, 92, 230, 134, 128, 129, 176,
		69, 12, 48, 157, 226, 86, 165, 95, 236, 1, 68, 235, 27, 175, 64, 218, 174, 134, 235, 82, 1,
		230, 136, 94, 71, 209, 27, 77, 134, 11, 128, 176, 159, 243, 25, 111, 2, 129, 11, 209, 39,
		201, 87, 223, 74, 34, 43, 88, 231, 39, 188, 85, 82, 203, 244, 243, 210, 130, 195, 11, 216,
		229, 116, 45, 4, 128, 27, 170, 128, 185, 171, 36, 86, 149, 1, 3, 141, 165, 112, 58, 151,
		76, 173, 42, 194, 99, 3, 128, 111, 24, 73, 218, 188, 98, 248, 131, 2, 133, 144, 13, 199,
		128, 228, 47, 101, 240, 208, 126, 232, 117, 244, 214, 251, 32, 156, 100, 187, 172, 176, 43,
		66, 11, 65, 31, 151, 60, 144, 38, 31, 144, 224, 8, 231, 48, 128, 181, 138, 162, 84, 16, 95,
		173, 29, 68, 36, 114, 41, 14, 202, 49, 118, 122, 217, 111, 17, 232, 90, 245, 209, 156, 24,
		166, 226, 198, 125, 50, 60, 128, 180, 252, 218, 186, 20, 149, 33, 62, 174, 164, 20, 79,
		142, 47, 223, 77, 59, 119, 217, 6, 66, 156, 228, 10, 214, 197, 58, 229, 191, 55, 190, 89,
		128, 106, 65, 233, 17, 197, 158, 227, 138, 206, 99, 55, 43, 253, 150, 216, 70, 203, 204,
		36, 240, 240, 33, 251, 20, 72, 164, 241, 35, 149, 115, 73, 11, 128, 88, 4, 127, 107, 206,
		29, 202, 251, 192, 11, 178, 154, 175, 76, 87, 186, 41, 125, 40, 77, 15, 120, 103, 61, 255,
		250, 249, 141, 154, 148, 75, 210, 128, 127, 100, 10, 54, 144, 120, 83, 126, 116, 134, 152,
		16, 60, 126, 95, 236, 68, 15, 58, 255, 88, 132, 68, 45, 175, 92, 240, 135, 181, 235, 247,
		244, 128, 72, 141, 204, 215, 77, 170, 223, 143, 214, 98, 4, 187, 108, 36, 250, 110, 102,
		85, 43, 149, 144, 65, 30, 22, 182, 17, 127, 209, 85, 112, 86, 102, 220, 128, 144, 0, 128,
		156, 202, 195, 220, 94, 182, 241, 128, 86, 53, 41, 25, 27, 157, 5, 11, 147, 52, 136, 217,
		143, 218, 241, 184, 200, 252, 21, 245, 13, 181, 132, 223, 72, 94, 253, 108, 40, 131, 107,
		154, 40, 82, 45, 201, 36, 17, 12, 244, 57, 4, 1, 169, 3, 128, 193, 141, 128, 18, 4, 108,
		63, 183, 78, 54, 92, 140, 175, 78, 128, 15, 42, 142, 50, 111, 178, 65, 76, 50, 93, 177,
		155, 220, 221, 17, 203, 108, 97, 238, 106, 128, 74, 102, 177, 203, 163, 57, 180, 237, 114,
		65, 106, 64, 138, 155, 227, 102, 205, 169, 15, 241, 41, 58, 214, 5, 8, 228, 235, 46, 251,
		17, 242, 70, 128, 37, 116, 66, 163, 25, 62, 191, 100, 251, 138, 0, 25, 178, 217, 252, 29,
		151, 115, 64, 17, 235, 90, 141, 46, 164, 20, 245, 255, 154, 39, 149, 43, 128, 203, 237,
		154, 160, 61, 0, 112, 179, 124, 62, 27, 194, 51, 108, 158, 32, 103, 153, 144, 156, 75, 62,
		177, 217, 57, 220, 158, 50, 182, 119, 152, 164, 128, 184, 20, 180, 45, 80, 204, 175, 220,
		80, 10, 146, 224, 160, 117, 199, 30, 147, 215, 228, 90, 109, 205, 47, 94, 34, 176, 69, 150,
		49, 174, 130, 179, 128, 16, 240, 45, 90, 236, 253, 84, 125, 1, 183, 60, 84, 104, 76, 209,
		67, 22, 128, 221, 168, 142, 3, 85, 123, 216, 4, 11, 145, 119, 212, 90, 28, 128, 162, 223,
		103, 50, 12, 6, 132, 10, 228, 240, 210, 60, 145, 169, 99, 142, 160, 230, 153, 199, 203, 96,
		126, 132, 167, 222, 174, 157, 43, 175, 112, 213, 177, 4, 128, 205, 39, 128, 143, 164, 47,
		29, 91, 241, 135, 211, 133, 90, 162, 31, 195, 159, 35, 106, 158, 138, 127, 241, 15, 233,
		63, 159, 208, 191, 94, 135, 87, 180, 219, 28, 128, 183, 184, 33, 28, 192, 56, 113, 70, 145,
		139, 85, 230, 147, 159, 185, 161, 191, 136, 166, 181, 61, 118, 223, 176, 184, 242, 119,
		170, 202, 153, 75, 156, 128, 71, 92, 71, 55, 64, 126, 185, 143, 178, 211, 211, 1, 145, 21,
		186, 141, 233, 104, 232, 197, 73, 174, 76, 238, 73, 227, 0, 110, 176, 246, 206, 237, 128,
		156, 97, 240, 184, 95, 148, 252, 219, 13, 153, 250, 85, 157, 237, 113, 58, 5, 212, 211,
		127, 15, 246, 86, 54, 184, 10, 160, 1, 53, 190, 202, 243, 128, 198, 57, 51, 84, 63, 121,
		22, 87, 53, 117, 0, 72, 3, 140, 47, 143, 202, 251, 205, 6, 44, 50, 138, 102, 76, 182, 241,
		51, 78, 113, 5, 89, 128, 7, 155, 125, 213, 17, 42, 54, 226, 53, 75, 57, 204, 242, 2, 31,
		102, 27, 57, 171, 62, 91, 217, 191, 87, 249, 188, 75, 86, 238, 116, 10, 95, 128, 171, 40,
		176, 113, 148, 172, 241, 31, 145, 179, 37, 180, 130, 89, 13, 243, 130, 82, 182, 176, 26,
		82, 92, 8, 153, 165, 13, 250, 216, 177, 186, 28, 128, 120, 122, 18, 124, 210, 11, 206, 42,
		192, 240, 177, 241, 145, 42, 246, 213, 213, 218, 72, 45, 209, 190, 255, 245, 53, 43, 223,
		52, 217, 5, 224, 40, 128, 145, 50, 94, 31, 219, 62, 224, 83, 127, 180, 105, 127, 209, 173,
		141, 168, 179, 73, 25, 85, 178, 109, 217, 249, 152, 74, 220, 105, 36, 87, 170, 9, 77, 8,
		128, 255, 255, 128, 1, 211, 203, 20, 241, 47, 57, 139, 92, 193, 190, 218, 148, 11, 32, 165,
		37, 93, 172, 186, 189, 10, 89, 255, 254, 97, 1, 135, 54, 81, 146, 229, 128, 70, 124, 105,
		74, 174, 69, 235, 121, 21, 17, 36, 238, 63, 157, 92, 11, 43, 236, 133, 1, 64, 51, 39, 68,
		127, 104, 79, 71, 224, 8, 246, 172, 128, 150, 180, 80, 121, 87, 212, 119, 66, 219, 189, 43,
		102, 67, 149, 75, 140, 51, 172, 118, 158, 62, 193, 16, 203, 157, 255, 236, 137, 110, 12,
		172, 255, 128, 113, 19, 104, 185, 100, 43, 248, 184, 83, 140, 164, 239, 164, 90, 35, 155,
		192, 189, 129, 213, 117, 88, 121, 187, 53, 223, 138, 244, 102, 172, 254, 54, 128, 122, 210,
		225, 53, 187, 31, 70, 237, 89, 167, 29, 155, 50, 13, 209, 16, 31, 160, 140, 173, 16, 128,
		49, 188, 191, 191, 204, 40, 253, 234, 7, 70, 128, 120, 225, 153, 145, 42, 136, 238, 43,
		180, 111, 33, 10, 255, 116, 216, 172, 63, 217, 108, 129, 25, 80, 226, 175, 148, 64, 54,
		145, 76, 151, 10, 117, 128, 203, 229, 144, 80, 56, 237, 128, 172, 60, 150, 144, 34, 239,
		209, 44, 240, 168, 225, 245, 213, 67, 181, 99, 153, 10, 54, 247, 188, 180, 3, 101, 49, 128,
		137, 48, 48, 124, 169, 20, 66, 214, 6, 12, 59, 171, 155, 20, 30, 200, 245, 85, 87, 253,
		224, 8, 247, 225, 236, 142, 190, 75, 238, 77, 125, 18, 128, 66, 95, 142, 4, 35, 78, 216,
		10, 50, 87, 176, 124, 18, 251, 131, 188, 202, 210, 225, 231, 119, 45, 145, 76, 130, 177,
		40, 39, 94, 149, 135, 213, 128, 109, 236, 209, 34, 48, 249, 102, 85, 26, 176, 83, 18, 37,
		67, 58, 232, 116, 217, 0, 1, 97, 179, 44, 93, 130, 157, 209, 236, 42, 157, 244, 228, 128,
		52, 65, 230, 33, 77, 218, 8, 97, 86, 178, 21, 183, 78, 248, 252, 167, 29, 60, 85, 51, 1,
		91, 72, 178, 222, 240, 178, 197, 110, 215, 87, 74, 128, 57, 94, 158, 180, 9, 59, 206, 182,
		248, 0, 183, 119, 74, 110, 71, 132, 254, 231, 83, 123, 158, 159, 230, 140, 120, 1, 198,
		148, 36, 139, 43, 33, 128, 110, 20, 12, 174, 201, 235, 232, 207, 136, 188, 71, 190, 2, 76,
		156, 62, 172, 54, 22, 12, 104, 40, 250, 198, 83, 24, 18, 39, 37, 73, 92, 113, 128, 125,
		215, 134, 25, 217, 59, 63, 13, 229, 87, 221, 10, 14, 122, 26, 140, 69, 160, 155, 8, 245,
		173, 234, 12, 137, 133, 9, 239, 29, 223, 107, 247, 128, 185, 96, 103, 209, 128, 83, 73, 60,
		72, 205, 136, 139, 132, 202, 136, 166, 147, 15, 91, 94, 41, 14, 84, 44, 110, 61, 71, 43,
		153, 222, 177, 209, 128, 172, 55, 16, 95, 18, 252, 184, 210, 89, 102, 137, 249, 116, 63,
		103, 141, 76, 99, 114, 218, 223, 29, 132, 217, 115, 78, 165, 129, 226, 69, 5, 61, 77, 8,
		128, 255, 255, 128, 37, 145, 248, 95, 69, 117, 161, 1, 45, 8, 242, 143, 84, 15, 196, 107,
		225, 155, 149, 120, 156, 104, 37, 30, 139, 78, 88, 165, 177, 211, 143, 78, 128, 96, 158,
		193, 56, 56, 67, 154, 50, 189, 63, 110, 88, 195, 252, 22, 162, 68, 129, 4, 221, 171, 81,
		34, 67, 185, 17, 178, 136, 27, 93, 158, 102, 128, 154, 31, 222, 159, 224, 161, 102, 29, 77,
		81, 163, 79, 21, 238, 142, 99, 239, 87, 96, 61, 236, 196, 177, 163, 74, 103, 104, 160, 231,
		152, 117, 146, 128, 18, 178, 94, 115, 129, 133, 179, 198, 67, 58, 98, 247, 150, 89, 107,
		190, 241, 59, 210, 248, 87, 155, 116, 26, 154, 236, 158, 79, 176, 120, 88, 117, 128, 233,
		122, 122, 109, 171, 113, 146, 93, 15, 117, 112, 62, 168, 217, 114, 197, 214, 88, 156, 224,
		183, 35, 116, 24, 242, 17, 65, 132, 155, 102, 112, 193, 128, 51, 67, 133, 77, 115, 79, 177,
		135, 17, 181, 155, 61, 209, 233, 112, 54, 207, 156, 250, 185, 7, 131, 28, 102, 148, 231,
		151, 24, 197, 182, 11, 61, 128, 160, 210, 212, 137, 222, 199, 6, 28, 56, 182, 32, 147, 169,
		207, 174, 101, 28, 65, 4, 107, 194, 155, 9, 12, 27, 63, 72, 81, 35, 97, 45, 23, 128, 91,
		19, 22, 225, 177, 0, 52, 43, 246, 223, 81, 242, 89, 78, 240, 187, 41, 85, 230, 62, 19, 188,
		43, 125, 26, 233, 137, 57, 217, 54, 147, 175, 128, 48, 43, 25, 115, 167, 252, 217, 68, 17,
		238, 80, 127, 216, 56, 110, 50, 32, 161, 228, 58, 59, 70, 56, 249, 11, 103, 246, 59, 219,
		29, 49, 240, 128, 30, 224, 92, 44, 4, 40, 135, 230, 249, 145, 123, 108, 10, 234, 176, 47,
		69, 191, 224, 213, 118, 172, 36, 18, 219, 137, 74, 170, 47, 202, 212, 25, 128, 60, 244, 20,
		57, 59, 127, 98, 84, 236, 3, 227, 187, 252, 104, 29, 157, 16, 51, 110, 7, 26, 183, 219,
		147, 43, 123, 73, 44, 191, 90, 35, 239, 128, 62, 155, 120, 71, 186, 135, 71, 67, 100, 122,
		122, 236, 124, 13, 58, 71, 237, 134, 28, 202, 104, 33, 99, 255, 19, 27, 131, 77, 133, 138,
		17, 159, 128, 1, 140, 147, 15, 196, 222, 218, 198, 63, 43, 240, 172, 223, 51, 242, 66, 135,
		193, 74, 142, 55, 11, 15, 83, 2, 47, 155, 134, 141, 44, 252, 174, 128, 45, 94, 41, 67, 160,
		200, 92, 27, 142, 89, 68, 240, 43, 46, 57, 63, 35, 122, 148, 45, 96, 240, 1, 211, 106, 9,
		211, 86, 62, 153, 123, 138, 128, 111, 52, 19, 137, 151, 156, 96, 47, 131, 146, 246, 231,
		56, 132, 236, 13, 157, 153, 69, 86, 23, 215, 75, 200, 145, 30, 234, 62, 106, 6, 211, 169,
		128, 2, 79, 163, 176, 196, 242, 12, 231, 138, 80, 180, 176, 107, 61, 211, 161, 10, 200, 2,
		145, 105, 133, 123, 55, 169, 40, 87, 112, 252, 207, 125, 70, 77, 8, 128, 255, 255, 128, 52,
		94, 30, 137, 125, 126, 94, 73, 93, 101, 113, 193, 118, 205, 211, 167, 49, 181, 149, 110,
		210, 87, 140, 232, 222, 180, 167, 23, 158, 234, 132, 72, 128, 8, 242, 177, 58, 179, 37,
		240, 245, 3, 45, 2, 101, 45, 18, 38, 92, 92, 57, 92, 171, 140, 166, 99, 231, 74, 127, 177,
		100, 151, 191, 160, 161, 128, 216, 186, 26, 149, 30, 175, 118, 44, 186, 57, 57, 1, 174, 70,
		222, 74, 124, 125, 90, 155, 135, 169, 133, 25, 219, 3, 89, 67, 55, 6, 190, 81, 128, 68, 15,
		223, 98, 229, 223, 228, 189, 152, 107, 215, 148, 65, 25, 143, 57, 13, 221, 33, 144, 197,
		156, 255, 106, 62, 33, 11, 62, 60, 56, 168, 222, 128, 251, 238, 222, 29, 119, 147, 247,
		201, 62, 53, 67, 143, 40, 177, 16, 33, 88, 175, 58, 157, 200, 27, 78, 114, 64, 227, 212,
		193, 141, 7, 75, 55, 128, 176, 228, 188, 241, 111, 13, 137, 150, 236, 68, 167, 135, 142, 3,
		208, 110, 163, 9, 46, 160, 12, 177, 3, 231, 36, 22, 232, 29, 187, 245, 221, 88, 128, 209,
		13, 80, 245, 136, 201, 1, 89, 64, 233, 16, 98, 30, 133, 209, 60, 228, 39, 47, 81, 228, 30,
		157, 219, 91, 4, 119, 244, 19, 167, 34, 141, 128, 218, 154, 144, 22, 110, 154, 5, 90, 145,
		1, 120, 39, 147, 232, 96, 71, 248, 135, 101, 36, 3, 233, 35, 4, 107, 227, 235, 0, 165, 119,
		205, 6, 128, 91, 227, 41, 45, 153, 119, 47, 27, 87, 251, 207, 121, 68, 101, 229, 251, 95,
		7, 182, 86, 168, 215, 184, 35, 128, 47, 86, 37, 89, 145, 114, 175, 128, 150, 190, 120, 110,
		138, 91, 46, 37, 222, 221, 215, 20, 199, 111, 210, 95, 91, 8, 220, 166, 180, 232, 69, 53,
		114, 216, 19, 161, 216, 89, 88, 64, 128, 2, 18, 120, 185, 150, 235, 52, 175, 30, 93, 152,
		106, 244, 129, 159, 34, 142, 181, 129, 184, 42, 143, 250, 160, 81, 14, 248, 229, 240, 219,
		236, 245, 128, 64, 49, 103, 196, 140, 221, 112, 16, 61, 8, 97, 231, 233, 230, 7, 81, 197,
		134, 170, 20, 55, 236, 154, 91, 211, 252, 66, 28, 9, 138, 243, 92, 128, 182, 165, 35, 224,
		70, 70, 163, 232, 228, 158, 241, 48, 141, 136, 67, 47, 99, 252, 11, 250, 73, 57, 134, 95,
		171, 174, 250, 208, 138, 204, 99, 132, 128, 109, 199, 36, 71, 169, 98, 87, 73, 77, 212, 29,
		35, 104, 250, 18, 12, 138, 25, 114, 121, 164, 192, 238, 214, 233, 193, 104, 84, 237, 159,
		135, 235, 128, 194, 160, 47, 130, 234, 197, 69, 151, 252, 72, 201, 54, 210, 134, 64, 214,
		59, 225, 152, 140, 93, 160, 6, 63, 200, 65, 157, 20, 220, 147, 117, 92, 128, 171, 52, 156,
		157, 195, 74, 68, 126, 80, 240, 232, 170, 55, 137, 174, 121, 11, 1, 162, 143, 85, 146, 210,
		142, 177, 125, 125, 36, 5, 4, 162, 20, 77, 8, 128, 255, 255, 128, 55, 174, 70, 201, 175,
		198, 83, 15, 116, 85, 143, 214, 55, 91, 226, 36, 104, 169, 122, 41, 17, 116, 234, 166, 42,
		172, 67, 137, 57, 86, 249, 110, 128, 73, 93, 208, 109, 99, 110, 152, 24, 168, 25, 122, 201,
		112, 142, 53, 130, 200, 19, 155, 180, 201, 32, 133, 170, 145, 132, 66, 2, 6, 189, 207, 89,
		128, 20, 148, 144, 222, 175, 101, 173, 132, 142, 215, 40, 127, 202, 45, 220, 128, 173, 16,
		192, 168, 155, 17, 41, 70, 66, 100, 147, 179, 121, 243, 135, 91, 128, 130, 168, 14, 8, 97,
		146, 235, 24, 26, 118, 179, 79, 189, 211, 252, 66, 220, 128, 116, 152, 31, 191, 232, 250,
		228, 213, 186, 135, 6, 217, 122, 177, 128, 30, 140, 233, 130, 125, 48, 81, 210, 116, 15,
		227, 248, 191, 126, 19, 125, 87, 85, 206, 200, 222, 25, 36, 192, 6, 236, 211, 246, 251,
		169, 81, 57, 128, 221, 193, 163, 162, 181, 21, 173, 230, 39, 3, 200, 54, 61, 174, 97, 58,
		168, 13, 214, 244, 237, 99, 180, 101, 20, 109, 225, 73, 222, 244, 234, 213, 128, 136, 2,
		222, 135, 179, 131, 206, 111, 37, 121, 63, 68, 49, 97, 109, 87, 41, 180, 191, 65, 62, 129,
		28, 144, 30, 23, 132, 209, 124, 190, 158, 109, 128, 36, 78, 48, 191, 40, 245, 33, 29, 147,
		237, 218, 248, 56, 219, 253, 65, 5, 71, 184, 131, 82, 197, 116, 50, 85, 69, 4, 138, 105,
		113, 146, 241, 128, 203, 152, 137, 39, 179, 148, 153, 165, 2, 172, 78, 87, 173, 158, 21,
		20, 108, 126, 96, 118, 136, 10, 33, 69, 49, 6, 162, 127, 18, 159, 73, 129, 128, 29, 18,
		142, 20, 249, 31, 73, 143, 82, 246, 79, 101, 8, 81, 3, 254, 219, 21, 97, 216, 239, 144,
		226, 226, 151, 95, 102, 33, 109, 119, 112, 60, 128, 79, 134, 200, 173, 111, 140, 26, 136,
		253, 56, 252, 169, 8, 225, 244, 102, 152, 34, 126, 207, 248, 209, 25, 127, 60, 239, 220,
		228, 129, 51, 214, 103, 128, 7, 169, 129, 214, 203, 235, 199, 83, 154, 9, 236, 115, 149,
		173, 52, 83, 86, 236, 169, 191, 198, 192, 192, 127, 119, 74, 238, 37, 83, 21, 223, 168,
		128, 1, 192, 2, 129, 8, 137, 62, 150, 144, 77, 108, 173, 4, 77, 44, 38, 38, 135, 21, 0,
		162, 110, 94, 179, 138, 104, 184, 240, 185, 174, 150, 194, 128, 221, 2, 129, 184, 183, 114,
		239, 185, 74, 166, 209, 247, 51, 11, 156, 220, 22, 246, 2, 68, 128, 12, 41, 140, 54, 39,
		40, 217, 74, 135, 66, 128, 128, 1, 170, 167, 118, 129, 246, 80, 148, 210, 136, 104, 23,
		137, 37, 145, 242, 170, 69, 221, 138, 158, 235, 162, 134, 13, 61, 233, 191, 111, 183, 23,
		203, 128, 176, 111, 194, 117, 50, 215, 236, 57, 133, 218, 139, 217, 100, 248, 31, 9, 36,
		201, 175, 228, 38, 2, 129, 32, 40, 249, 92, 230, 73, 242, 254, 130, 77, 8, 128, 255, 255,
		128, 149, 235, 225, 113, 165, 152, 241, 195, 72, 149, 169, 8, 148, 103, 182, 152, 203, 105,
		102, 146, 39, 192, 23, 51, 14, 185, 179, 189, 200, 169, 24, 88, 128, 113, 31, 181, 99, 167,
		190, 161, 164, 234, 140, 82, 144, 225, 89, 162, 253, 204, 157, 63, 192, 237, 103, 202, 43,
		205, 220, 127, 126, 219, 44, 30, 78, 128, 119, 28, 103, 116, 116, 238, 153, 47, 197, 13,
		73, 74, 95, 96, 173, 98, 100, 25, 24, 142, 25, 78, 231, 247, 149, 213, 239, 104, 233, 229,
		228, 103, 128, 250, 100, 50, 97, 38, 111, 112, 154, 251, 139, 252, 98, 50, 206, 197, 188,
		149, 212, 66, 215, 154, 41, 63, 72, 240, 104, 93, 54, 2, 206, 197, 171, 128, 145, 232, 147,
		100, 46, 113, 230, 60, 4, 182, 250, 163, 76, 40, 217, 122, 207, 137, 61, 85, 221, 48, 145,
		96, 55, 125, 86, 198, 110, 106, 254, 234, 128, 247, 13, 208, 153, 106, 87, 148, 142, 143,
		90, 161, 83, 116, 134, 160, 180, 251, 171, 2, 114, 221, 245, 32, 37, 160, 166, 196, 14,
		143, 157, 219, 230, 128, 20, 221, 121, 213, 132, 180, 195, 17, 203, 61, 47, 68, 40, 42,
		207, 10, 138, 206, 198, 59, 93, 190, 26, 240, 153, 0, 83, 147, 115, 186, 9, 115, 128, 137,
		254, 145, 166, 206, 223, 129, 148, 47, 124, 125, 87, 7, 26, 181, 20, 56, 199, 67, 114, 66,
		15, 101, 86, 238, 141, 114, 249, 215, 67, 219, 15, 128, 174, 101, 93, 78, 85, 98, 186, 227,
		198, 228, 110, 91, 208, 233, 180, 91, 98, 131, 24, 94, 88, 196, 160, 22, 81, 239, 245, 244,
		120, 142, 21, 5, 128, 231, 153, 254, 154, 197, 140, 13, 155, 27, 103, 14, 65, 75, 156, 108,
		66, 241, 49, 166, 71, 112, 20, 56, 5, 149, 151, 107, 179, 39, 99, 41, 58, 128, 190, 248,
		229, 231, 6, 74, 114, 227, 20, 186, 83, 180, 128, 100, 161, 87, 126, 97, 20, 254, 166, 95,
		136, 133, 68, 255, 107, 108, 68, 47, 74, 94, 128, 246, 217, 128, 96, 216, 253, 181, 121,
		242, 162, 61, 119, 190, 197, 2, 235, 188, 131, 232, 108, 160, 19, 145, 187, 111, 254, 73,
		115, 66, 217, 90, 129, 128, 107, 65, 43, 74, 78, 6, 205, 4, 179, 68, 52, 188, 7, 31, 95,
		12, 218, 240, 224, 174, 75, 194, 241, 213, 3, 208, 49, 79, 223, 13, 146, 26, 128, 169, 230,
		166, 219, 199, 239, 118, 79, 254, 224, 237, 156, 8, 193, 169, 138, 233, 216, 55, 100, 116,
		105, 141, 195, 152, 82, 127, 170, 175, 218, 200, 67, 128, 42, 85, 19, 108, 224, 136, 94,
		105, 175, 180, 207, 130, 221, 196, 50, 82, 159, 96, 185, 53, 216, 33, 224, 251, 174, 46,
		197, 4, 181, 248, 227, 38, 128, 193, 84, 3, 218, 160, 153, 10, 187, 186, 47, 22, 11, 220,
		46, 191, 208, 144, 211, 39, 145, 245, 56, 82, 232, 114, 213, 91, 53, 232, 75, 0, 225, 77,
		8, 128, 255, 255, 128, 228, 116, 88, 106, 172, 212, 40, 113, 123, 164, 155, 198, 242, 160,
		180, 96, 116, 154, 220, 231, 43, 64, 96, 15, 148, 247, 151, 22, 20, 57, 234, 35, 128, 191,
		239, 26, 210, 119, 49, 211, 181, 171, 113, 208, 63, 65, 15, 207, 214, 83, 128, 175, 2, 142,
		188, 106, 74, 230, 154, 75, 207, 210, 61, 150, 40, 128, 219, 2, 240, 49, 244, 80, 70, 249,
		230, 223, 170, 124, 119, 90, 249, 9, 127, 43, 223, 61, 181, 102, 207, 52, 193, 34, 129,
		181, 228, 74, 185, 43, 128, 236, 21, 179, 160, 117, 166, 103, 15, 158, 241, 146, 68, 28,
		10, 199, 94, 247, 237, 195, 127, 23, 145, 115, 226, 242, 27, 32, 190, 224, 169, 114, 211,
		128, 68, 26, 139, 245, 10, 234, 20, 46, 80, 66, 226, 238, 114, 166, 59, 96, 45, 177, 202,
		18, 47, 13, 134, 226, 191, 215, 58, 170, 246, 100, 239, 123, 128, 250, 83, 39, 58, 228,
		133, 203, 130, 31, 238, 185, 20, 112, 202, 237, 101, 165, 104, 177, 78, 44, 215, 169, 124,
		122, 51, 100, 221, 185, 103, 138, 159, 128, 252, 55, 66, 102, 94, 121, 165, 48, 91, 52,
		141, 20, 4, 191, 129, 249, 235, 29, 185, 243, 6, 182, 111, 29, 33, 90, 134, 151, 13, 213,
		245, 168, 128, 72, 167, 73, 25, 32, 153, 164, 203, 187, 186, 42, 248, 52, 35, 146, 168,
		204, 103, 69, 135, 124, 83, 186, 71, 164, 170, 90, 53, 138, 150, 205, 34, 128, 159, 167,
		32, 148, 54, 22, 105, 97, 251, 36, 20, 204, 18, 189, 99, 123, 140, 158, 171, 135, 37, 114,
		182, 175, 181, 89, 194, 248, 186, 198, 146, 133, 128, 196, 122, 151, 147, 59, 236, 238,
		159, 34, 73, 153, 141, 146, 224, 170, 246, 236, 238, 193, 4, 67, 101, 140, 137, 142, 38,
		186, 61, 141, 183, 65, 250, 128, 11, 61, 172, 94, 112, 172, 107, 166, 217, 214, 165, 0, 72,
		39, 6, 119, 113, 232, 195, 165, 221, 63, 1, 106, 13, 199, 49, 254, 239, 201, 247, 73, 128,
		99, 221, 59, 180, 197, 191, 151, 22, 103, 69, 108, 131, 222, 24, 192, 165, 136, 13, 225,
		105, 25, 154, 203, 3, 246, 221, 34, 134, 211, 168, 164, 162, 128, 112, 224, 64, 254, 130,
		165, 208, 243, 107, 62, 208, 56, 180, 227, 129, 118, 253, 207, 207, 154, 183, 107, 102, 88,
		73, 134, 117, 80, 35, 19, 88, 210, 128, 97, 212, 221, 0, 124, 203, 56, 245, 10, 204, 222,
		168, 178, 119, 163, 79, 57, 39, 19, 196, 110, 169, 140, 101, 254, 1, 171, 103, 104, 209,
		207, 70, 128, 63, 5, 72, 126, 165, 217, 209, 121, 146, 220, 102, 173, 239, 204, 209, 230,
		186, 153, 99, 113, 167, 27, 63, 203, 216, 89, 253, 95, 172, 119, 140, 1, 128, 97, 148, 57,
		238, 11, 2, 41, 5, 205, 28, 153, 152, 186, 10, 136, 110, 248, 48, 245, 149, 137, 52, 245,
		139, 73, 148, 88, 174, 92, 92, 5, 37, 33, 1, 157, 4, 103, 160, 150, 188, 215, 26, 91, 106,
		12, 129, 85, 226, 8, 16, 24, 0, 128, 48, 26, 146, 52, 48, 28, 142, 88, 139, 4, 247, 183,
		94, 219, 206, 168, 253, 254, 239, 64, 69, 195, 112, 44, 138, 198, 147, 213, 216, 97, 66,
		225, 80, 95, 14, 123, 144, 18, 9, 107, 65, 196, 235, 58, 175, 148, 127, 110, 164, 41, 8, 0,
		0, 41, 2, 158, 38, 18, 118, 204, 157, 31, 133, 152, 234, 75, 106, 116, 177, 92, 47, 54, 0,
		128, 244, 109, 180, 102, 245, 75, 210, 54, 82, 112, 145, 195, 108, 76, 122, 55, 36, 201,
		131, 27, 200, 171, 231, 89, 120, 90, 42, 204, 45, 106, 83, 185, 128, 1, 196, 249, 248, 63,
		201, 68, 85, 6, 52, 87, 254, 48, 124, 178, 69, 194, 117, 63, 15, 106, 95, 75, 19, 188, 158,
		93, 241, 5, 232, 210, 56, 80, 95, 14, 123, 144, 18, 9, 107, 65, 196, 235, 58, 175, 148,
		127, 110, 164, 41, 8, 1, 0, 128, 184, 240, 97, 176, 67, 160, 111, 96, 187, 67, 189, 92, 59,
		110, 98, 219, 41, 147, 128, 145, 123, 203, 50, 54, 199, 33, 238, 84, 99, 115, 231, 210,
		137, 8, 158, 71, 4, 181, 104, 210, 22, 103, 53, 106, 90, 5, 12, 17, 135, 70, 255, 255, 128,
		70, 147, 132, 238, 121, 12, 35, 151, 203, 115, 177, 20, 49, 132, 96, 218, 136, 232, 2, 208,
		151, 194, 28, 173, 174, 109, 92, 136, 144, 199, 193, 93, 128, 195, 196, 106, 197, 126, 103,
		24, 199, 255, 250, 104, 29, 161, 229, 216, 115, 1, 251, 135, 86, 172, 234, 115, 195, 34,
		186, 115, 75, 25, 204, 181, 203, 128, 191, 41, 55, 107, 97, 14, 77, 203, 137, 62, 34, 193,
		120, 221, 139, 115, 199, 244, 165, 63, 152, 150, 251, 158, 185, 180, 111, 86, 17, 253, 29,
		142, 128, 169, 211, 136, 104, 47, 207, 251, 15, 49, 11, 50, 105, 173, 233, 186, 220, 40,
		193, 137, 65, 253, 150, 165, 45, 170, 101, 19, 218, 175, 183, 199, 251, 128, 55, 181, 131,
		68, 221, 235, 231, 227, 241, 223, 134, 149, 96, 159, 72, 174, 127, 10, 40, 237, 194, 153,
		168, 179, 34, 153, 76, 97, 68, 154, 7, 29, 128, 201, 230, 52, 206, 150, 47, 226, 37, 91,
		228, 138, 55, 61, 40, 60, 53, 248, 43, 19, 141, 104, 215, 114, 210, 227, 145, 108, 210, 98,
		173, 176, 232, 128, 170, 182, 158, 132, 52, 121, 15, 115, 61, 60, 177, 209, 108, 172, 184,
		222, 72, 147, 163, 239, 220, 248, 102, 127, 9, 8, 130, 110, 102, 174, 253, 30, 128, 86,
		141, 206, 190, 84, 227, 36, 185, 199, 16, 48, 209, 4, 232, 24, 146, 121, 172, 208, 71, 18,
		101, 21, 187, 21, 30, 86, 76, 34, 249, 191, 22, 128, 244, 216, 198, 82, 26, 113, 156, 185,
		120, 53, 29, 122, 253, 83, 62, 189, 63, 77, 67, 48, 141, 149, 87, 85, 0, 130, 97, 74, 16,
		52, 166, 36, 128, 178, 154, 208, 195, 196, 187, 119, 7, 207, 233, 107, 211, 140, 125, 150,
		99, 122, 191, 182, 234, 217, 222, 196, 161, 60, 218, 12, 243, 255, 191, 170, 106, 128, 107,
		167, 210, 9, 96, 187, 175, 55, 252, 198, 101, 202, 63, 133, 54, 97, 10, 227, 96, 138, 63,
		252, 233, 105, 147, 153, 102, 28, 156, 111, 87, 50, 128, 29, 2, 98, 143, 17, 126, 172, 22,
		89, 101, 209, 108, 232, 37, 226, 247, 65, 171, 186, 186, 181, 155, 205, 245, 20, 227, 147,
		198, 161, 231, 223, 182, 128, 59, 232, 180, 150, 106, 31, 118, 175, 124, 90, 245, 93, 151,
		121, 149, 151, 176, 155, 25, 252, 245, 79, 239, 39, 163, 136, 23, 83, 235, 66, 235, 229,
		128, 243, 45, 83, 124, 234, 189, 209, 244, 103, 222, 197, 161, 70, 86, 65, 25, 151, 106,
		22, 226, 196, 153, 105, 133, 121, 94, 135, 207, 6, 98, 94, 69, 128, 189, 79, 136, 242, 237,
		46, 25, 168, 245, 236, 197, 213, 57, 8, 69, 187, 97, 104, 237, 199, 197, 188, 28, 110, 201,
		212, 171, 179, 244, 244, 188, 190, 128, 13, 162, 241, 109, 223, 23, 128, 189, 34, 237, 63,
		192, 38, 174, 86, 98, 73, 217, 101, 60, 111, 239, 117, 82, 147, 137, 181, 213, 198, 2, 220,
		154, 189, 4, 158, 170, 57, 78, 234, 86, 48, 224, 124, 72, 174, 12, 149, 88, 206, 247, 57,
		159, 128, 163, 151, 95, 162, 170, 7, 71, 88, 74, 242, 67, 87, 153, 149, 39, 224, 9, 82,
		212, 151, 155, 190, 236, 178, 18, 47, 193, 26, 5, 9, 209, 99, 116, 95, 4, 171, 245, 203,
		52, 214, 36, 67, 120, 205, 219, 241, 142, 132, 157, 150, 44, 0, 0, 0, 0, 7, 35, 13, 102,
		21, 13, 0, 80, 95, 14, 123, 144, 18, 9, 107, 65, 196, 235, 58, 175, 148, 127, 110, 164, 41,
		8, 0, 0, 76, 95, 6, 132, 160, 34, 163, 77, 216, 191, 162, 186, 175, 68, 241, 114, 183, 16,
		4, 1, 128, 100, 201, 157, 182, 176, 105, 252, 40, 103, 127, 82, 111, 122, 112, 169, 25, 63,
		32, 11, 253, 245, 67, 229, 174, 77, 93, 225, 119, 140, 249, 60, 135, 128, 228, 219, 15,
		239, 88, 194, 6, 66, 131, 16, 143, 100, 141, 16, 212, 203, 249, 8, 240, 79, 180, 166, 139,
		91, 137, 167, 163, 186, 159, 119, 88, 183, 128, 15, 39, 15, 31, 154, 127, 14, 63, 145, 235,
		200, 59, 200, 174, 159, 37, 101, 17, 68, 180, 254, 57, 12, 229, 245, 154, 65, 220, 17, 81,
		179, 169, 128, 211, 190, 255, 68, 115, 252, 199, 36, 152, 174, 57, 152, 179, 198, 211, 233,
		251, 30, 13, 62, 250, 241, 117, 125, 122, 202, 170, 10, 251, 184, 183, 185, 76, 95, 2, 26,
		171, 3, 42, 170, 110, 148, 108, 165, 10, 211, 154, 182, 102, 3, 4, 1, 112, 95, 9, 204, 233,
		200, 136, 70, 155, 177, 160, 220, 234, 161, 41, 103, 46, 248, 40, 145, 146, 28, 119, 101,
		115, 116, 101, 110, 100, 141, 8, 159, 9, 157, 136, 14, 198, 129, 121, 156, 12, 243, 14,
		136, 134, 55, 29, 169, 255, 255, 128, 8, 5, 149, 84, 15, 242, 192, 165, 247, 232, 121, 27,
		83, 12, 59, 85, 140, 241, 198, 74, 247, 38, 86, 53, 60, 49, 207, 118, 199, 164, 74, 192,
		128, 48, 136, 171, 183, 136, 214, 222, 72, 245, 37, 220, 80, 141, 11, 238, 204, 110, 35,
		71, 31, 115, 163, 101, 208, 226, 166, 192, 214, 190, 186, 103, 231, 128, 243, 234, 51, 128,
		74, 237, 212, 183, 67, 161, 197, 47, 135, 128, 77, 179, 207, 2, 80, 223, 136, 147, 36, 242,
		57, 182, 29, 130, 77, 63, 90, 50, 128, 7, 214, 13, 48, 91, 61, 48, 207, 44, 87, 69, 133,
		28, 118, 220, 191, 43, 188, 157, 200, 5, 215, 130, 157, 54, 35, 69, 195, 53, 252, 83, 23,
		128, 95, 206, 68, 206, 224, 27, 220, 138, 46, 139, 17, 185, 81, 249, 123, 71, 220, 204, 54,
		196, 254, 97, 135, 210, 27, 139, 115, 193, 92, 31, 131, 190, 128, 107, 205, 124, 8, 248,
		87, 123, 58, 58, 19, 156, 32, 227, 226, 62, 173, 151, 206, 233, 187, 220, 217, 24, 216,
		127, 33, 34, 152, 35, 216, 114, 243, 128, 164, 111, 34, 230, 218, 85, 165, 111, 61, 18,
		178, 213, 42, 234, 148, 162, 193, 210, 82, 7, 81, 29, 219, 134, 156, 54, 58, 182, 237, 183,
		90, 68, 128, 182, 145, 117, 249, 108, 72, 95, 72, 225, 179, 80, 252, 184, 74, 58, 31, 17,
		242, 172, 58, 211, 252, 110, 140, 114, 18, 86, 13, 12, 194, 20, 109, 128, 219, 173, 125,
		48, 63, 121, 56, 23, 133, 163, 1, 163, 241, 150, 67, 253, 84, 41, 18, 185, 243, 36, 63, 40,
		144, 58, 127, 44, 61, 185, 8, 150, 128, 210, 152, 144, 102, 19, 214, 199, 237, 113, 218,
		144, 2, 21, 238, 191, 161, 201, 4, 85, 237, 101, 193, 90, 108, 210, 231, 152, 25, 33, 31,
		57, 123, 128, 158, 148, 144, 66, 31, 148, 244, 90, 158, 191, 217, 188, 43, 81, 18, 32, 29,
		161, 29, 246, 250, 117, 71, 239, 145, 146, 2, 185, 162, 82, 40, 234, 128, 60, 128, 247, 61,
		33, 117, 146, 141, 156, 25, 22, 213, 44, 77, 236, 243, 66, 105, 207, 42, 88, 24, 7, 4, 248,
		39, 217, 20, 27, 48, 57, 138, 128, 138, 83, 127, 90, 159, 23, 159, 232, 74, 236, 220, 156,
		207, 75, 36, 145, 213, 198, 237, 251, 185, 243, 60, 85, 215, 174, 95, 145, 23, 190, 212, 6,
		128, 161, 101, 84, 232, 199, 40, 128, 56, 98, 33, 228, 35, 245, 22, 225, 59, 148, 255, 70,
		12, 76, 8, 212, 96, 63, 230, 93, 55, 8, 59, 34, 133, 128, 240, 168, 155, 184, 133, 70, 10,
		170, 160, 216, 89, 108, 217, 191, 250, 117, 189, 182, 160, 254, 50, 7, 220, 189, 251, 143,
		231, 132, 28, 236, 1, 187, 128, 43, 195, 81, 22, 247, 100, 18, 37, 9, 240, 251, 206, 95,
		87, 143, 166, 170, 251, 140, 126, 149, 24, 148, 25, 19, 172, 71, 188, 37, 110, 155, 43,
	];
	let decoded: Vec<Vec<u8>> = Decode::decode(&mut &pb_proof[..]).unwrap();
	let mut count_ko = 0;
	let target = decoded[1].clone();
	let target_hash = RefHasher::hash(&target);
	for (i, enc_node) in decoded.iter().enumerate() {
		let hash = RefHasher::hash(enc_node.as_slice());
		let decoded = NodeCodec::<RefHasher>::decode(&enc_node);
//		println!("{:?}", (&hash, &decoded));
		if let Ok(node) = decoded {
			match node {
				Node::NibbledBranch(_, children, value) | Node::Branch(children, value) => {
					for (j, c) in children.iter().enumerate() {
						match c {
							Some(NodeHandle::Hash(h)) => {
								if *h == &target_hash[..] {
									println!("found hash as children {i} {j}");
								}
								if &h[..] == &target[..] {
									println!("found value as children {i} {j}");
								}
							},
							Some(NodeHandle::Inline(h)) => {
								if *h == &target_hash[..] {
									println!("found hash as inline children {i} {j}");
								}
								if &h[..] == &target[..] {
									println!("found value as inline children {i} {j}");
								}
							},
							None => (),
						}
					}
					match value {
						Some(Value::Node(value)) => {
							if value == &target_hash[..] {
								println!("found hash as branch detached value {i}");
							}
							if value == &target[..] {
								println!("found value as branch detached value {i}");
							}
						},
						Some(Value::Inline(value)) => {
							if value == &target_hash[..] {
								println!("found hash as leaf value {i}");
							}
							if value == &target[..] {
								println!("found value as leaf value {i}");
							}
						},
						None => (),
					}
				},
				Node::Leaf(_, value) => match value {
					Value::Node(value) => {
						if value == &target_hash[..] {
							println!("found hash as leaf detached value {i}");
						}
						if value == &target[..] {
							println!("found value as leaf detached value {i}");
						}
					},
					Value::Inline(value) => {
						if value == &target_hash[..] {
							println!("found hash as leaf value {i}");
						}
						if value == &target[..] {
							println!("found value as leaf value {i}");
						}
					},
				},
				_ => (),
			}
		} else {
			count_ko += 1;
		}
	}
	println!("fail {:?} / {:?}", count_ko, decoded.len());

	panic!("disp {:?}", NodeCodec::<RefHasher>::hashed_null_node());
}
