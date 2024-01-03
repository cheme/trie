// Copyright 2019 Parity Technologies
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

//! Nibble oriented methods.

use crate::{
	node::{NodeHandle, NodeHandlePlan, NodeKey},
	rstd::{cmp, vec, vec::Vec},
};
use hash_db::MaybeDebug;

pub use self::leftnibbleslice::LeftNibbleSlice;

mod leftnibbleslice;
mod nibbleslice;
mod nibblevec;

// Work-around absence of constant function for math pow.
const TWO_EXP: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

pub const fn bitmap_size<const N: usize>() -> usize {
	if N % 8 == 0 {
		N / 8
	} else {
		N / 8 + 1
	}
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord, Copy)]
/// Struct for methods depending on radix
pub struct NibbleOps<const N: usize>;
impl<const N: usize> NibbleOps<N> {
	/// Single nibble length in bit.
	pub const fn bit_per_nibble() -> usize {
		N.trailing_zeros() as usize
	}

	pub const fn nibble_per_byte() -> usize {
		8 / Self::bit_per_nibble()
	}

	pub const fn bitmap_size() -> usize {
		if N % 8 == 0 {
			N / 8
		} else {
			N / 8 + 1
		}
	}

	/// Padding bitmasks, internally use for working on padding byte.
	pub const fn padding_bitmask(ix: usize) -> u8 {
		let offset = Self::bit_per_nibble() * ix;
		255u8 >> offset
	}

	/// Pad left aligned representation for a given number of element.
	/// Mask a byte from a `ix` > 0 (ix being content).
	/// Result is a byte containing `ix` nibble of left aligned content and padded with 0.
	#[inline(always)]
	pub fn pad_left(ix: u8, b: u8) -> u8 {
		debug_assert!(ix > 0); // 0 just padd all TODO EMCH allow it?
		b & !Self::padding_bitmask(ix as usize)
		//b & !(255u8 >> (Self::bit_per_nibble * ix)) // TODO EMCH compare perf with that
	}

	/// Pad right aligned representation for a given number of element.
	/// Mask a byte from a ix > 0 (ix being content).
	/// Result is a byte containing `ix` nibble of right aligned content and padded with 0.
	#[inline(always)]
	pub fn pad_right(ix: u8, b: u8) -> u8 {
		// TODO EMCH change code to avoid this test (panic on 0 to see)
		// it means there is calls to pad_right where we do not use the number
		// of elements!
		if ix > 0 {
			b & Self::padding_bitmask(Self::nibble_per_byte() - ix as usize)
		//b & !(255u8 << (Self::bit_per_nibble() * (N - ix as usize)))
		//b & Self::PADDING_BITMASK[N - ix as usize].0
		} else {
			b
		}
	}

	/// Get u8 nibble value at a given index of a byte.
	#[inline(always)]
	pub fn at_left(ix: u8, b: u8) -> u8 {
		// 8 -> ix 0 -> offset 0
		// 4 -> ix 0 -> offset 4
		//   ->    1 -> offset 0
		// 2 -> ix 0 -> offset 6
		//   ->    1 -> offset 4
		//   ->    2 -> offset 2
		//   ->    3 -> offset 0
		// 1 -> ix 0 -> offset 7
		//   ->    1 -> offset 6
		//   ->    6 -> offset 1
		//   ->    7 -> offset 0

		let offset = 8 - (Self::bit_per_nibble() as u8 * (ix + 1)); // TODO this is **** bullshit do test
															// TODO EMCH compare perf without padding bitmask
		(b & Self::padding_bitmask(ix as usize)) >> offset
	}

	/// Get u8 nibble value at a given index in a left aligned array.
	#[inline(always)]
	pub fn left_nibble_at(v1: &[u8], mut ix: usize) -> u8 {
		let n = NibbleOps::<N>::nibble_per_byte();
		let pad = ix % n;
		ix = ix / n;
		Self::at_left(pad as u8, v1[ix])
	}

	/// Get u8 nibble value at a given index in a `NibbleSlice`.
	#[inline(always)]
	pub fn at(s: &NibbleSlice<N>, ix: usize) -> u8 {
		// same as left with offset
		Self::left_nibble_at(&s.data[..], s.offset + ix)
	}

	/// Push u8 nibble value at a given index into an existing byte.
	/// Note that existing value must be null (padded with 0).
	#[inline(always)]
	pub fn push_at_left(ix: u8, v: u8, into: u8) -> u8 {
		let offset = 8 - (Self::bit_per_nibble() as u8 * (ix + 1));
		//into | (v << (8 - (bit_per_nibble * ix)))
		debug_assert!((into & (v << offset)) == 0);
		into | (v << offset)
	}

	#[inline]
	/// Calculate the number of needed padding for an array of nibble length `i`.
	pub fn number_padding(i: usize) -> usize {
		let n = NibbleOps::<N>::nibble_per_byte();
		(n - (i % n)) % n
	}

	/// Count the biggest common depth between two left aligned packed nibble slice.
	pub fn biggest_depth(v1: &[u8], v2: &[u8]) -> usize {
		let n = NibbleOps::<N>::nibble_per_byte();
		let upper_bound = cmp::min(v1.len(), v2.len());
		for a in 0..upper_bound {
			if v1[a] != v2[a] {
				return a * n + Self::left_common(v1[a], v2[a])
			}
		}
		upper_bound * n
	}

	/// Calculate the number of common nibble between two left aligned bytes.
	#[inline(always)]
	pub fn left_common(a: u8, b: u8) -> usize {
		((a ^ b).leading_zeros() as usize) / NibbleOps::<N>::bit_per_nibble()
		/*		let mut i = 0;
		while i < N {
			//if (a >> Self::PADDING_BITMASK[i].1)
			//	!= (b >> Self::PADDING_BITMASK[i].1) {
			let offset = i * Self::bit_per_nibble;
			if (a >> offset) != (b >> offset) {
				break;
			}
			i += 1;
		}
		return i;*/
	}

	/// The nibble shifts needed to align.
	/// We use two value, one is a left shift and
	/// the other is a right shift.
	/// Calculate the number of common nibble between two left aligned bytes.
	#[inline(always)]
	pub fn split_shifts(pad: usize) -> (usize, usize) {
		let s2 = Self::bit_per_nibble() * pad;
		let s1 = 8 - s2;
		(s1, s2)
	}

	/// Shifts right aligned key to add a given left offset.
	/// Resulting in possibly padding at both left and right
	/// (used when combining two keys).
	pub fn shift_key(key: &mut NodeKey, ofset: usize) -> bool {
		let old_offset = key.0;
		key.0 = ofset;
		if old_offset > ofset {
			// shift left
			let shift = old_offset - ofset;
			let (s1, s2) = Self::split_shifts(shift);
			let kl = key.1.len();
			(0..kl - 1).for_each(|i| key.1[i] = key.1[i] << s2 | key.1[i + 1] >> s1);
			key.1[kl - 1] = key.1[kl - 1] << s2;
			true
		} else if old_offset < ofset {
			// shift right
			let shift = ofset - old_offset;
			let (s1, s2) = Self::split_shifts(shift);
			key.1.push(0);
			(1..key.1.len())
				.rev()
				.for_each(|i| key.1[i] = key.1[i - 1] << s1 | key.1[i] >> s2);
			key.1[0] = key.1[0] >> s2;
			true
		} else {
			false
		}
	}
}

/// Backing storage for `NibbleVec`s.
pub(crate) type BackingByteVec = smallvec::SmallVec<[u8; 40]>;

/// Owning, nibble-oriented byte vector. Counterpart to `NibbleSlice`.
/// Nibbles are always left aligned, so making a `NibbleVec` from
/// a `NibbleSlice` can get costy.
#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, PartialEq, Eq)]
pub struct NibbleVec<const N: usize> {
	inner: BackingByteVec,
	len: usize,
}

/// Nibble-orientated view onto byte-slice, allowing nibble-precision offsets.
///
/// This is an immutable struct. No operations actually change it.
///
/// # Example
/// ```snippet
/// use patricia_trie::nibbleslice::NibbleSlice;
/// fn main() {
///   let d1 = &[0x01u8, 0x23, 0x45];
///   let d2 = &[0x34u8, 0x50, 0x12];
///   let d3 = &[0x00u8, 0x12];
///   let n1 = NibbleSlice::new(d1);			// 0,1,2,3,4,5
///   let n2 = NibbleSlice::new(d2);			// 3,4,5,0,1,2
///   let n3 = NibbleSlice::new_offset(d3, 1);	// 0,1,2
///   assert!(n1 > n3);							// 0,1,2,... > 0,1,2
///   assert!(n1 < n2);							// 0,... < 3,...
///   assert!(n2.mid(3) == n3);					// 0,1,2 == 0,1,2
///   assert!(n1.starts_with(&n3));
///   assert_eq!(n1.common_prefix(&n3), 3);
///   assert_eq!(n2.mid(3).common_prefix(&n1), 3);
/// }
/// ```
#[derive(Copy, Clone)]
pub struct NibbleSlice<'a, const N: usize> {
	data: &'a [u8],
	offset: usize,
}

/// Iterator type for a nibble slice.
pub struct NibbleSliceIterator<'a, const N: usize> {
	p: &'a NibbleSlice<'a, N>,
	i: usize,
}

/// Technical trait only to access child slice from an encoded
/// representation of a branch.
pub trait ChildIndex<V>:
	AsRef<[Option<V>]> + AsMut<[Option<V>]> + Default + Eq + PartialEq + crate::MaybeDebug + Clone
{
	/// Constant length for the number of children.
	/// TODO EMCH see if can delete (by using consto or associated value in layout)
	const NIBBLE_LENGTH: usize;

	#[inline]
	fn from_iter(nodes: impl Iterator<Item = Option<V>>) -> Self {
		let mut index = Self::default();
		for (i, node) in nodes.enumerate() {
			index.as_mut()[i] = node;
		}
		index
	}

	#[inline]
	fn at(&self, ix: usize) -> Option<&V> {
		self.as_ref()[ix].as_ref()
	}

	#[inline]
	fn take(&mut self, ix: usize) -> Option<V> {
		self.as_mut()[ix].take()
	}

	#[inline]
	fn at_mut(&mut self, ix: usize) -> &mut Option<V> {
		&mut self.as_mut()[ix]
	}

	#[inline]
	fn iter_mut(&mut self) -> crate::rstd::slice::IterMut<Option<V>> {
		self.as_mut().iter_mut()
	}
}

pub trait ChildSliceIndex: ChildIndex<NodeHandlePlan> {
	#[inline]
	fn slice_at<'a>(&self, ix: usize, data: &'a [u8]) -> Option<NodeHandle<'a>> {
		self.at(ix).map(|plan| plan.build(data))
	}

	/// Iterator over the children slice.
	fn iter<'a>(&'a self, data: &'a [u8]) -> IterChildSliceIndex<'a, Self> {
		IterChildSliceIndex(self, 0, data)
	}
}

impl<I: ChildIndex<NodeHandlePlan>> ChildSliceIndex for I {}

/// Iterator over `ChildSliceIndex` trait.
pub struct IterChildSliceIndex<'a, CS>(&'a CS, usize, &'a [u8]);

impl<'a, CS: ChildSliceIndex> Iterator for IterChildSliceIndex<'a, CS> {
	type Item = Option<NodeHandle<'a>>;
	fn next(&mut self) -> Option<Self::Item> {
		if self.1 == CS::NIBBLE_LENGTH {
			return None
		}
		self.1 += 1;
		Some(self.0.slice_at(self.1 - 1, self.2))
	}
}

macro_rules! child_slice_index {
	($me: ident, $size: expr) => {
		#[cfg_attr(feature = "std", derive(Debug))]
		#[derive(Eq, PartialEq, Clone)]
		/// Child slice indexes for radix $size.
		pub struct $me<V>([Option<V>; $size]);

		impl<V> AsRef<[Option<V>]> for $me<V> {
			fn as_ref(&self) -> &[Option<V>] {
				&self.0[..]
			}
		}

		impl<V> AsMut<[Option<V>]> for $me<V> {
			fn as_mut(&mut self) -> &mut [Option<V>] {
				&mut self.0[..]
			}
		}

		impl<V> ChildIndex<V> for $me<V>
		where
			V: MaybeDebug + Eq + PartialEq + Clone,
		{
			const NIBBLE_LENGTH: usize = $size;
		}
	};
}

child_slice_index!(ChildIndex16, 16);
child_slice_index!(ChildIndex4, 4);
child_slice_index!(ChildIndex2, 2);

macro_rules! exponential_out {
	(@3, [$($inpp:expr),*]) => { exponential_out!(@2, [$($inpp,)* $($inpp),*]) };
	(@2, [$($inpp:expr),*]) => { exponential_out!(@1, [$($inpp,)* $($inpp),*]) };
	(@1, [$($inpp:expr),*]) => { [$($inpp,)* $($inpp),*] };
}

impl<V> Default for ChildIndex2<V> {
	fn default() -> Self {
		ChildIndex2(exponential_out!(@1, [None]))
	}
}

impl<V> Default for ChildIndex4<V> {
	fn default() -> Self {
		ChildIndex4(exponential_out!(@2, [None]))
	}
}

impl<V> Default for ChildIndex16<V> {
	fn default() -> Self {
		ChildIndex16(exponential_out!(@3, [None, None]))
	}
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Eq, PartialEq, Clone)]
/// Child slice indexes for radix 256.
///
/// TODO EMCH no default impl for array of len 257,
/// but could use bench to see if worth implementing
/// (probably sparse vec implementation is better:
/// need to remove asref and asmut bound).
pub struct ChildIndex256<V>(Vec<Option<V>>);

impl<V: Clone> Default for ChildIndex256<V> {
	fn default() -> Self {
		ChildIndex256(vec![None; 256])
	}
}

impl<V> AsRef<[Option<V>]> for ChildIndex256<V> {
	fn as_ref(&self) -> &[Option<V>] {
		&self.0[..]
	}
}

impl<V> AsMut<[Option<V>]> for ChildIndex256<V> {
	fn as_mut(&mut self) -> &mut [Option<V>] {
		&mut self.0[..]
	}
}

impl<V> ChildIndex<V> for ChildIndex256<V>
where
	V: MaybeDebug + Eq + PartialEq + Clone,
{
	const NIBBLE_LENGTH: usize = 256;
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn nibble_vec_size() {
		assert_eq!(std::mem::size_of::<NibbleVec<16>>(), 56);
	}
}
