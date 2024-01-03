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

//! Nibble-orientated view onto byte-slice, allowing nibble-precision offsets.

use super::{BackingByteVec, NibbleSlice, NibbleSliceIterator, NibbleVec};
#[cfg(feature = "std")]
use crate::rstd::fmt;
use crate::{
	node::NodeKey,
	node_codec::Partial,
	rstd::{cmp::*, marker::PhantomData},
	NibbleOps,
};
use hash_db::Prefix;

impl<'a, const N: usize> Iterator for NibbleSliceIterator<'a, N> {
	type Item = u8;
	fn next(&mut self) -> Option<u8> {
		self.i += 1;
		match self.i <= self.p.len() {
			true => Some(self.p.at(self.i - 1)),
			false => None,
		}
	}
}

impl<'a, const N: usize> NibbleSlice<'a, N> {
	/// Create a new nibble slice with the given byte-slice.
	pub fn new(data: &'a [u8]) -> Self {
		NibbleSlice::new_slice(data, 0)
	}

	/// Create a new nibble slice with the given byte-slice with a nibble offset.
	pub fn new_offset(data: &'a [u8], offset: usize) -> Self {
		Self::new_slice(data, offset)
	}

	fn new_slice(data: &'a [u8], offset: usize) -> Self {
		NibbleSlice { data, offset }
	}

	/// Get an iterator for the series of nibbles.
	pub fn iter(&'a self) -> NibbleSliceIterator<'a, N> {
		NibbleSliceIterator { p: self, i: 0 }
	}

	/// Get nibble slice from a `NodeKey`.
	pub fn from_stored(i: &NodeKey) -> NibbleSlice<N> {
		NibbleSlice::new_offset(&i.1[..], i.0)
	}

	/// Helper function to create a owned `NodeKey` from this `NibbleSlice`.
	pub fn to_stored(&self) -> NodeKey {
		let n = NibbleOps::<N>::nibble_per_byte();
		let split = self.offset / n;
		let offset = self.offset % n;
		(offset, self.data[split..].into())
	}

	/// Helper function to create a owned `NodeKey` from this `NibbleSlice`,
	/// and for a given number of nibble.
	/// Warning this method can be slow (number of nibble does not align the
	/// original padding).
	pub fn to_stored_range(&self, nb: usize) -> NodeKey {
		let n = NibbleOps::<N>::nibble_per_byte();
		if nb >= self.len() {
			return self.to_stored()
		}
		if (self.offset + nb) % n == 0 {
			// aligned
			let start = self.offset / n;
			let end = (self.offset + nb) / n;
			(self.offset % n, BackingByteVec::from_slice(&self.data[start..end]))
		} else {
			// unaligned
			let start = self.offset / n;
			let end = (self.offset + nb) / n;
			let ea = BackingByteVec::from_slice(&self.data[start..=end]);
			let ea_offset = self.offset % n;
			let n_offset = NibbleOps::<N>::number_padding(nb);
			let mut result = (ea_offset, ea);
			NibbleOps::<N>::shift_key(&mut result, n_offset);
			result.1.pop();
			result
		}
	}

	/// Return true if the slice contains no nibbles.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Get the length (in nibbles, naturally) of this slice.
	#[inline]
	pub fn len(&self) -> usize {
		(self.data.len() * NibbleOps::<N>::nibble_per_byte()) - self.offset
	}

	/// Get the nibble at position `i`.
	#[inline(always)]
	pub fn at(&self, i: usize) -> u8 {
		NibbleOps::<N>::at(&self, i)
	}

	/// Return object which represents a view on to this slice (further) offset by `i` nibbles.
	pub fn mid(&self, i: usize) -> NibbleSlice<'a, N> {
		NibbleSlice { data: self.data, offset: self.offset + i }
	}

	/// Advance the view on the slice by `i` nibbles.
	pub fn advance(&mut self, i: usize) {
		debug_assert!(self.len() >= i);
		self.offset += i;
	}

	/// Move back to a previously valid fix offset position.
	pub fn back(&self, i: usize) -> NibbleSlice<'a, N> {
		NibbleSlice { data: self.data, offset: i }
	}

	/// Do we start with the same nibbles as the whole of `them`?
	pub fn starts_with(&self, them: &Self) -> bool {
		self.common_prefix(them) == them.len()
	}

	/// How many of the same nibbles at the beginning do we match with `them`?
	pub fn common_prefix(&self, them: &Self) -> usize {
		let n = NibbleOps::<N>::nibble_per_byte();
		let self_align = self.offset % n;
		let them_align = them.offset % n;
		if self_align == them_align {
			let mut self_start = self.offset / n;
			let mut them_start = them.offset / n;
			let mut first = 0;
			if self_align != 0 {
				let self_first =
					NibbleOps::<N>::pad_right((n - self_align) as u8, self.data[self_start]);
				let them_first =
					NibbleOps::<N>::pad_right((n - them_align) as u8, them.data[them_start]);
				if self_first != them_first {
					return if N == 16 {
						0
					} else {
						let common = NibbleOps::<N>::left_common(self_first, them_first);
						common - self_align
					}
				}
				self_start += 1;
				them_start += 1;
				first = n - self_align;
			}
			NibbleOps::<N>::biggest_depth(&self.data[self_start..], &them.data[them_start..]) +
				first
		} else {
			let s = min(self.len(), them.len());
			let mut i = 0usize;
			while i < s {
				if self.at(i) != them.at(i) {
					break
				}
				i += 1;
			}
			i
		}
	}

	/// Return `Partial` representation of this slice:
	/// first encoded byte and following slice.
	pub fn right(&'a self) -> Partial {
		let n = NibbleOps::<N>::nibble_per_byte();
		let split = self.offset / n;
		let nb = (self.len() % n) as u8;
		if nb > 0 {
			((nb, NibbleOps::<N>::pad_right(nb, self.data[split])), &self.data[split + 1..])
		} else {
			((0, 0), &self.data[split..])
		}
	}

	/// Return an iterator over `Partial` bytes representation.
	pub fn right_iter(&'a self) -> impl Iterator<Item = u8> + 'a {
		let (mut first, sl) = self.right();
		let mut ix = 0;
		crate::rstd::iter::from_fn(move || {
			if first.0 > 0 {
				first.0 = 0;
				Some(NibbleOps::<N>::pad_right(first.0, first.1))
			} else if ix < sl.len() {
				ix += 1;
				Some(sl[ix - 1])
			} else {
				None
			}
		})
	}

	/// Return `Partial` bytes iterator over a range of byte..
	/// Warning can be slow when unaligned (similar to `to_stored_range`).
	pub fn right_range_iter(&'a self, nb_nibbles: usize) -> impl Iterator<Item = u8> + 'a {
		let n = NibbleOps::<N>::nibble_per_byte();
		let mut res_mask = (n - (nb_nibbles % n)) % n;
		let offset_mask = self.offset % n;
		let ix_init = self.offset / n;
		let mut ix = ix_init;
		let ix_lim = if (self.offset + nb_nibbles) % n > 0 {
			((self.offset + nb_nibbles) / n) 
		} else {
			((self.offset + nb_nibbles) / n)
		};
		
		let aligned = res_mask == offset_mask;
		let (sub, s1, s2) = if aligned {
			(false, 0, 0)
		} else if res_mask > offset_mask {
			let (s1, s2) = NibbleOps::<N>::split_shifts(res_mask - offset_mask);
			(false, s2, s1)
		} else {
			let (s1, s2) = NibbleOps::<N>::split_shifts(offset_mask - res_mask);
			(true, s2, s1)
		};
		crate::rstd::iter::from_fn(move || {
			if aligned {
				if res_mask > 0 {
					let v = NibbleOps::<N>::pad_right(res_mask as u8, self.data[ix]);
					res_mask = 0;
					ix += 1;
					Some(v)
				} else if ix < ix_lim {
					ix += 1;
					Some(self.data[ix - 1])
				} else {
					None
				}
			} else {
				// unaligned
				if (sub && ix < ix_lim) || (!sub && ix <= ix_lim) {
					let mut b = if sub {
						let mut b = self.data[ix] << s1;
						if ix + 1 <= ix_lim {
							b |= self.data[ix + 1] >> s2;
						}
						b
					} else {
						
						let mut b = self.data[ix] >> s1;
						if ix > ix_init {
							b |= self.data[ix - 1] << s2;
						}
						b
					};
					if res_mask > 0 {
						b = NibbleOps::<N>::pad_right((n - res_mask) as u8, b);
						res_mask = 0;
					}
					ix += 1;
					Some(b)
				} else {
					None
				}
			}
		})
	}

	/// Return left portion of `NibbleSlice`, if the slice
	/// originates from a full key it will be the `Prefix of
	/// the node`.
	pub fn left(&'a self) -> Prefix {
		let n = NibbleOps::<N>::nibble_per_byte();
		let split = self.offset / n;
		let ix = (self.offset % n) as u8;
		if ix == 0 {
			Prefix { slice: &self.data[..split], last: 0, align: 0 }
		} else {
			Prefix {
				slice: &self.data[..split],
				last: NibbleOps::<N>::pad_left(ix, self.data[split]),
				align: ix,
			}
		}
	}

	/// Get [`Prefix`] representation of the inner data.
	///
	/// This means the entire inner data will be returned as [`Prefix`], ignoring any `offset`.
	pub fn original_data_as_prefix(&self) -> Prefix {
		Prefix { slice: &self.data, last: 0, align: 0 }
	}

	/// Owned version of a `Prefix` from a `left` method call.
	pub fn left_owned(&'a self) -> (BackingByteVec, u8, u8) {
		let prefix = self.left();
		(prefix.slice.into(), prefix.last, prefix.align)
	}

	/// Same as [`Self::starts_with`] but using [`NibbleVec`].
	pub fn starts_with_vec(&self, other: &NibbleVec<N>) -> bool {
		if self.len() < other.len() {
			return false
		}

		match other.as_nibbleslice() {
			Some(other) => self.starts_with(&other),
			None => {
				for i in 0..other.len() {
					if self.at(i) != other.at(i) {
						return false
					}
				}
				true
			},
		}
	}
}

impl<'a, const N: usize> From<NibbleSlice<'a, N>> for NodeKey {
	fn from(slice: NibbleSlice<'a, N>) -> NodeKey {
		(slice.offset, slice.data.into())
	}
}

impl<'a, const N: usize> PartialEq for NibbleSlice<'a, N> {
	fn eq(&self, them: &Self) -> bool {
		self.len() == them.len() && self.starts_with(them)
	}
}

impl<'a, const N: usize> PartialEq<NibbleVec<N>> for NibbleSlice<'a, N> {
	fn eq(&self, other: &NibbleVec<N>) -> bool {
		if self.len() != other.len() {
			return false
		}

		match other.as_nibbleslice() {
			Some(other) => *self == other,
			None => self.iter().enumerate().all(|(index, l)| l == other.at(index)),
		}
	}
}

impl<'a, const N: usize> Eq for NibbleSlice<'a, N> {}

impl<'a, const N: usize> PartialOrd for NibbleSlice<'a, N> {
	fn partial_cmp(&self, them: &Self) -> Option<Ordering> {
		Some(self.cmp(them))
	}
}

impl<'a, const N: usize> Ord for NibbleSlice<'a, N> {
	fn cmp(&self, them: &Self) -> Ordering {
		let s = min(self.len(), them.len());
		let mut i = 0usize;
		while i < s {
			match self.at(i).partial_cmp(&them.at(i)).unwrap() {
				Ordering::Less => return Ordering::Less,
				Ordering::Greater => return Ordering::Greater,
				_ => i += 1,
			}
		}
		self.len().cmp(&them.len())
	}
}

#[cfg(feature = "std")]
impl<'a, const N: usize> fmt::Debug for NibbleSlice<'a, N> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		for i in 0..self.len() {
			match i {
				0 => write!(f, "{:01x}", self.at(i))?,
				_ => write!(f, "'{:01x}", self.at(i))?,
			}
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use crate::nibble::{BackingByteVec, NibbleOps, NibbleSlice};
	static D: &'static [u8; 3] = &[0x01u8, 0x23, 0x45];

	#[test]
	fn basics() {
		basics_inner::<16>();
	}

	fn basics_inner<const N: usize>() {
		let n = NibbleSlice::<N>::new(D);
		assert_eq!(n.len(), 6);
		assert!(!n.is_empty());

		let n = NibbleSlice::<N>::new_offset(D, 6);
		assert!(n.is_empty());

		let n = NibbleSlice::<N>::new_offset(D, 3);
		assert_eq!(n.len(), 3);
		for i in 0..3 {
			assert_eq!(n.at(i), i as u8 + 3);
		}
	}

	#[test]
	fn iterator() {
		iterator_inner::<16>();
	}

	fn iterator_inner<const N: usize>() {
		let n = NibbleSlice::<N>::new(D);
		let mut nibbles: Vec<u8> = vec![];
		nibbles.extend(n.iter());
		assert_eq!(nibbles, (0u8..6).collect::<Vec<_>>())
	}

	#[test]
	fn mid() {
		mid_inner::<16>();
	}

	fn mid_inner<const N: usize>() {
		let n = NibbleSlice::<N>::new(D);
		let m = n.mid(2);
		for i in 0..4 {
			assert_eq!(m.at(i), i as u8 + 2);
		}
		let m = n.mid(3);
		for i in 0..3 {
			assert_eq!(m.at(i), i as u8 + 3);
		}
	}

	#[test]
	fn encoded_pre() {
		let n = NibbleSlice::<16>::new(D);
		assert_eq!(n.to_stored(), (0, BackingByteVec::from_slice(&[0x01, 0x23, 0x45])));
		assert_eq!(n.mid(1).to_stored(), (1, BackingByteVec::from_slice(&[0x01, 0x23, 0x45])));
		assert_eq!(n.mid(2).to_stored(), (0, BackingByteVec::from_slice(&[0x23, 0x45])));
		assert_eq!(n.mid(3).to_stored(), (1, BackingByteVec::from_slice(&[0x23, 0x45])));
	}

	#[test]
	fn from_encoded_pre() {
		let n = NibbleSlice::<16>::new(D);
		let stored: BackingByteVec = [0x01, 0x23, 0x45][..].into();
		assert_eq!(n, NibbleSlice::from_stored(&(0, stored.clone())));
		assert_eq!(n.mid(1), NibbleSlice::from_stored(&(1, stored)));
	}

	#[test]
	fn range_iter() {
		let n = NibbleSlice::<16>::new(D);
		let n2 = NibbleSlice::<4>::new(D);
		for i in [
			vec![],
			vec![0x00],
			vec![0x01],
			vec![0x00, 0x12],
			vec![0x01, 0x23],
			vec![0x00, 0x12, 0x34],
			vec![0x01, 0x23, 0x45],
		]
		.iter()
		.enumerate()
		{
			range_iter_test::<16>(n, i.0, None, &i.1[..]);
			range_iter_test::<4>(n2, i.0 * 2, None, &i.1[..]);
		}
		for i in [
			vec![],
			vec![0x01],
			vec![0x12],
			vec![0x01, 0x23],
			vec![0x12, 0x34],
			vec![0x01, 0x23, 0x45],
		]
		.iter()
		.enumerate()
		{
			range_iter_test::<16>(n, i.0, Some(1), &i.1[..]);
			range_iter_test::<4>(n2, i.0 * 2, Some(2), &i.1[..]);
		}
		for i in [vec![], vec![0x02], vec![0x23], vec![0x02, 0x34], vec![0x23, 0x45]]
			.iter()
			.enumerate()
		{
			range_iter_test::<16>(n, i.0, Some(2), &i.1[..]);
			range_iter_test::<4>(n2, i.0 * 2, Some(4), &i.1[..]);
		}
		for i in [vec![], vec![0x03], vec![0x34], vec![0x03, 0x45]].iter().enumerate() {
			range_iter_test::<16>(n, i.0, Some(3), &i.1[..]);
			range_iter_test::<4>(n2, i.0 * 2, Some(6), &i.1[..]);
		}
	}

	fn range_iter_test<const N: usize>(
		n: NibbleSlice<N>,
		nb: usize,
		mid: Option<usize>,
		res: &[u8],
	) {
		let n = if let Some(i) = mid { n.mid(i) } else { n };
		assert_eq!(&n.right_range_iter(nb).collect::<Vec<_>>()[..], res);
	}

	#[test]
	fn shared() {
		shared_inner::<16>();
	}
	fn shared_inner<const N: usize>() {
		let n = NibbleSlice::<N>::new(D);

		let other = &[0x01u8, 0x23, 0x01, 0x23, 0x45, 0x67];
		let m = NibbleSlice::new(other);

		assert_eq!(n.common_prefix(&m), 4);
		assert_eq!(m.common_prefix(&n), 4);
		assert_eq!(n.mid(1).common_prefix(&m.mid(1)), 3);
		assert_eq!(n.mid(1).common_prefix(&m.mid(2)), 0);
		assert_eq!(n.common_prefix(&m.mid(4)), 6);
		assert!(!n.starts_with(&m.mid(4)));
		assert!(m.mid(4).starts_with(&n));
	}

	#[test]
	fn compare() {
		compare_inner::<16>();
	}
	fn compare_inner<const N: usize>() {
		let other = &[0x01u8, 0x23, 0x01, 0x23, 0x45];
		let n = NibbleSlice::<N>::new(D);
		let m = NibbleSlice::new(other);

		assert!(n != m);
		assert!(n > m);
		assert!(m < n);

		assert!(n == m.mid(4));
		assert!(n >= m.mid(4));
		assert!(n <= m.mid(4));
	}
}
