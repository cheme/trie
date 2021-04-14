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

use crate::rstd::cmp::{self, Ordering};

use crate::nibble::{nibble_ops::{self, NIBBLE_PER_BYTE}, NibbleSlice};

/// A representation of a nibble slice which is left-aligned. The regular `NibbleSlice` is
/// right-aligned, meaning it does not support efficient truncation from the right side.
///
/// This is an immutable struct. No operations actually change it.
/// TODO len is to big, just store a bool to indicate if aligned or not (for radix 16 it is).
pub struct LeftNibbleSlice<'a> {
	pub(super) bytes: &'a [u8],
	pub(super) len: usize,
}

impl<'a> LeftNibbleSlice<'a> {
	/// Constructs a byte-aligned nibble slice from a byte slice.
	pub fn new(bytes: &'a [u8]) -> Self {
		LeftNibbleSlice {
			bytes,
			len: bytes.len() * NIBBLE_PER_BYTE,
		}
	}

	/// Constructs a byte-aligned nibble slice from a byte slice, and a given
	/// number of nibble len.
	/// TODO find better name
	pub fn new_len(bytes: &'a [u8], len: usize) -> Self {
		debug_assert!(bytes.len() * NIBBLE_PER_BYTE >= len) ;
		LeftNibbleSlice {
			bytes,
			len,
		}
	}

	/// Returns the length of the slice in nibbles.
	pub fn len(&self) -> usize {
		self.len
	}

	/// TODO remove: do no help, just temporar use.
	pub fn bytes(&self) -> &[u8] {
		self.bytes
	}

	/// Get the nibble at a nibble index padding with a 0 nibble. Returns None if the index is
	/// out of bounds.
	pub fn at(&self, index: usize) -> Option<u8> {
		if index < self.len() {
			Some(nibble_ops::left_nibble_at(self.bytes, index))
		} else {
			None
		}
	}

	/// Returns a new slice truncated from the right side to the given length. If the given length
	/// is greater than that of this slice, the function just returns a copy.
	pub fn truncate(&self, len: usize) -> Self {
		LeftNibbleSlice {
			bytes: self.bytes,
			len: cmp::min(len, self.len),
		}
	}

	/// Returns whether the given slice is a prefix of this one.
	pub fn starts_with(&self, prefix: &LeftNibbleSlice<'a>) -> bool {
		self.truncate(prefix.len()) == *prefix
	}

	/// Returns whether another regular (right-aligned) nibble slice is contained in this one at
	/// the given offset.
	pub fn contains(&self, partial: &NibbleSlice, offset: usize) -> bool {
		(0..partial.len()).all(|i| self.at(offset + i) == Some(partial.at(i)))
	}

	fn cmp(&self, other: &Self) -> Ordering {
		let common_len = cmp::min(self.len(), other.len());
		let common_byte_len = common_len / NIBBLE_PER_BYTE;

		// Quickly compare the common prefix of the byte slices.
		match self.bytes[..common_byte_len].cmp(&other.bytes[..common_byte_len]) {
			Ordering::Equal => {}
			ordering => return ordering,
		}

		// Compare nibble-by-nibble (either 0 or 1 nibbles) any after the common byte prefix.
		for i in (common_byte_len * NIBBLE_PER_BYTE)..common_len {
			let a = self.at(i).expect("i < len; len == self.len() qed");
			let b = other.at(i).expect("i < len; len == other.len(); qed");
			match a.cmp(&b) {
				Ordering::Equal => {}
				ordering => return ordering,
			}
		}

		// If common nibble prefix is the same, finally compare lengths.
		self.len().cmp(&other.len())
	}

	/// Compare a slice with another, also returns true if one does contain another
	/// (if cmp less then second contains first, if cmp greater first contains second).
	/// TODO return only to value? TODO force inline?
	pub fn cmp_common_and_starts_with(&self, other: &Self) -> (Ordering, usize, bool) {
		let common_len = cmp::min(self.len(), other.len());
		let common_byte_len = common_len / NIBBLE_PER_BYTE;

		for a in 0 .. common_byte_len {
			if self.bytes[a] != other.bytes[a] {
				let ordering = self.bytes[a].cmp(&other.bytes[a]);
				let commons = a * NIBBLE_PER_BYTE + nibble_ops::left_common(self.bytes[a], other.bytes[a]);
				let starts_with = match ordering {
					Ordering::Greater => {
						commons == other.len()
					},
					Ordering::Less => {
						commons == self.len()
					},
					Ordering::Equal => unreachable!(),
				};
				return (ordering, commons, starts_with);
			}
		}
	
		// Compare nibble-by-nibble (either 0 or 1 nibbles) any after the common byte prefix.
		for i in (common_byte_len * NIBBLE_PER_BYTE)..common_len {
			let a = self.at(i).expect("i < len; len == self.len() qed");
			let b = other.at(i).expect("i < len; len == other.len(); qed");
			match a.cmp(&b) {
				Ordering::Equal => {}
				Ordering::Greater => {
					return (Ordering::Greater, i, i == other.len());
				},
				Ordering::Less => {
					return (Ordering::Less, i, i == self.len());
				},
			}
		}

		// If common nibble prefix is the same, finally compare lengths.
		(self.len().cmp(&other.len()), common_len, true)
	}

	// TODO consider renaming to common_depth
	pub fn common_length(&self, other: &Self) -> usize {
		let common_len = cmp::min(self.len(), other.len());
		let common_byte_len = common_len / NIBBLE_PER_BYTE;

		for a in 0 .. common_byte_len {
			if self.bytes[a] != other.bytes[a] {
				return a * NIBBLE_PER_BYTE + nibble_ops::left_common(self.bytes[a], other.bytes[a]);
			}
		}
	
		// Compare nibble-by-nibble (either 0 or 1 nibbles) any after the common byte prefix.
		for i in (common_byte_len * NIBBLE_PER_BYTE)..common_len {
			let a = self.at(i).expect("i < len; len == self.len() qed");
			let b = other.at(i).expect("i < len; len == other.len(); qed");
			if a != b {
				return i;
			}
		}

		common_len
	}
}

impl<'a> PartialEq for LeftNibbleSlice<'a> {
	fn eq(&self, other: &Self) -> bool {
		let len = self.len();
		if other.len() != len {
			return false;
		}

		// Quickly compare the common prefix of the byte slices.
		let byte_len = len / NIBBLE_PER_BYTE;
		if self.bytes[..byte_len] != other.bytes[..byte_len] {
			return false;
		}

		// Compare nibble-by-nibble (either 0 or 1 nibbles) any after the common byte prefix.
		for i in (byte_len * NIBBLE_PER_BYTE)..len {
			let a = self.at(i).expect("i < len; len == self.len() qed");
			let b = other.at(i).expect("i < len; len == other.len(); qed");
			if a != b {
				return false
			}
		}

		true
	}
}

impl<'a> Eq for LeftNibbleSlice<'a> {}

impl<'a> PartialOrd for LeftNibbleSlice<'a> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl<'a> Ord for LeftNibbleSlice<'a> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.cmp(other)
	}
}

#[cfg(feature = "std")]
impl<'a> std::fmt::Debug for LeftNibbleSlice<'a> {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		for i in 0..self.len() {
			let nibble = self.at(i).expect("i < self.len(); qed");
			match i {
				0 => write!(f, "{:01x}", nibble)?,
				_ => write!(f, "'{:01x}", nibble)?,
			}
		}
		Ok(())
	}
}

impl<'a> Into<hash_db::Prefix<'a>> for LeftNibbleSlice<'a> {
	fn into(self) -> hash_db::Prefix<'a> {
		match self.len % nibble_ops::NIBBLE_PER_BYTE {
			0 => {
				(&self.bytes[..self.len / 2], None)
			},
			1 => {
				(&self.bytes[..self.len / 2], self.at(self.len - 1))
			},
			_ => {
				unreachable!();
			},
		}
	}
}

impl<'a> From<&'a crate::nibble::NibbleVec> for LeftNibbleSlice<'a> {
	fn from(s: &'a crate::nibble::NibbleVec) -> Self {
		LeftNibbleSlice {
			bytes: s.inner.as_slice(),
			len: s.len,
		}
	}
}

impl<'a> From<&'a [u8]> for LeftNibbleSlice<'a> {
	fn from(s: &'a [u8]) -> Self {
		LeftNibbleSlice {
			bytes: s,
			len: s.len() * nibble_ops::NIBBLE_PER_BYTE,
		}
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_len() {
		assert_eq!(LeftNibbleSlice::new(&[]).len(), 0);
		assert_eq!(LeftNibbleSlice::new(&b"hello"[..]).len(), 10);
		assert_eq!(LeftNibbleSlice::new(&b"hello"[..]).truncate(7).len(), 7);
	}

	#[test]
	fn test_at() {
		let slice = LeftNibbleSlice::new(&b"\x01\x23\x45\x67"[..]).truncate(7);
		assert_eq!(slice.at(0), Some(0));
		assert_eq!(slice.at(6), Some(6));
		assert_eq!(slice.at(7), None);
		assert_eq!(slice.at(8), None);
	}

	#[test]
	fn test_starts_with() {
		assert!(
			LeftNibbleSlice::new(b"hello").starts_with(&LeftNibbleSlice::new(b"heli").truncate(7))
		);
		assert!(
			!LeftNibbleSlice::new(b"hello").starts_with(&LeftNibbleSlice::new(b"heli").truncate(8))
		);
	}

	#[test]
	fn test_contains() {
		assert!(
			LeftNibbleSlice::new(b"hello").contains(&NibbleSlice::new_offset(b"ello", 0), 2)
		);
		assert!(
			LeftNibbleSlice::new(b"hello").contains(&NibbleSlice::new_offset(b"ello", 1), 3)
		);
		assert!(
			!LeftNibbleSlice::new(b"hello").contains(&NibbleSlice::new_offset(b"allo", 1), 3)
		);
		assert!(
			!LeftNibbleSlice::new(b"hello").contains(&NibbleSlice::new_offset(b"ello!", 1), 3)
		);
	}

	#[test]
	fn test_cmp() {
		assert!(LeftNibbleSlice::new(b"hallo") < LeftNibbleSlice::new(b"hello"));
		assert!(LeftNibbleSlice::new(b"hello") > LeftNibbleSlice::new(b"hallo"));
		assert_eq!(
			LeftNibbleSlice::new(b"hello").cmp(&LeftNibbleSlice::new(b"hello")),
			Ordering::Equal
		);

		assert!(
			LeftNibbleSlice::new(b"hello\x10")
				< LeftNibbleSlice::new(b"hello\x20").truncate(11)
		);
		assert!(
			LeftNibbleSlice::new(b"hello\x20").truncate(11)
				> LeftNibbleSlice::new(b"hello\x10")
		);

		assert!(
			LeftNibbleSlice::new(b"hello\x10").truncate(11)
				< LeftNibbleSlice::new(b"hello\x10")
		);
		assert!(
			LeftNibbleSlice::new(b"hello\x10")
				> LeftNibbleSlice::new(b"hello\x10").truncate(11)
		);
		assert_eq!(
			LeftNibbleSlice::new(b"hello\x10").truncate(11)
				.cmp(&LeftNibbleSlice::new(b"hello\x10").truncate(11)),
			Ordering::Equal
		);
	}
}
