// Copyright 2017, 2019 Parity Technologies
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

//! Byte packed encoded array.
//! Can be use to replace an array and skip serializing
//! deserializing step for persistent storage.

// TODO parameterized u64 (historied value) state (put it in config).

// TODO next split between consecutive indexed values
// TODO next split consecutive with range indexing

use crate::rstd::marker::PhantomData;
use crate::rstd::borrow::Cow;
use super::HistoriedValue;

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
/// Arraylike buffer with in place byte data.
/// Can be written as is in underlying
/// storage.
/// Could be use for direct access memory to.
pub struct EncodedArray<'a, F>(EncodedArrayBuff<'a>, PhantomData<F>);

#[derive(Debug)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
enum EncodedArrayBuff<'a> {
	Cow(Cow<'a, [u8]>),
	Mut(&'a mut Vec<u8>),
}

impl<'a> EncodedArrayBuff<'a> {
	pub fn to_mut(&mut self) -> &mut Vec<u8> {
		match self {
			EncodedArrayBuff::Cow(c) => c.to_mut(),
			EncodedArrayBuff::Mut(m) => m,
		}
	}
	pub fn into_owned(self) -> Vec<u8> {
		match self {
			EncodedArrayBuff::Cow(c) => c.into_owned(),
			EncodedArrayBuff::Mut(m) => m.clone(),
		}
	}
}

impl<'a> crate::rstd::ops::Deref for EncodedArrayBuff<'a> {
	type Target = [u8];
	fn deref(&self) -> &Self::Target {
		match self {
			EncodedArrayBuff::Cow(c) => c.deref(),
			EncodedArrayBuff::Mut(m) => m.deref(),
		}
	}
}

impl<'a> crate::rstd::ops::DerefMut for EncodedArrayBuff<'a> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.to_mut()[..]
	}
}

impl<'a> Clone for EncodedArrayBuff<'a> {
	fn clone(&self) -> Self {
		match self {
			EncodedArrayBuff::Cow(c) => EncodedArrayBuff::Cow(c.clone()),
			EncodedArrayBuff::Mut(m) => {
				let m: Vec<u8> = (*m).clone();
				EncodedArrayBuff::Cow(Cow::Owned(m))
			}
		}
	}
}

/// EncodedArray specific behavior.
pub trait EncodedArrayConfig {
	/// encoded empty slice
	fn empty() -> &'static [u8];
	/// size at start for encoding version.
	fn version_len() -> usize;
}

#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
/// Serialize without versioning.
pub struct NoVersion;

#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
/// Serialize with default verison
pub struct DefaultVersion;

impl EncodedArrayConfig for NoVersion {
	fn empty() -> &'static [u8] {
		&EMPTY_SERIALIZED
	}
	fn version_len() -> usize {
		0
	}
}

impl EncodedArrayConfig for DefaultVersion {
	fn empty() -> &'static [u8] {
		&DEFAULT_VERSION_EMPTY_SERIALIZED
	}
	fn version_len() -> usize {
		1
	}
}

// encoding size as u64
const SIZE_BYTE_LEN: usize = 8;

// Basis implementation to be on par with implementation using
// vec like container. Those method could be move to a trait
// implementation.
// Those function requires checked index.
impl<'a, F: EncodedArrayConfig> EncodedArray<'a, F> {

	pub fn into_owned(self) -> EncodedArray<'static, F> {
    EncodedArray(EncodedArrayBuff::Cow(Cow::from(self.0.into_owned())), PhantomData)
  }

	pub fn into_vec(self) -> Vec<u8> {
    self.0.into_owned()
  }

	pub(crate) fn len(&self) -> usize {
		let len = self.0.len();
		self.read_le_usize(len - SIZE_BYTE_LEN) as usize
	}

	pub(crate) fn clear(&mut self) {
		self.write_le_usize(F::version_len(), 0);
		self.0.to_mut().truncate(F::version_len() + SIZE_BYTE_LEN);
	}

	#[cfg(test)]
	fn truncate(&mut self, index: usize) {
		if index == 0 {
			self.clear();
			return;
		}
		let len = self.len();
		if index >= len {
			return;
		}
		let start_ix = self.index_start();
		let new_start = self.index_element(index) as usize;
		let len_ix = index * SIZE_BYTE_LEN;
		self.slice_copy(start_ix, new_start, len_ix);
		self.write_le_usize(new_start + len_ix - SIZE_BYTE_LEN, index);
		self.0.to_mut().truncate(new_start + len_ix);
	}

	// index stay in truncated content
	pub(crate) fn truncate_until(&mut self, index: usize) {
		self.remove_range(0, index);
	}

	pub(crate) fn pop(&mut self) -> Option<HistoriedValue<Vec<u8>, u64>> {
		let len = self.len();
		if len == 0 {
			return None;
		}
		let start_ix = self.index_element(len - 1);
		let end_ix = self.index_start();
		let state = self.read_le_u64(start_ix);
		let value = self.0[start_ix + SIZE_BYTE_LEN..end_ix].to_vec();
		if len - 1 == 0 {
			self.clear();
			return Some(HistoriedValue { value, state })	
		} else {
			self.write_le_usize(self.0.len() - (SIZE_BYTE_LEN * 2), len - 1);
		};
		let ix_size = (len * SIZE_BYTE_LEN) - SIZE_BYTE_LEN;
		self.slice_copy(end_ix, start_ix, ix_size);
		self.0.to_mut().truncate(start_ix + ix_size);
		Some(HistoriedValue { value, state })
	}

	pub(crate) fn push(&mut self, val: HistoriedValue<&[u8], u64>) {
		self.push_extra(val, &[])
	}

	/// variant of push where part of the value is in a second slice.
	pub(crate) fn push_extra(&mut self, val: HistoriedValue<&[u8], u64>, extra: &[u8]) {
		let len = self.len();
		let start_ix = self.index_start();
		let end_ix = self.0.len();
		// A sized buffer and multiple index to avoid to big copy
		// should be use here.
		let mut new_ix = self.0[start_ix..end_ix].to_vec();
		// truncate here can be bad
		self.0.to_mut().truncate(start_ix + SIZE_BYTE_LEN);
		self.write_le_u64(start_ix, val.state);
		self.0.to_mut().extend_from_slice(val.value);
		self.0.to_mut().extend_from_slice(extra);
		self.0.to_mut().append(&mut new_ix);
		if len > 0 {
			self.write_le_usize(self.0.len() - SIZE_BYTE_LEN, start_ix);
			self.append_le_usize(len + 1);
		} else {
			self.write_le_usize(self.0.len() - SIZE_BYTE_LEN, 1);
		}
	}

	#[cfg(test)]
	fn remove(&mut self, index: usize) {
		self.remove_range(index, index + 1);
	}

	fn remove_range(&mut self, index: usize, end: usize) {
		if end == 0 {
			return;
		}
		let len = self.len();
		if len <= end - index && index == 0 {
			self.clear();
			return;
		}
		// eager removal is costy, running some gc impl
		// can be interesting.
		let elt_start = self.index_element(index);
		let start_ix = self.index_start();
		let elt_end = if end == len {
			start_ix
		} else {
			self.index_element(end) 
		};
		let delete_size = elt_end - elt_start;
		for _ in elt_start..elt_end {
			let _ = self.0.to_mut().remove(elt_start);
		}
		let start_ix = start_ix - delete_size;

		let len = len - (end - index);
		for i in index..end {
			let pos = i + (end - index);
			if pos < len {
				let old_value = self.read_le_usize(start_ix + pos * SIZE_BYTE_LEN);
				self.write_le_usize(start_ix + i * SIZE_BYTE_LEN, old_value - delete_size);
			}
		}
		let end_index = start_ix + len * SIZE_BYTE_LEN;
		self.write_le_usize(end_index - SIZE_BYTE_LEN, len);
		self.0.to_mut().truncate(end_index);

	}

	pub(crate) fn get_state(&self, index: usize) -> HistoriedValue<&[u8], u64> {
		let start_ix = self.index_element(index);
		let len = self.len();
		let end_ix = if index == len - 1 {
			self.index_start()
		} else {
			self.index_element(index + 1)
		};
		let state = self.read_le_u64(start_ix);
		HistoriedValue {
			value: &self.0[start_ix + SIZE_BYTE_LEN..end_ix],
			state,
		}
	}

}

const EMPTY_SERIALIZED: [u8; SIZE_BYTE_LEN] = [0u8; SIZE_BYTE_LEN];
const DEFAULT_VERSION: u8 = 1;
const DEFAULT_VERSION_EMPTY_SERIALIZED: [u8; SIZE_BYTE_LEN + 1] = {
	let mut buf = [0u8; SIZE_BYTE_LEN + 1];
	buf[0] = DEFAULT_VERSION;
	buf
};

impl<'a, F: EncodedArrayConfig> Default for EncodedArray<'a, F> {
	fn default() -> Self {
		EncodedArray(EncodedArrayBuff::Cow(Cow::Borrowed(F::empty())), PhantomData)
	}
}

impl<'a, F> Into<EncodedArray<'a, F>> for &'a[u8] {
	fn into(self) -> EncodedArray<'a, F> {
		EncodedArray(EncodedArrayBuff::Cow(Cow::Borrowed(self)), PhantomData)
	}
}

impl<F> Into<EncodedArray<'static, F>> for Vec<u8> {
	fn into(self) -> EncodedArray<'static, F> {
		EncodedArray(EncodedArrayBuff::Cow(Cow::Owned(self)), PhantomData)
	}
}

impl<'a, F> Into<EncodedArray<'a, F>> for &'a mut Vec<u8> {
	fn into(self) -> EncodedArray<'a, F> {
		EncodedArray(EncodedArrayBuff::Mut(self), PhantomData)
	}
}


// Utility function for basis implementation.
impl<'a, F: EncodedArrayConfig> EncodedArray<'a, F> {
	
	// Index at end, also contains the encoded size
	fn index_start(&self) -> usize {
		let nb_ix = self.len();
		if nb_ix == 0 { return F::version_len(); }
		let end = self.0.len();
		end - (nb_ix * SIZE_BYTE_LEN)
	}

	fn index_element(&self, position: usize) -> usize {
		if position == 0 {
			return F::version_len();
		}
		let i = self.index_start() + (position - 1) * SIZE_BYTE_LEN;
		self.read_le_usize(i)
	}

	// move part of array that can overlap
	// This is a memory inefficient implementation.
	fn slice_copy(&mut self, start_from: usize, start_to: usize, size: usize) {
		let buffer = self.0[start_from..start_from + size].to_vec();
		self.0.to_mut()[start_to..start_to + size].copy_from_slice(&buffer[..]);
	}

	// Usize encoded as le u64 (for historied value).
	fn read_le_u64(&self, pos: usize) -> u64 {
		let mut buffer = [0u8; SIZE_BYTE_LEN];
		buffer.copy_from_slice(&self.0[pos..pos + SIZE_BYTE_LEN]);
		u64::from_le_bytes(buffer)
	}

	// Usize encoded as le u64 (only for internal indexing).
	fn read_le_usize(&self, pos: usize) -> usize {
		let mut buffer = [0u8; SIZE_BYTE_LEN];
		buffer.copy_from_slice(&self.0[pos..pos + SIZE_BYTE_LEN]);
		u64::from_le_bytes(buffer) as usize
	}

	// Usize encoded as le u64.
	fn write_le_usize(&mut self, pos: usize, value: usize) {
		let buffer = (value as u64).to_le_bytes();
		self.0.to_mut()[pos..pos + SIZE_BYTE_LEN].copy_from_slice(&buffer[..]);
	}

	// Usize encoded as le u64.
	fn append_le_usize(&mut self, value: usize) {
		let buffer = (value as u64).to_le_bytes();
		self.0.to_mut().extend_from_slice(&buffer[..]);
	}

	// Usize encoded as le u64.
	fn write_le_u64(&mut self, pos: usize, value: u64) {
		let buffer = (value as u64).to_le_bytes();
		self.0.to_mut()[pos..pos + SIZE_BYTE_LEN].copy_from_slice(&buffer[..]);
	}

}

#[cfg(test)]
mod test {
	use super::*;

	fn test_serialized_basis<F: EncodedArrayConfig>(mut ser: EncodedArray<F>) {
		// test basis unsafe function similar to a simple vec
		// without index checking.
		let v1 = &b"val1"[..];
		let v2 = &b"value_2"[..];
		let v3 = &b"a third value 3"[..];

		assert_eq!(ser.len(), 0);
		assert_eq!(ser.pop(), None);
		ser.push((v1, 1).into());
		assert_eq!(ser.get_state(0), (v1, 1).into());
		assert_eq!(ser.pop(), Some((v1.to_vec(), 1).into()));
		assert_eq!(ser.len(), 0);
		ser.push((v1, 1).into());
		ser.push((v2, 2).into());
		ser.push((v3, 3).into());
		assert_eq!(ser.get_state(0), (v1, 1).into());
		assert_eq!(ser.get_state(1), (v2, 2).into());
		assert_eq!(ser.get_state(2), (v3, 3).into());
		assert_eq!(ser.pop(), Some((v3.to_vec(), 3).into()));
		assert_eq!(ser.len(), 2);
		ser.push((v3, 3).into());
		assert_eq!(ser.get_state(2), (v3, 3).into());
		ser.remove(0);
		assert_eq!(ser.len(), 2);
		assert_eq!(ser.get_state(0), (v2, 2).into());
		assert_eq!(ser.get_state(1), (v3, 3).into());
		ser.push((v1, 1).into());
		ser.remove(1);
		assert_eq!(ser.len(), 2);
		assert_eq!(ser.get_state(0), (v2, 2).into());
		assert_eq!(ser.get_state(1), (v1, 1).into());
		ser.push((v1, 1).into());
		ser.truncate(1);
		assert_eq!(ser.len(), 1);
		assert_eq!(ser.get_state(0), (v2, 2).into());
		ser.push((v1, 1).into());
		ser.push((v3, 3).into());
		ser.truncate_until(1);
		assert_eq!(ser.len(), 2);
		assert_eq!(ser.get_state(0), (v1, 1).into());
		assert_eq!(ser.get_state(1), (v3, 3).into());
		ser.push((v2, 2).into());
		ser.truncate_until(2);
		assert_eq!(ser.len(), 1);
		assert_eq!(ser.get_state(0), (v2, 2).into());

	}

	#[test]
	fn serialized_basis() {
		let ser1: EncodedArray<NoVersion> = Default::default();
		let ser2: EncodedArray<DefaultVersion> = Default::default();
		test_serialized_basis(ser1);
		test_serialized_basis(ser2);
	}
}
