// Copyright 2019, 2024 Parity Technologies
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

use crate::{iterator::OpHash, nibble_ops, rstd::vec::Vec};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RangeProofError {
	/// No content to read from.
	EndOfStream,

	/// Stopping the range proof only on value from non inline node.
	ShouldSuspendOnValue,

	/// Unsupported usage.
	Unsupported,

	/// Error from IO, generally Read or Write.
	#[cfg(feature = "std")]
	StdIO(std::io::ErrorKind),
}

type Result<T> = core::result::Result<T, RangeProofError>;

/// Trait similar to std::io::Read, but that can run in no_std.
pub trait Read {
	fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>;
}

impl<'a> Read for &'a [u8] {
	fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
		let to_copy = buf.len();
		if self.len() < to_copy {
			return Err(RangeProofError::EndOfStream);
		}
		buf.copy_from_slice(&self[..to_copy]);
		*self = &self[to_copy..];
		Ok(())
	}
}

/// Trait similar to std::io::Write, but that can run in no_std.
pub trait Write {
	fn write_all(&mut self, buf: &[u8]) -> Result<()>;
	fn flush(&mut self) -> Result<()>;
}

/// Counter over Write.
pub trait CountedWrite: Write {
	// size written in write.
	// Warning depending on implementation this
	// is not starting at same size, so should
	// always be used to compare with an initial size.
	fn written(&self) -> usize;
}

pub struct Counted<T: Write> {
	pub inner: T,
	pub written: usize,
}

impl<T: Write> From<T> for Counted<T> {
	fn from(inner: T) -> Self {
		Self { inner, written: 0 }
	}
}

impl<T: Write> Write for Counted<T> {
	fn write_all(&mut self, buf: &[u8]) -> Result<()> {
		self.inner.write_all(buf)?;
		self.written += buf.len();
		Ok(())
	}

	fn flush(&mut self) -> Result<()> {
		self.inner.flush()
	}
}

impl<T: Write> CountedWrite for Counted<T> {
	fn written(&self) -> usize {
		self.written
	}
}

impl CountedWrite for Vec<u8> {
	fn written(&self) -> usize {
		self.len()
	}
}

impl Write for Vec<u8> {
	fn write_all(&mut self, buff: &[u8]) -> Result<()> {
		self.extend_from_slice(buff);
		Ok(())
	}
	fn flush(&mut self) -> Result<()> {
		Ok(())
	}
}

#[cfg(feature = "std")]
impl From<std::io::Error> for RangeProofError {
	fn from(e: std::io::Error) -> Self {
		if let std::io::ErrorKind::UnexpectedEof = e.kind() {
			return RangeProofError::EndOfStream;
		}

		RangeProofError::StdIO(e.kind())
	}
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ProofOp {
	/// Followed by size and partial key content.
	/// Emmitted everytime the iterator cursor advance on a key.
	/// Two consecutive `Partial` are not allowed (should be merged).
	/// In header attached could be size.
	Partial,
	/// Followed by value.
	/// Emmitted every time a node with an attached value is queried.
	/// In header attached could be size.
	Value,
	/// Folowed by depth.
	/// Emmitted everytime the iterator cursor goes back on a key.
	/// In header attached could be size.
	DropPartial,
	/// Folowed by bitmap of next items. Bitmap sized one byte max.
	/// Items in bitmap if present and possibly varlength (inline node or inline value), are
	/// using next bit of bitmap to indicate that.
	/// After each bitmap, hashes are written (fix size or prefix with len if varsize).
	/// Bitmap and hashes are emmited only when content is not queried and node is accessed.
	/// Can be emmitted when entering a child node for value if first access and child node not
	/// emmited before the accessed child node index.
	/// Can be emmitted when exiting node for children that are after last accessed child (value
	/// and all children if first access).
	/// In header attached could be bitmap content.
	Hashes,
}

impl ProofOp {
	// warn this does not presume from header encoding,
	// just a helper.
	fn as_u8(self: Self) -> u8 {
		match self {
			ProofOp::Partial => 0,
			ProofOp::Value => 1,
			ProofOp::DropPartial => 2,
			ProofOp::Hashes => 3,
		}
	}

	// warn this does not presume from header encoding,
	// just a helper.
	fn from_u8(encoded: u8) -> Option<Self> {
		Some(match encoded {
			0 => ProofOp::Partial,
			1 => ProofOp::Value,
			2 => ProofOp::DropPartial,
			3 => ProofOp::Hashes,
			_ => return None,
		})
	}
}

fn varint_encoded_len(len: u32) -> usize {
	if len == 0 {
		return 1
	}
	let len = 32 - len.leading_zeros() as usize;
	if len % 7 == 0 {
		len / 7
	} else {
		len / 7 + 1
	}
}

/// Tools for encoding range proof.
/// Mainly single byte header for ops and size encoding.
pub trait RangeProofCodec {
	fn varint_encode_into(len: u32, out: &mut impl CountedWrite) -> Result<()> {
		let mut to_encode = len;
		for _ in 0..varint_encoded_len(len) - 1 {
			out.write_all(&[0b1000_0000 | to_encode as u8])?;
			to_encode >>= 7;
		}
		out.write_all(&[to_encode as u8])?;
		Ok(())
	}

	fn varint_decode_from(input: &mut impl Read) -> Result<u32> {
		let mut value = 0u32;
		let mut buff = [0u8];
		let mut i = 0;
		loop {
			input.read_exact(&mut buff[..])?;
			let byte = buff[0];
			let last = byte & 0b1000_0000 == 0;
			value |= ((byte & 0b0111_1111) as u32) << (i * 7);
			if last {
				return Ok(value);
			}
			i += 1;
		}
	}

	/// return range of value that can be attached to this op.
	/// for bitmap it returns bitmap len.
	/// for others it attached a max size (if max then it got sub from varint next).
	fn op_attached_range(op: ProofOp) -> Option<u8>;

	/// op can have some data attached (depending on encoding).
	fn encode_op(op: ProofOp, attached: Option<u8>) -> u8;

	/// Return op and range attached.
	fn decode_op(encoded: u8) -> Option<(ProofOp, Option<u8>)>;

	fn encode_with_size(op: ProofOp, size: usize, output: &mut impl CountedWrite) -> Result<()> {
		let (attached, size) = if let Some(enc_size) = Self::op_attached_range(op) {
			if size < enc_size as usize {
				(Some(size as u8), None)
			} else {
				(Some(enc_size), Some(size - enc_size as usize))
			}
		} else {
			(None, Some(size))
		};

		let header = Self::encode_op(op, attached);

		output.write_all(&[header])?;
		if let Some(size) = size {
			Self::varint_encode_into(size as u32, output)?;
		}
		Ok(())
	}

	fn decode_size(op: ProofOp, attached: Option<u8>, input: &mut impl Read) -> Result<usize> {
		Ok(if let Some(s) = attached {
			let max = Self::op_attached_range(op).expect("got attached");
			if s < max {
				s as usize
			} else {
				s as usize + Self::varint_decode_from(input)? as usize
			}
		} else {
			Self::varint_decode_from(input)? as usize
		})
	}

	fn push_partial_key(
		to: usize,
		from: usize,
		key: &[u8],
		output: &mut impl CountedWrite,
	) -> Result<()> {
		let to_write = to - from;
		let aligned = to_write % nibble_ops::NIBBLE_PER_BYTE == 0;
		Self::encode_with_size(ProofOp::Partial, to_write, output)?;
		let start_aligned = from % nibble_ops::NIBBLE_PER_BYTE == 0;
		let start_ix = from / nibble_ops::NIBBLE_PER_BYTE;
		if start_aligned {
			let off = if aligned { 0 } else { 1 };
			let slice_to_write = &key[start_ix..key.len() - off];
			output.write_all(slice_to_write)?;
			if !aligned {
				output.write_all(&[nibble_ops::pad_left(key[key.len() - 1])])?;
			}
		} else {
			for i in start_ix..key.len() - 1 {
				let mut b = key[i] << 4;
				b |= key[i + 1] >> 4;
				output.write_all(&[b])?;
			}
			if !aligned {
				let b = key[key.len() - 1] << 4;
				output.write_all(&[b])?;
			}
		}
		Ok(())
	}

	fn push_value(value: &[u8], output: &mut impl CountedWrite) -> Result<()> {
		Self::encode_with_size(ProofOp::Value, value.len(), output)?;
		output.write_all(value)?;
		Ok(())
	}

	fn push_hashes<'a, I, O>(output: &mut O, mut iter_possible: I) -> Result<()>
	where
		O: CountedWrite,
		I: Iterator<Item = OpHash<'a>>,
	{
		let mut nexts: [OpHash; 8] = [OpHash::None; 8];
		let mut header_written = false;
		let mut i_hash;
		let mut i_bitmap;
		// if bit in previous bitmap (presence to true and type expected next).
		let mut prev_bit: Option<OpHash> = None;

		loop {
			i_bitmap = 0;
			i_hash = 0;
			let mut bitmap = Bitmap1::default();

			let header_bitmap_len = Self::op_attached_range(ProofOp::Hashes);
			let bound = if !header_written {
				if let Some(l) = header_bitmap_len {
					l as usize
				} else {
					8
				}
			} else {
				8
			};

			if let Some(h) = prev_bit.take() {
				debug_assert!(h.is_some());
				if h.is_var() {
					bitmap.set(i_bitmap);
				}
				i_bitmap += 1;
				if h.is_some() {
					nexts[i_hash] = h;
					i_hash += 1;
				}
			}

			while let Some(h) = iter_possible.next() {
				if h.is_some() {
					bitmap.set(i_bitmap);
					i_bitmap += 1;
					if i_bitmap == bound {
						prev_bit = Some(h);
						break;
					}
					if h.is_var() {
						bitmap.set(i_bitmap);
					}
				}
				i_bitmap += 1;
				if h.is_some() {
					nexts[i_hash] = h;
					i_hash += 1;
				}
				if i_bitmap == bound {
					break;
				}
			}

			if i_bitmap == 0 {
				debug_assert!(header_written && i_hash == 0);
				break
			}

			if !header_written {
				header_written = true;
				let header = Self::encode_op(ProofOp::Hashes, header_bitmap_len.map(|_| bitmap.0));
				if header_bitmap_len.is_some() {
					output.write_all(&[header])?;
				} else {
					output.write_all(&[header, bitmap.0])?;
				}
			} else {
				output.write_all(&[bitmap.0])?;
			}
			for j in 0..i_hash {
				match nexts[j] {
					OpHash::Fix(s) => {
						output.write_all(s)?;
					},
					OpHash::Var(s) => {
						Self::varint_encode_into(s.len() as u32, output)?;
						output.write_all(s)?;
					},
					OpHash::None => unreachable!(),
				}
			}
		}
		Ok(())
	}
}

/// Test codec.
pub struct VarIntSimple;

impl RangeProofCodec for VarIntSimple {
	fn op_attached_range(_: ProofOp) -> Option<u8> {
		None
	}

	fn encode_op(op: ProofOp, attached: Option<u8>) -> u8 {
		debug_assert!(attached.is_none());
		op.as_u8()
	}

	fn decode_op(encoded: u8) -> Option<(ProofOp, Option<u8>)> {
		ProofOp::from_u8(encoded).map(|e| (e, None))
	}
}

/// Test codec, with six bit any attached op.
pub struct VarIntSix;

impl RangeProofCodec for VarIntSix {
	fn op_attached_range(op: ProofOp) -> Option<u8> {
		match op {
			ProofOp::Partial | ProofOp::DropPartial => Some(1u8 << 6), // 0 len partial is invalid
			ProofOp::Value => Some((1u8 << 6) - 1),                    // 0 is valid
			// ranche.
			ProofOp::Hashes => Some(6),
		}
	}

	fn encode_op(op: ProofOp, attached: Option<u8>) -> u8 {
		let base = op.as_u8();
		debug_assert!(attached.is_some());
		let attached = if let Some(mut attached) = attached {
			match op {
				ProofOp::Partial | ProofOp::DropPartial => {
					attached -= 1;
				},
				_ => (),
			}
			debug_assert!((attached & 0b1100_0000) == 0);
			attached << 2
		} else {
			0
		};
		base | attached
	}

	fn decode_op(encoded: u8) -> Option<(ProofOp, Option<u8>)> {
		if let Some(op) = ProofOp::from_u8(encoded & 0b11) {
			let mut attached = encoded >> 2;
			match op {
				ProofOp::Partial | ProofOp::DropPartial => {
					// no 0 value
					attached += 1;
				},
				_ => (),
			}
			Some((op, Some(attached)))
		} else {
			None
		}
	}
}

#[derive(Default, Clone)]
pub struct Bitmap1(pub u8);

impl Bitmap1 {
	pub fn check(expected_len: usize) -> bool {
		debug_assert!(expected_len > 0);
		debug_assert!(expected_len < 9);
		(0xff >> expected_len) == 0
	}

	pub fn get(&self, i: usize) -> bool {
		debug_assert!(i < 8);
		self.0 & (0b0000_0001 << i) != 0
	}

	pub fn set(&mut self, i: usize) {
		debug_assert!(i < 8);
		self.0 |= 0b0000_0001 << i;
	}
}

#[test]
fn varint_encode_decode() {
	let mut buf = Vec::new();
	for i in 0..u16::MAX as u32 + 1 {
		VarIntSimple::varint_encode_into(i, &mut buf);
		assert_eq!(buf.len(), varint_encoded_len(i));
		assert_eq!((i, buf.len()), VarIntSimple::varint_decode_from(&mut buf.as_slice()).unwrap());
		buf.clear();
	}
}
