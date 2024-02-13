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

#[cfg_attr(feature = "std", derive(Debug))]
pub enum RangeProofError {
	/// No content to read from.
	EndOfStream,

	/// Unsupported usage.
	Unsupported,

	/// Error from IO, generally Read or Write.
	#[cfg(feature = "std")]
	StdIO(std::io::Error),
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
	fn write(&mut self, buf: &[u8]) -> Result<usize>;
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
	fn write(&mut self, buf: &[u8]) -> Result<usize> {
		let written = self.inner.write(buf)?;
		self.written += written;
		Ok(written)
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
	fn write(&mut self, buff: &[u8]) -> Result<usize> {
		self.extend_from_slice(buff);
		Ok(buff.len())
	}
	fn flush(&mut self) -> Result<()> {
		Ok(())
	}
}

#[derive(Clone, Copy)]
pub struct NoWrite;

// TODO no need for NoWrite
impl Write for NoWrite {
	fn write(&mut self, _: &[u8]) -> Result<usize> {
		Err(RangeProofError::Unsupported)
	}

	fn flush(&mut self) -> Result<()> {
		Ok(())
	}
}

impl CountedWrite for NoWrite {
	fn written(&self) -> usize {
		0
	}
}

#[cfg(feature = "std")]
impl From<std::io::Error> for RangeProofError {
	fn from(e: std::io::Error) -> Self {
		if let std::io::ErrorKind::UnexpectedEof = e.kind() {
			return RangeProofError::EndOfStream;
		}

		RangeProofError::StdIO(e)
	}
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, Copy)]
#[repr(u8)]
// TODO rename Partial per key
pub enum ProofOp {
	Partial,     // slice next, with size as number of nibbles. Attached could be size.
	Value,       // value next. Attached could be size.
	DropPartial, // followed by depth. Attached could be size.
	Hashes,      /* followed by consecutive defined hash, then bitmap of maximum 8 possibly
	              * defined hash then defined amongst them, then 8 next and repeat
	              * for possible. Attached could be bitmap.
	              * When structure allows either inline or hash, we add a bit in the bitmap
	              * to indicate if inline or single value hash: the first one */
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
			out.write(&[0b1000_0000 | to_encode as u8])?;
			to_encode >>= 7;
		}
		out.write(&[to_encode as u8])?;
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
		Err(RangeProofError::EndOfStream)
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

		output.write(&[header])?;
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
			match (op) {
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

	// TODO useless??
	pub fn encode<I: Iterator<Item = bool>>(&mut self, has_children: I) {
		for (i, v) in has_children.enumerate() {
			if v {
				self.set(i);
			}
		}
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
