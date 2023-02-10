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

//! Hasher implementation for the Keccak-256 hash

use digest::{
	consts::{U16, U32, U8},
	Digest,
};
use hash256_std_hasher::Hash256StdHasher;
use hash_db::Hasher;
use tiny_keccak::{Hasher as _, Keccak};

/// The `Keccak` hash output type.
pub type KeccakHash = [u8; 32];

type Blake2b256 = blake2::Blake2b<U32>;
pub fn blake2_256_into(data: &[u8], dest: &mut [u8; 32]) {
	dest.copy_from_slice(Blake2b256::digest(data).as_slice());
}

/// Do a Blake2 256-bit hash and return result.
pub fn blake2_256(data: &[u8]) -> [u8; 32] {
	let mut r = [0; 32];
	blake2_256_into(data, &mut r);
	r
}

/// Concrete `Hasher` impl for the Keccak-256 hash
#[derive(Default, Debug, Clone, PartialEq)]
pub struct KeccakHasher;
impl Hasher for KeccakHasher {
	type Out = KeccakHash;

	type StdHasher = Hash256StdHasher;

	const LENGTH: usize = 32;

	fn hash(data: &[u8]) -> Self::Out {
		blake2_256(data).into()
	}
	/*
	fn hash(x: &[u8]) -> Self::Out {
		let mut keccak = Keccak::v256();
		keccak.update(x);
		let mut out = [0u8; 32];
		keccak.finalize(&mut out);
		out
	}*/
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::collections::HashMap;

	#[test]
	fn hash256_std_hasher_works() {
		let hello_bytes = b"Hello world!";
		let hello_key = KeccakHasher::hash(hello_bytes);

		let mut h: HashMap<<KeccakHasher as Hasher>::Out, Vec<u8>> = Default::default();
		h.insert(hello_key, hello_bytes.to_vec());
		h.remove(&hello_key);

		let mut h: HashMap<
			<KeccakHasher as Hasher>::Out,
			Vec<u8>,
			std::hash::BuildHasherDefault<Hash256StdHasher>,
		> = Default::default();
		h.insert(hello_key, hello_bytes.to_vec());
		h.remove(&hello_key);
	}
}
