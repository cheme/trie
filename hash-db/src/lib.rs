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

//! Database of byte-slices keyed to their hash.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
use std::fmt::Debug;
#[cfg(feature = "std")]
use std::hash;
#[cfg(not(feature = "std"))]
use core::hash;

#[cfg(feature = "std")]
pub trait MaybeDebug: Debug {}
#[cfg(feature = "std")]
impl<T: Debug> MaybeDebug for T {}
#[cfg(not(feature = "std"))]
pub trait MaybeDebug {}
#[cfg(not(feature = "std"))]
impl<T> MaybeDebug for T {}

#[cfg(feature = "std")]
use std::ops::Range;
#[cfg(not(feature = "std"))]
use core::ops::Range;

/// A trie node prefix, it is the nibble path from the trie root
/// to the trie node.
/// For a node containing no partial key value it is the full key.
/// For a value node or node containing a partial key, it is the full key minus its node partial
/// nibbles (the node key can be split into prefix and node partial).
/// Therefore it is always the leftmost portion of the node key, so its internal representation
/// is a non expanded byte slice followed by a last padded byte representation.
/// The padded byte is an optional padded value.
pub type Prefix<'a> = (&'a[u8], Option<u8>);

/// An empty prefix constant.
/// Can be use when the prefix is not use internally
/// or for root nodes.
pub static EMPTY_PREFIX: Prefix<'static> = (&[], None);

/// Trait describing an object that can hash a slice of bytes. Used to abstract
/// other types over the hashing algorithm. Defines a single `hash` method and an
/// `Out` associated type with the necessary bounds.
pub trait Hasher: Sync + Send {
	/// The output type of the `Hasher`
	type Out: AsRef<[u8]> + AsMut<[u8]> + Default + MaybeDebug + PartialEq + Eq
		+ hash::Hash + Send + Sync + Clone + Copy;
	/// What to use to build `HashMap`s with this `Hasher`.
	type StdHasher: Sync + Send + Default + hash::Hasher;
	/// The length in bytes of the `Hasher` output.
	const LENGTH: usize;

	/// Compute the hash of the provided slice of bytes returning the `Out` type of the `Hasher`.
	fn hash(x: &[u8]) -> Self::Out;
}

/// Small trait for to allow using buffer of type [u8; H::LENGTH * 2].
pub trait BinaryHasher: Hasher {
	/// Hash for the empty content (is hash(&[])).
	const NULL_HASH: &'static [u8];
	type Buffer: AsRef<[u8]> + AsMut<[u8]> + Default;
}

pub trait HasherComplex: BinaryHasher {

	/// Alternate hash with complex proof allowed
	fn hash_complex<
		I: Iterator<Item = (bool, Range<usize>)>,
		I2: Iterator<Item = <Self as Hasher>::Out>,
	>(
		x: &[u8],
		l: ComplexLayout<I, I2>,
	) -> Self::Out;
}

/// Trait modelling a plain datastore whose key is a fixed type.
/// The caller should ensure that a key only corresponds to
/// one value.
pub trait PlainDB<K, V>: Send + Sync + AsPlainDB<K, V> {
	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &K) -> Option<V>;

	/// Check for the existence of a hash-key.
	fn contains(&self, key: &K) -> bool;

	/// Insert a datum item into the DB. Insertions are counted and the equivalent
	/// number of `remove()`s must be performed before the data is considered dead.
	/// The caller should ensure that a key only corresponds to one value.
	fn emplace(&mut self, key: K, value: V);

	/// Remove a datum previously inserted. Insertions can be "owed" such that the
	/// same number of `insert()`s may happen without the data being eventually
	/// being inserted into the DB. It can be "owed" more than once.
	/// The caller should ensure that a key only corresponds to one value.
	fn remove(&mut self, key: &K);
}

/// Trait for immutable reference of PlainDB.
pub trait PlainDBRef<K, V> {
	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &K) -> Option<V>;

	/// Check for the existance of a hash-key.
	fn contains(&self, key: &K) -> bool;
}

impl<'a, K, V> PlainDBRef<K, V> for &'a dyn PlainDB<K, V> {
	fn get(&self, key: &K) -> Option<V> { PlainDB::get(*self, key) }
	fn contains(&self, key: &K) -> bool { PlainDB::contains(*self, key) }
}

impl<'a, K, V> PlainDBRef<K, V> for &'a mut dyn PlainDB<K, V> {
	fn get(&self, key: &K) -> Option<V> { PlainDB::get(*self, key) }
	fn contains(&self, key: &K) -> bool { PlainDB::contains(*self, key) }
}

/// Trait modelling datastore keyed by a hash defined by the `Hasher`.
pub trait HashDB<H: Hasher, T>: Send + Sync + AsHashDB<H, T> {
	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T>;

	/// Check for the existence of a hash-key.
	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool;

	/// Insert a datum item into the DB and return the datum's hash for a later lookup. Insertions
	/// are counted and the equivalent number of `remove()`s must be performed before the data
	/// is considered dead.
	fn insert(&mut self, prefix: Prefix, value: &[u8]) -> H::Out;

	/// Like `insert()`, except you provide the key and the data is all moved.
	fn emplace(&mut self, key: H::Out, prefix: Prefix, value: T);

	/// Remove a datum previously inserted. Insertions can be "owed" such that the same number of
	/// `insert()`s may happen without the data being eventually being inserted into the DB.
	/// It can be "owed" more than once.
	fn remove(&mut self, key: &H::Out, prefix: Prefix);
}

/// Describes how to hash a node given its layout
pub struct ComplexLayout<I, I2> {
	pub nb_children: usize,
	// can be calculated from decoded node from the
	// children bitmap and the children range
	pub children: I, // impl iterator < (is_defined, range) >
	pub nb_additional_hashes: usize,
	// TODO switch to iter??
	pub additional_hashes: I2,
}

pub struct ComplexLayoutIterValues<'a, HO, I> {
	nb_children: usize, 
	children: I, 
	buf: HO,
	node: &'a [u8],
}

impl<'a, HO: Default, I> ComplexLayoutIterValues<'a, HO, I> {
	pub fn new(nb_children: usize, children: I, node: &'a[u8]) -> Self {
		ComplexLayoutIterValues {
			nb_children,
			children,
			buf: Default::default(),
			node,
		}
	}
}

impl<'a, HO: AsMut<[u8]> + Clone, I: Iterator<Item = (bool, Range<usize>)>> Iterator for ComplexLayoutIterValues<'a, HO, I> {
	type Item = (bool, HO);

	fn next(&mut self) -> Option<Self::Item> {
		if let Some((is_defined, range)) = self.children.next() {
			let dest = self.buf.as_mut();
			let range_len = range.len();
			dest[..range_len].copy_from_slice(&self.node[range]);
			for i in range_len..dest.len() {
				dest[i] = 0;
			}
			// TODO the input iterator is HO but could really be &HO, would need some
			// change on trie_root to.
			Some((is_defined, self.buf.clone()))
		} else {
			None
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		// warning we unsafely presume the complexe layout
		// is only defined values.
			(self.nb_children, Some(self.nb_children))
	}
}

pub struct ComplexLayoutIterValues2<'a, HO, I> {
	nb_children: usize,
	children: I,
	buf: HO,
	node: &'a [u8],
	ix: usize,
}

impl<'a, HO: AsMut<[u8]> + Clone, I: Iterator<Item = (bool, Range<usize>)>> Iterator for ComplexLayoutIterValues2<'a, HO, I> {
	type Item = (usize, HO);

	fn next(&mut self) -> Option<Self::Item> {
		if let Some((is_defined, range)) = self.children.next() {
			assert!(is_defined);
			let dest = self.buf.as_mut();
			let range_len = range.len();
			dest[..range_len].copy_from_slice(&self.node[range]);
			for i in range_len..dest.len() {
				dest[i] = 0;
			}
			let ix = self.ix;
			self.ix += 1;
			Some((ix, self.buf.clone()))
		} else {
			None
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(0, Some(self.nb_children))
	}
}


/// Same as HashDB but can modify the value upon storage, and apply
/// `HasherComplex`.
pub trait HashDBComplex<H: HasherComplex, T>: Send + Sync + AsHashDB<H, T> {
	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T>;

	/// Check for the existence of a hash-key.
	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool;

	/// Insert a datum item into the DB and return the datum's hash for a later lookup. Insertions
	/// are counted and the equivalent number of `remove()`s must be performed before the data
	/// is considered dead.
	fn insert_complex<
		I: Iterator<Item = (bool, Range<usize>)>,
		I2: Iterator<Item = H::Out>,
	>(
		&mut self,
		prefix: Prefix,
		value: &[u8],
		layout: ComplexLayout<I, I2>,
	) -> H::Out;

	/// Like `insert()`, except you provide the key and the data is all moved.
	fn emplace(&mut self, key: H::Out, prefix: Prefix, value: T);

	/// Remove a datum previously inserted. Insertions can be "owed" such that the same number of
	/// `insert()`s may happen without the data being eventually being inserted into the DB.
	/// It can be "owed" more than once.
	fn remove(&mut self, key: &H::Out, prefix: Prefix);
}

/// Trait for immutable reference of HashDB.
pub trait HashDBRef<H: Hasher, T> {
	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T>;

	/// Check for the existance of a hash-key.
	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool;
}

impl<'a, H: Hasher, T> HashDBRef<H, T> for &'a dyn HashDB<H, T> {
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T> { HashDB::get(*self, key, prefix) }
	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool {
		HashDB::contains(*self, key, prefix)
	}
}

impl<'a, H: Hasher, T> HashDBRef<H, T> for &'a mut dyn HashDB<H, T> {
	fn get(&self, key: &H::Out, prefix: Prefix) -> Option<T> { HashDB::get(*self, key, prefix) }
	fn contains(&self, key: &H::Out, prefix: Prefix) -> bool {
		HashDB::contains(*self, key, prefix)
	}
}

/// Upcast trait for HashDB.
pub trait AsHashDB<H: Hasher, T> {
	/// Perform upcast to HashDB for anything that derives from HashDB.
	fn as_hash_db(&self) -> &dyn HashDB<H, T>;
	/// Perform mutable upcast to HashDB for anything that derives from HashDB.
	fn as_hash_db_mut<'a>(&'a mut self) -> &'a mut (dyn HashDB<H, T> + 'a);
}

/// Upcast trait for PlainDB.
pub trait AsPlainDB<K, V> {
	/// Perform upcast to PlainDB for anything that derives from PlainDB.
	fn as_plain_db(&self) -> &dyn PlainDB<K, V>;
	/// Perform mutable upcast to PlainDB for anything that derives from PlainDB.
	fn as_plain_db_mut<'a>(&'a mut self) -> &'a mut (dyn PlainDB<K, V> + 'a);
}

// NOTE: There used to be a `impl<T> AsHashDB for T` but that does not work with generics.
// See https://stackoverflow.com/questions/48432842/
// implementing-a-trait-for-reference-and-non-reference-types-causes-conflicting-im
// This means we need concrete impls of AsHashDB in several places, which somewhat defeats
// the point of the trait.
impl<'a, H: Hasher, T> AsHashDB<H, T> for &'a mut dyn HashDB<H, T> {
	fn as_hash_db(&self) -> &dyn HashDB<H, T> { &**self }
	fn as_hash_db_mut<'b>(&'b mut self) -> &'b mut (dyn HashDB<H, T> + 'b) { &mut **self }
}

#[cfg(feature = "std")]
impl<'a, K, V> AsPlainDB<K, V> for &'a mut dyn PlainDB<K, V> {
	fn as_plain_db(&self) -> &dyn PlainDB<K, V> { &**self }
	fn as_plain_db_mut<'b>(&'b mut self) -> &'b mut (dyn PlainDB<K, V> + 'b) { &mut **self }
}

/*

TODO this is rather shit at this point not delete yet just in case

/// Fix hash implementation, it needs to have
/// same output length as input for rounds of hashing.
/// TODO consider moving in its own crate.
pub trait FixHash {
	type Hasher: Hasher;
	/// if true, then when processing two leaf we do a finalize round.
	/// TODO I think only one may be insecure but do not remember
	/// the rational, also this should be an associated constant
	/// (TODO group types parameters)
	///
	/// Tells if state/iv can be initiated from first element of pair.
	///
	/// TODO could it be skipped in any circumstance or the other way
	/// arount, todo check blake3 permutation
	///
	/// Need first hashed implies that we cannot use back a state by
	/// calling a second hash.
	/// TODO write a test case for that !!!
	/// TODO rename to need finalize??
	/// TODO for keccack we could only hash(hash1 xor hash2)?
	const NEED_FIRST_HASHED: bool;
	/// Value of empty hash at some given depth 
	/// TODO test case it
	const EMPTY_HASHES: &'static [&'static [u8]];

	// The fix hash type of the `Hasher`. -> is Hasher::Out
//	type Out: AsRef<[u8]> + AsMut<[u8]> + Default + hash_db::MaybeDebug + PartialEq + Eq
//		+ Send + Sync + Clone + Copy;

	// The length in bytes of the `Out` output. -> is Hasher::Length
//	const LENGTH: usize;

	/// Compute the hash 
	fn new(first: <Self::Hasher as Hasher>::Out) -> Self;
	/// Compute the hash 
	fn hash(&mut self, second: &<Self::Hasher as Hasher>::Out);
	/// Access current state (if NEED_FIRST_HASHED is false).
	fn current_state(&self) -> &<Self::Hasher as Hasher>::Out;
	/// Extract hash (if NEED_FIRST_HASHED is true).
	fn finalize(self) -> <Self::Hasher as Hasher>::Out;
}
*/

