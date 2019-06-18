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

//! Reference-counted memory-based `HashDB` implementation.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), feature(alloc))]

extern crate hash_db;
extern crate parity_util_mem;
#[cfg(feature = "deprecated")]
#[cfg(feature = "std")]
extern crate heapsize;
#[cfg(not(feature = "std"))]
extern crate hashmap_core;
#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(test)] extern crate keccak_hasher;

use hash_db::{HashDB, HashDBRef, PlainDB, PlainDBRef, Hasher as KeyHasher, AsHashDB, AsPlainDB};
use parity_util_mem::{MallocSizeOf, MallocSizeOfOps};
#[cfg(feature = "deprecated")]
#[cfg(feature = "std")]
use heapsize::HeapSizeOf;
#[cfg(feature = "std")]
use std::{
	collections::hash_map::Entry,
	collections::HashMap,
	hash,
	mem,
	marker::PhantomData,
	cmp::Eq,
	borrow::Borrow,
};

#[cfg(not(feature = "std"))]
use hashmap_core::{
	HashMap,
	map::Entry,
};

#[cfg(not(feature = "std"))]
use core::{
	hash,
	mem,
	marker::PhantomData,
	cmp::Eq,
	borrow::Borrow,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
pub trait MaybeDebug: std::fmt::Debug {}
#[cfg(feature = "std")]
impl<T: std::fmt::Debug> MaybeDebug for T {}
#[cfg(not(feature = "std"))]
pub trait MaybeDebug {}
#[cfg(not(feature = "std"))]
impl<T> MaybeDebug for T {}

/// Reference-counted memory-based `HashDB` implementation.
///
/// Use `new()` to create a new database. Insert items with `insert()`, remove items
/// with `remove()`, check for existence with `contains()` and lookup a hash to derive
/// the data with `get()`. Clear with `clear()` and purge the portions of the data
/// that have no references with `purge()`.
///
/// # Example
/// ```rust
/// extern crate hash_db;
/// extern crate keccak_hasher;
/// extern crate memory_db;
///
/// use hash_db::{Hasher, HashDB};
/// use keccak_hasher::KeccakHasher;
/// use memory_db::{MemoryDB, HashKey};
/// fn main() {
///   let mut m = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
///   let d = "Hello world!".as_bytes();
///
///   let k = m.insert(&[], d);
///   assert!(m.contains(&k, &[]));
///   assert_eq!(m.get(&k, &[]).unwrap(), d);
///
///   m.insert(&[], d);
///   assert!(m.contains(&k, &[]));
///
///   m.remove(&k, &[]);
///   assert!(m.contains(&k, &[]));
///
///   m.remove(&k, &[]);
///   assert!(!m.contains(&k, &[]));
///
///   m.remove(&k, &[]);
///   assert!(!m.contains(&k, &[]));
///
///   m.insert(&[], d);
///   assert!(!m.contains(&k, &[]));

///   m.insert(&[], d);
///   assert!(m.contains(&k, &[]));
///   assert_eq!(m.get(&k, &[]).unwrap(), d);
///
///   m.remove(&k, &[]);
///   assert!(!m.contains(&k, &[]));
/// }
/// ```
#[derive(Clone)]
pub struct MemoryDB<H, KF, T>
	where
	H: KeyHasher,
	KF: KeyFunction<H>,
{
	data: HashMap<KF::Key, (T, i32)>,
	hashed_null_node: H::Out,
	null_node_data: T,
	_kf: PhantomData<KF>,
}

impl<H, KF, T> PartialEq<MemoryDB<H, KF, T>> for MemoryDB<H, KF, T>
	where 
	H: KeyHasher,
	KF: KeyFunction<H>,
	<KF as KeyFunction<H>>::Key: Eq + MaybeDebug,
	T: Eq + MaybeDebug,
{
	fn eq(&self, other: &MemoryDB<H, KF, T>) -> bool {
		for a in self.data.iter() {
			match other.data.get(&a.0) {
				Some(v) if v != a.1 => return false,
				None => return false,
				_ => (),
			}
		}
		true
	}
}

impl<H, KF, T> Eq for MemoryDB<H, KF, T>
	where 
	H: KeyHasher,
	KF: KeyFunction<H>,
	<KF as KeyFunction<H>>::Key: Eq + MaybeDebug,
				T: Eq + MaybeDebug,
{}
 
pub trait KeyFunction<H: KeyHasher> {
	type Key: Send + Sync + Clone + hash::Hash + Eq;

	fn key(hash: &H::Out, prefix: &[u8]) -> Self::Key;
}

/// Make database key from hash and prefix.
pub fn prefixed_key<H: KeyHasher>(key: &H::Out, prefix: &[u8]) -> Vec<u8> {
	let mut prefixed_key = Vec::with_capacity(key.as_ref().len() + prefix.len());
	prefixed_key.extend_from_slice(prefix);
	prefixed_key.extend_from_slice(key.as_ref());
	prefixed_key
}

/// Make database key from hash only.
pub fn hash_key<H: KeyHasher>(key: &H::Out, _prefix: &[u8]) -> H::Out {
	key.clone()
}

#[derive(Clone,Debug)]
/// Key function that only uses the hash
pub struct HashKey<H: KeyHasher>(PhantomData<H>);

impl<H: KeyHasher> KeyFunction<H> for HashKey<H> {
	type Key = H::Out;

	fn key(hash: &H::Out, prefix: &[u8]) -> H::Out {
		hash_key::<H>(hash, prefix)
	}
}

#[derive(Clone,Debug)]
/// Key function that concatenates prefix and hash.
pub struct PrefixedKey<H: KeyHasher>(PhantomData<H>);

impl<H: KeyHasher> KeyFunction<H> for PrefixedKey<H> {
	type Key = Vec<u8>;

	fn key(hash: &H::Out, prefix: &[u8]) -> Vec<u8> {
		prefixed_key::<H>(hash, prefix)
	}
}

impl<'a, H, KF, T> Default for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: From<&'a [u8]>,
	KF: KeyFunction<H>,
{
	fn default() -> Self {
		Self::from_null_node(&[0u8][..], [0u8][..].into())
	}
}

/// Create a new `MemoryDB` from a given null key/data
impl<H, KF, T> MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default,
	KF: KeyFunction<H>,
{
	/// Remove an element and delete it from storage if reference count reaches zero.
	/// If the value was purged, return the old value.
	pub fn remove_and_purge(&mut self, key: &<H as KeyHasher>::Out, prefix: &[u8]) -> Option<T> {
		if key == &self.hashed_null_node {
			return None;
		}
		let key = KF::key(key, prefix);
		match self.data.entry(key) {
			Entry::Occupied(mut entry) =>
				if entry.get().1 == 1 {
					Some(entry.remove().0)
				} else {
					entry.get_mut().1 -= 1;
					None
				},
			Entry::Vacant(entry) => {
				entry.insert((T::default(), -1)); // FIXME: shouldn't it be purged?
				None
			}
		}
	}
}

impl<'a, H: KeyHasher, KF, T> MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: From<&'a [u8]>,
	KF: KeyFunction<H>,
{
	/// Create a new `MemoryDB` from a given null key/data
	pub fn from_null_node(null_key: &'a [u8], null_node_data: T) -> Self {
		MemoryDB {
			data: HashMap::default(),
			hashed_null_node: H::hash(null_key),
			null_node_data,
			_kf: Default::default(),
		}
	}

	pub fn new(data: &'a [u8]) -> Self {
		Self::from_null_node(data, data.into())
	}

	/// Clear all data from the database.
	///
	/// # Examples
	/// ```rust
	/// extern crate hash_db;
	/// extern crate keccak_hasher;
	/// extern crate memory_db;
	///
	/// use hash_db::{Hasher, HashDB};
	/// use keccak_hasher::KeccakHasher;
	/// use memory_db::{MemoryDB, HashKey};
	///
	/// fn main() {
	///   let mut m = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
	///   let hello_bytes = "Hello world!".as_bytes();
	///   let hash = m.insert(&[], hello_bytes);
	///   assert!(m.contains(&hash, &[]));
	///   m.clear();
	///   assert!(!m.contains(&hash, &[]));
	/// }
	/// ```
	pub fn clear(&mut self) {
		self.data.clear();
	}

	/// Purge all zero-referenced data from the database.
	pub fn purge(&mut self) {
		self.data.retain(|_, &mut (_, rc)| rc != 0);
	}

	/// Return the internal map of hashes to data, clearing the current state.
	pub fn drain(&mut self) -> HashMap<KF::Key, (T, i32)> {
		mem::replace(&mut self.data, Default::default())
	}

	/// Grab the raw information associated with a key. Returns None if the key
	/// doesn't exist.
	///
	/// Even when Some is returned, the data is only guaranteed to be useful
	/// when the refs > 0.
	pub fn raw(&self, key: &<H as KeyHasher>::Out, prefix: &[u8]) -> Option<(&T, i32)> {
		if key == &self.hashed_null_node {
			return Some((&self.null_node_data, 1));
		}
		self.data.get(&KF::key(key, prefix)).map(|(value, count)| (value, *count))
	}

	/// Consolidate all the entries of `other` into `self`.
	pub fn consolidate(&mut self, mut other: Self) {
		for (key, (value, rc)) in other.drain() {
			match self.data.entry(key) {
				Entry::Occupied(mut entry) => {
					if entry.get().1 < 0 {
						entry.get_mut().0 = value;
					}

					entry.get_mut().1 += rc;
				}
				Entry::Vacant(entry) => {
					entry.insert((value, rc));
				}
			}
		}
	}

	/// Get the keys in the database together with number of underlying references.
	pub fn keys(&self) -> HashMap<KF::Key, i32> {
		self.data.iter()
			.filter_map(|(k, v)| if v.1 != 0 {
				Some((k.clone(), v.1))
			} else {
				None
			})
			.collect()
	}
}

#[cfg(feature = "deprecated")]
#[cfg(feature = "std")]
impl<H, KF, T> MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: HeapSizeOf,
	KF: KeyFunction<H>,
{
	#[deprecated(since="0.12.0", note="please use `size_of` instead")]
	/// Returns the size of allocated heap memory
	pub fn mem_used(&self) -> usize {
		0//self.data.heap_size_of_children()
		// TODO Reenable above when HeapSizeOf supports arrays.
	}
}

// `no_std` implementation requires that hasmap
// is implementated in parity-util-mem, that
// is currently not the case.
#[cfg(feature = "std")]
impl<H, KF, T> MallocSizeOf for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	H::Out: MallocSizeOf,
	T: MallocSizeOf,
	KF: KeyFunction<H>,
	KF::Key: MallocSizeOf,
{
	fn size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
		self.data.size_of(ops)
			+ self.null_node_data.size_of(ops)
			+ self.hashed_null_node.size_of(ops)
	}
}

// This is temporary code, we should use
// `parity-util-mem`, see
// https://github.com/paritytech/trie/issues/21
#[cfg(not(feature = "std"))]
impl<H, KF, T> MallocSizeOf for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	H::Out: MallocSizeOf,
	T: MallocSizeOf,
	KF: KeyFunction<H>,
	KF::Key: MallocSizeOf,
{
	fn size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
		use core::mem::size_of;
		let mut n = self.data.capacity() * (size_of::<T>() + size_of::<H>() + size_of::<usize>());
		for (k, v) in self.data.iter() {
			n += k.size_of(ops) + v.size_of(ops);
		}
		n + self.null_node_data.size_of(ops) + self.hashed_null_node.size_of(ops)
	}
}



impl<H, KF, T> PlainDB<H::Out, T> for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default + PartialEq<T> + for<'a> From<&'a [u8]> + Clone + Send + Sync,
	KF: Send + Sync + KeyFunction<H>,
	KF::Key: Borrow<[u8]> + for <'a> From<&'a [u8]>,
{
	fn get(&self, key: &H::Out) -> Option<T> {
		match self.data.get(key.as_ref()) {
			Some(&(ref d, rc)) if rc > 0 => Some(d.clone()),
			_ => None
		}
	}

	fn contains(&self, key: &H::Out) -> bool {
		match self.data.get(key.as_ref()) {
			Some(&(_, x)) if x > 0 => true,
			_ => false
		}
	}

	fn emplace(&mut self, key: H::Out, value: T) {
		match self.data.entry(key.as_ref().into()) {
			Entry::Occupied(mut entry) => {
				let &mut (ref mut old_value, ref mut rc) = entry.get_mut();
				if *rc <= 0 {
					*old_value = value;
				}
				*rc += 1;
			},
			Entry::Vacant(entry) => {
				entry.insert((value, 1));
			},
		}
	}

	fn remove(&mut self, key: &H::Out) {
		match self.data.entry(key.as_ref().into()) {
			Entry::Occupied(mut entry) => {
				let &mut (_, ref mut rc) = entry.get_mut();
				*rc -= 1;
			},
			Entry::Vacant(entry) => {
				entry.insert((T::default(), -1));
			},
		}
	}
}

impl<H, KF, T> PlainDBRef<H::Out, T> for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default + PartialEq<T> + for<'a> From<&'a [u8]> + Clone + Send + Sync,
	KF: Send + Sync + KeyFunction<H>,
	KF::Key: Borrow<[u8]> + for <'a> From<&'a [u8]>,
{
	fn get(&self, key: &H::Out) -> Option<T> { PlainDB::get(self, key) }
	fn contains(&self, key: &H::Out) -> bool { PlainDB::contains(self, key) }
}

impl<H, KF, T> HashDB<H, T> for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default + PartialEq<T> + for<'a> From<&'a [u8]> + Clone + Send + Sync,
	KF: Send + Sync + KeyFunction<H>,
{
	fn get(&self, key: &H::Out, prefix: &[u8]) -> Option<T> {
		if key == &self.hashed_null_node {
			return Some(self.null_node_data.clone());
		}

		let key = KF::key(key, prefix);
		match self.data.get(&key) {
			Some(&(ref d, rc)) if rc > 0 => Some(d.clone()),
			_ => None
		}
	}

	fn contains(&self, key: &H::Out, prefix: &[u8]) -> bool {
		if key == &self.hashed_null_node {
			return true;
		}

		let key = KF::key(key, prefix);
		match self.data.get(&key) {
			Some(&(_, x)) if x > 0 => true,
			_ => false
		}
	}

	fn emplace(&mut self, key: H::Out, prefix: &[u8], value: T) {
		if value == self.null_node_data {
			return;
		}

		let key = KF::key(&key, prefix);
		match self.data.entry(key) {
			Entry::Occupied(mut entry) => {
				let &mut (ref mut old_value, ref mut rc) = entry.get_mut();
				if *rc <= 0 {
					*old_value = value;
				}
				*rc += 1;
			},
			Entry::Vacant(entry) => {
				entry.insert((value, 1));
			},
		}
	}

	fn insert(&mut self, prefix: &[u8], value: &[u8]) -> H::Out {
		if T::from(value) == self.null_node_data {
			return self.hashed_null_node.clone();
		}

		let key = H::hash(value);
		HashDB::emplace(self, key, prefix, value.into());
		key
	}

	fn remove(&mut self, key: &H::Out, prefix: &[u8]) {
		if key == &self.hashed_null_node {
			return;
		}

		let key = KF::key(key, prefix);
		match self.data.entry(key) {
			Entry::Occupied(mut entry) => {
				let &mut (_, ref mut rc) = entry.get_mut();
				*rc -= 1;
			},
			Entry::Vacant(entry) => {
				entry.insert((T::default(), -1));
			},
		}
	}
}

impl<H, KF, T> HashDBRef<H, T> for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default + PartialEq<T> + for<'a> From<&'a [u8]> + Clone + Send + Sync,
	KF: Send + Sync + KeyFunction<H>,
{
	fn get(&self, key: &H::Out, prefix: &[u8]) -> Option<T> { HashDB::get(self, key, prefix) }
	fn contains(&self, key: &H::Out, prefix: &[u8]) -> bool { HashDB::contains(self, key, prefix) }
}

impl<H, KF, T> AsPlainDB<H::Out, T> for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default + PartialEq<T> + for<'a> From<&'a[u8]> + Clone + Send + Sync,
	KF: Send + Sync + KeyFunction<H>,
	KF::Key: Borrow<[u8]> + for <'a> From<&'a [u8]>,
{
	fn as_plain_db(&self) -> &dyn PlainDB<H::Out, T> { self }
	fn as_plain_db_mut(&mut self) -> &mut dyn PlainDB<H::Out, T> { self }
}

impl<H, KF, T> AsHashDB<H, T> for MemoryDB<H, KF, T>
where
	H: KeyHasher,
	T: Default + PartialEq<T> + for<'a> From<&'a[u8]> + Clone + Send + Sync,
	KF: Send + Sync + KeyFunction<H>,
{
	fn as_hash_db(&self) -> &dyn HashDB<H, T> { self }
	fn as_hash_db_mut(&mut self) -> &mut dyn HashDB<H, T> { self }
}

#[cfg(test)]
mod tests {
	use super::{MemoryDB, HashDB, KeyHasher, HashKey};
	use keccak_hasher::KeccakHasher;

	#[test]
	fn memorydb_remove_and_purge() {
		let hello_bytes = b"Hello world!";
		let hello_key = KeccakHasher::hash(hello_bytes);

		let mut m = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
		m.remove(&hello_key, &[]);
		assert_eq!(m.raw(&hello_key, &[]).unwrap().1, -1);
		m.purge();
		assert_eq!(m.raw(&hello_key, &[]).unwrap().1, -1);
		m.insert(&[], hello_bytes);
		assert_eq!(m.raw(&hello_key, &[]).unwrap().1, 0);
		m.purge();
		assert_eq!(m.raw(&hello_key, &[]), None);

		let mut m = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
		assert!(m.remove_and_purge(&hello_key, &[]).is_none());
		assert_eq!(m.raw(&hello_key, &[]).unwrap().1, -1);
		m.insert(&[], hello_bytes);
		m.insert(&[], hello_bytes);
		assert_eq!(m.raw(&hello_key, &[]).unwrap().1, 1);
		assert_eq!(&*m.remove_and_purge(&hello_key, &[]).unwrap(), hello_bytes);
		assert_eq!(m.raw(&hello_key, &[]), None);
		assert!(m.remove_and_purge(&hello_key, &[]).is_none());
	}

	#[test]
	fn consolidate() {
		let mut main = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
		let mut other = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
		let remove_key = other.insert(&[], b"doggo");
		main.remove(&remove_key, &[]);

		let insert_key = other.insert(&[], b"arf");
		main.emplace(insert_key, &[], "arf".as_bytes().to_vec());

		let negative_remove_key = other.insert(&[], b"negative");
		other.remove(&negative_remove_key, &[]);	// ref cnt: 0
		other.remove(&negative_remove_key, &[]);	// ref cnt: -1
		main.remove(&negative_remove_key, &[]);	// ref cnt: -1

		main.consolidate(other);

		assert_eq!(main.raw(&remove_key, &[]).unwrap(), (&"doggo".as_bytes().to_vec(), 0));
		assert_eq!(main.raw(&insert_key, &[]).unwrap(), (&"arf".as_bytes().to_vec(), 2));
		assert_eq!(main.raw(&negative_remove_key, &[]).unwrap(), (&"negative".as_bytes().to_vec(), -2));
	}

	#[test]
	fn default_works() {
		let mut db = MemoryDB::<KeccakHasher, HashKey<_>, Vec<u8>>::default();
		let hashed_null_node = KeccakHasher::hash(&[0u8][..]);
		assert_eq!(db.insert(&[], &[0u8][..]), hashed_null_node);
	}
}
