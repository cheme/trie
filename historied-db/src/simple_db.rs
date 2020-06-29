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

//! Simple db tools to store non historied information.

use derivative::Derivative;
use crate::rstd::{BTreeMap, btree_map::Entry, marker::PhantomData, vec::Vec};

/// simple serialize trait, could be a noop.
pub trait SerializeDB: Sized {
	/// When false, the trait implementation
	/// do not need to actually manage a db
	/// (see `()` implementation).
	const ACTIVE: bool = true;
	type Iter: Iterator<Item = (Vec<u8>, Vec<u8>)>;

	fn write(&mut self, c: &'static [u8], k: &[u8], v: &[u8]);
	fn remove(&mut self, c: &'static [u8], k: &[u8]);
	fn read(&self, c: &'static [u8], k: &[u8]) -> Option<Vec<u8>>;
	fn iter(&self, c: &'static [u8]) -> Self::Iter;

	fn contains_collection(collection: &[u8]) -> bool;

	// TODO remove ?
	fn read_collection<'a, DB, I>(db: &'a DB, collection: &'static[u8]) -> Option<Collection<'a, Self, I>>;
	// TODO remove ?
	fn write_collection<'a, DB, I>(db: &'a mut DB, collection: &'static[u8]) -> Option<CollectionMut<'a, Self, I>>;
}

pub struct Collection<'a, DB, Instance> {
	db: &'a DB,
	instance: &'a Instance,
}

pub struct CollectionMut<'a, DB, Instance> {
	db: &'a mut DB,
	instance: &'a Instance,
}

impl<'a, DB: SerializeDB, Instance: SerializeInstance> Collection<'a, DB, Instance> {
	pub fn read(&self, k: &[u8]) -> Option<Vec<u8>> {
		self.db.read(Instance::STATIC_COL, k)
	}
	pub fn iter(&self) -> DB::Iter {
		self.db.iter(Instance::STATIC_COL)
	}
}

impl<'a, DB: SerializeDB, Instance: SerializeInstance> CollectionMut<'a, DB, Instance> {
	pub fn write(&mut self, k: &[u8], v: &[u8]) {
		self.db.write(Instance::STATIC_COL, k, v)
	}
	pub fn remove(&mut self, k: &[u8]) {
		self.db.remove(Instance::STATIC_COL, k)
	}
	pub fn read(&self, k: &[u8]) -> Option<Vec<u8>> {
		self.db.read(Instance::STATIC_COL, k)
	}
	pub fn iter(&self) -> DB::Iter {
		self.db.iter(Instance::STATIC_COL)
	}
}

/// Serialize trait, when dynamic collection
/// are allowed.
pub trait DynSerializeDB: SerializeDB {

	/// Create a new collection and return its identifier.
	fn new_dyn_collection(&mut self) -> Vec<u8>;

	fn dyn_write(&mut self, c: &[u8], k: &[u8], v: &[u8]);
	fn dyn_remove(&mut self, c: &[u8], k: &[u8]);
	fn dyn_read(&self, c: &[u8], k: &[u8]) -> Option<Vec<u8>>;
	fn dyn_iter(&self, c: &[u8]) -> Self::Iter;

//	fn dyn_read_collection<'a, DB, I>(db: &'a DB, collection: &'static[u8]) -> Option<DynCollection<'a, Self, I>>;
//	fn dyn_write_collection<'a, DB, I>(db: &'a mut DB, collection: &'static[u8]) -> Option<DynCollectionMut<'a, Self, I>>;
}

impl<'a, DB: DynSerializeDB, Instance: DynSerializeInstance> Collection<'a, DB, Instance> {
	pub fn dyn_read(&self, k: &[u8]) -> Option<Vec<u8>> {
		if let Some(c) = self.instance.dyn_collection() {
			self.db.dyn_read(c, k)
		} else {
			self.db.read(Instance::STATIC_COL, k)
		}
	}
	pub fn dyn_iter(&self) -> DB::Iter {
		if let Some(c) = self.instance.dyn_collection() {
			self.db.dyn_iter(c)
		} else {
			self.db.iter(Instance::STATIC_COL)
		}
	}
}

impl<'a, DB: DynSerializeDB, Instance: DynSerializeInstance> CollectionMut<'a, DB, Instance> {
	pub fn dyn_write(&mut self, k: &[u8], v: &[u8]) {
		if let Some(c) = self.instance.dyn_collection() {
			self.db.dyn_write(c, k, v)
		} else {
			self.db.write(Instance::STATIC_COL, k, v)
		}
	}
	pub fn dyn_remove(&mut self, k: &[u8]) {
		if let Some(c) = self.instance.dyn_collection() {
			self.db.dyn_remove(c, k)
		} else {
			self.db.remove(Instance::STATIC_COL, k)
		}
	}
	pub fn dyn_read(&self, k: &[u8]) -> Option<Vec<u8>> {
		if let Some(c) = self.instance.dyn_collection() {
			self.db.dyn_read(c, k)
		} else {
			self.db.read(Instance::STATIC_COL, k)
		}
	}
	pub fn dyn_iter(&self) -> DB::Iter {
		if let Some(c) = self.instance.dyn_collection() {
			self.db.dyn_iter(c)
		} else {
			self.db.iter(Instance::STATIC_COL)
		}
	}
}

/// Info for serialize usage.
///
/// Static collection are using a know identifier that never change.
pub trait SerializeInstance: Default + Clone {
	/// If collection is static this contains its
	/// unique identifier.
	const STATIC_COL: &'static [u8];
}

impl SerializeInstance for () {
	const STATIC_COL: &'static [u8] = &[];
}

/// Dynamic collection can be change.
/// Static and dynamic collection are mutually exclusive, yet instance using both trait should run
/// dynamic first.
pub trait DynSerializeInstance: SerializeInstance {
	/// If collection is dynamic this returns the
	/// current collection unique identifier.
	fn dyn_collection(&self) -> Option<&[u8]>;
}

pub trait SerializeInstanceVariable: SerializeInstance {
	/// Location of the variable in its collection.
	const PATH: &'static [u8];
}

impl SerializeDB for () {
	const ACTIVE: bool = false;
	type Iter = crate::rstd::iter::Empty<(Vec<u8>, Vec<u8>)>;
	fn write(&mut self, _c: &[u8], _k: &[u8], _v: &[u8]) { }
	fn remove(&mut self, _c: &[u8], _k: &[u8]) { }
	fn read(&self, _c: &[u8], _k: &[u8]) -> Option<Vec<u8>> {
		None
	}
	fn iter(&self, _collection: &[u8]) -> Self::Iter {
		crate::rstd::iter::empty()
	}
	fn contains_collection(_collection: &[u8]) -> bool {
		false
	}
	fn read_collection<'a, DB, I>(_db: &'a DB, _collection: &'static[u8]) -> Option<Collection<'a, Self, I>> {
		None
	}
	fn write_collection<'a, DB, I>(_db: &'a mut DB, _collection: &'static[u8]) -> Option<CollectionMut<'a, Self, I>> {
		None
	}
}


use codec::{Codec, Encode, Decode};

#[derive(Debug, Clone)]
#[derive(Derivative)]
#[cfg_attr(any(test, feature = "test"), derivative(PartialEq, Eq))]
#[derivative(Default(bound="I: Default"))]
/// Lazy loading serialized map with cache.
/// Updates happens immediatelly.
pub struct SerializeMap<K: Ord, V, S, I> {
	inner: BTreeMap<K, Option<V>>,
	instance: I,
	#[derivative(PartialEq="ignore")]
	_ph: PhantomData<S>,
}

impl<'a, K: Ord, V, S, I> SerializeMap<K, V, S, I> {
	pub fn handle(&'a mut self, db: &'a mut S) -> SerializeMapHandle<'a, K, V, S, I> {
		SerializeMapHandle {
			cache: &mut self.inner,
			collection: CollectionMut { db, instance: &self.instance },
		}
	}
}

pub struct SerializeMapHandle<'a, K, V, S, I> {
	cache: &'a mut BTreeMap<K, Option<V>>,
	collection: CollectionMut<'a, S, I>,
}

/// Notice that entry map does not write imediatelly but on
/// drop or on `flush`.
pub struct EntryMap<'a, K, V, S, I>
	where
		K: Encode + Ord,
		V: Codec,
		S: SerializeDB,
		I: SerializeInstance,
{
	entry: crate::rstd::btree_map::OccupiedEntry<'a, K, Option<V>>,
	collection: CollectionMut<'a, S, I>,
	need_write: bool,
	is_fetch: bool,
}

impl<'a, K, V, S, I> SerializeMapHandle<'a, K, V, S, I>
	where
		K: Encode + Clone + Ord,
		V: Codec + Clone,
		S: SerializeDB,
		I: SerializeInstance,
{
	pub fn get(&mut self, k: &K) -> Option<&V> {
		if S::ACTIVE {
			let collection = &self.collection;
			self.cache.entry(k.clone())
				.or_insert_with(|| {
					let enc_k = k.encode();
					collection.read(&enc_k).and_then(|v| V::decode(&mut v.as_slice()).ok())
				}).as_ref()
		} else {
			self.cache.get(k).and_then(|r|r.as_ref())
		}
	}
	pub fn remove(&mut self, k: &K) -> Option<V> {
		if !S::ACTIVE {
			return self.cache.remove(k).flatten()
		}
		let mut value = match self.cache.get(k) {
			Some(None) => return None, // Delete is synch, nothing to do
			Some(Some(v)) => Some(v.clone()),
			None => None,
		};
		// We cache all deleted value
		self.cache.insert(k.clone(), None);
		let k = k.encode();
		if value.is_none() {
			// TODO see if it is the right api (we may skip this get)
			value = self.collection.read(&k).and_then(|v| V::decode(&mut v.as_slice()).ok());
		}
		self.collection.remove(k.as_slice());
		value
	}
	pub fn insert(&'a mut self, k: K, v: V) -> &V {
		self.collection.write(k.encode().as_slice(), v.encode().as_slice());
		let res = self.cache.entry(k)
			.and_modify(|old| *old = Some(v.clone()))
			.or_insert_with(|| Some(v));
		res.as_ref().expect("Init to some")
	}

	pub fn entry(&'a mut self, k: &K) -> EntryMap<'a, K, V, S, I> {
		let mut is_fetch = true;
		if !self.cache.contains_key(k) {
			is_fetch = false;
			self.cache.insert(k.clone(), None);
		}
		let entry = match self.cache.entry(k.clone()) {
			Entry::Occupied(o) => o,
			Entry::Vacant(..) => unreachable!("set above"),
		};
		EntryMap {
			entry,
			collection: CollectionMut {
				instance: &self.collection.instance,
				db: &mut self.collection.db,
			},
			need_write: false,
			is_fetch,
		}
	}
}

impl<'a, K, V, S, I> SerializeMapHandle<'a, K, V, S, I>
	where
		K: Codec + Clone + Ord,
		V: Codec + Clone,
		S: SerializeDB,
		I: SerializeInstance,
{
	pub fn iter(&'a self) -> SerializeMapIter<'a, K, V, S> {
		if !S::ACTIVE {
			SerializeMapIter::Cache(self.cache.iter())
		} else {
			SerializeMapIter::Collection(self.collection.iter())
		}
	}
}

pub enum SerializeMapIter<'a, K, V, S>
	where
		K: Codec + Ord + Clone,
		V: Codec + Clone,
		S: SerializeDB,
{
	Cache(crate::rstd::btree_map::Iter<'a, K, Option<V>>),
	Collection(S::Iter),
}

impl<'a, K, V, S> Iterator for SerializeMapIter<'a, K, V, S>
	where
		K: Codec + Ord + Clone,
		V: Codec + Clone,
		S: SerializeDB,
{
	type Item = (K, V);

	fn next(&mut self) -> Option<Self::Item> {
		match self {
			SerializeMapIter::Cache(i) => loop {
				match i.next() {
					Some((k, Some(v))) => return Some((k.clone(), v.clone())),
					Some((_k, None)) => (),
					None => return None,
				}
			},
			SerializeMapIter::Collection(i) => loop {
				match i.next() {
					Some((k, v)) => {
						match (K::decode(&mut k.as_slice()), V::decode(&mut v.as_slice())) {
							(Ok(k), Ok(v)) => return Some((k, v)),
							_ => (),
						}
					},
					None => return None,
				}
			},
		}
	}
}

// TODO add Eq check to avoid useless write (store the orig value and check on write only)
// (replacing is_fetch by value)
impl<'a, K, V, S, I> EntryMap<'a, K, V, S, I>
	where
		K: Encode + Clone + Ord,
		V: Codec + Clone,
		S: SerializeDB,
		I: SerializeInstance,
{
	pub fn key(&self) -> &K {
		self.entry.key()
	}

	pub fn and_modify<F>(mut self, f: impl FnOnce(&mut V)) -> Self {
		self.fetch();

		if let Some(v) = self.entry.get_mut() {
			self.need_write = true;
			f(v)
		}
		self
	}

	pub fn or_insert_with<F>(&mut self, value: impl FnOnce() -> V) -> &V {
		if self.entry.get().is_none() {
			*self.entry.get_mut() = Some(value());
			self.need_write = true;
		}
		self.entry.get().as_ref().expect("init above")
	}

	pub fn get(&mut self) -> Option<&V> {
		self.fetch();
		self.entry.get().as_ref()
	}

	pub fn set(&mut self, v: V) -> &V {
		self.need_write = true;
		*self.entry.get_mut() = Some(v);
		self.entry.get().as_ref().expect("init above")
	}

	pub fn clear(&mut self) {
		self.need_write = true;
		*self.entry.get_mut() = None;
		// we do not fetch
		self.is_fetch = true;
	}
}

impl<'a, K, V, S, I> EntryMap<'a, K, V, S, I>
	where
		K: Encode + Ord,
		V: Codec,
		S: SerializeDB,
		I: SerializeInstance,
{
	pub fn fetch(&mut self) {
		if S::ACTIVE && !self.is_fetch {
			let k = self.entry.key();
			let k = k.encode();
			let v = self.collection.read(k.as_slice())
				.and_then(|v| V::decode(&mut v.as_slice()).ok());
			*self.entry.get_mut() = v;
			self.need_write = false;
			self.is_fetch = true;
		}
	}

	pub fn flush(&mut self) {
		if S::ACTIVE {
			let k = self.entry.key();
			if self.need_write {
				match self.entry.get() {
					Some(v) => {
						let k = k.encode();
						let v = v.encode();
						self.collection.write(k.as_slice(), v.as_slice());
					},
					None => {
						let k = k.encode();
						self.collection.remove(k.as_slice());
					},
				}
				self.need_write = false;
			}
		}
	}
}

impl<'a, K, V, S, I> Drop for EntryMap<'a, K, V, S, I>
	where
		K: Encode + Ord,
		V: Codec,
		S: SerializeDB,
		I: SerializeInstance,
{
	fn drop(&mut self) {
		self.flush()
	}
}

/// Is db variable or default if undefined.
pub struct SerializeVariable<V, S, I> {
	// None indicate we did not fetch.
	inner: Option<V>,
	instance: I,
	_ph: PhantomData<S>,
}

pub struct SerializeVariableHandle<'a, V, S, I>
	where
		I: SerializeInstanceVariable,
		S: SerializeDB,
		V: Encode,
{
	inner: &'a mut V,
	collection: CollectionMut<'a, S, I>,
	need_write: bool,
}

impl<'a, V, S, I> SerializeVariable<V, S, I>
	where
		I: SerializeInstanceVariable,
		S: SerializeDB,
		V: Codec + Default,
{
	pub fn handle(&'a mut self, db: &'a mut S) -> SerializeVariableHandle<'a, V, S, I> {
		let collection = CollectionMut { db, instance: &self.instance };
		if self.inner.is_none() {
			self.inner = Some(collection.read(I::PATH).and_then(|v| V::decode(&mut v.as_slice()).ok()).unwrap_or_default());
		}
		SerializeVariableHandle {
			inner: self.inner.as_mut().expect("Init above"),
			collection,
			need_write: false,
		}
	}
}

impl<'a, V, S, I> SerializeVariableHandle<'a, V, S, I>
	where
		I: SerializeInstanceVariable,
		S: SerializeDB,
		V: Encode,
{
	pub fn get(&self) -> &V {
		&self.inner
	}
	pub fn set(&mut self, value: V) {
		*self.inner = value;
		self.need_write = true;
	}
	pub fn flush(&mut self) {
		let encoded = self.inner.encode();
		self.collection.write(I::PATH, encoded.as_slice());
		self.need_write = false;
	}
}

impl<'a, V, S, I> Drop for SerializeVariableHandle<'a, V, S, I>
	where
		I: SerializeInstanceVariable,
		S: SerializeDB,
		V: Encode,
{
	fn drop(&mut self) {
		self.flush()
	}
}
