// Copyright 2020, 2020 Parity Technologies
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

//! Linear historied data.

#[cfg(not(feature = "std"))]
use crate::rstd::{vec::Vec, vec};
use crate::rstd::marker::PhantomData;
use crate::{StateDBRef, UpdateResult, InMemoryStateDBRef, StateDB, ManagementRef,
	Management, Migrate, LinearManagement};
use hash_db::{PlainDB, PlainDBRef};
use crate::Latest;

pub mod linear;
pub mod tree_management;
pub mod tree;
pub mod encoded_array;

/// Trait for historied value
pub trait ValueRef<V> {
	/// State to query for this value.
	type S;

	/// Get value at this state.
	fn get(&self, at: &Self::S) -> Option<V>;

	/// Check if a value exists at this state.
	fn contains(&self, at: &Self::S) -> bool;

	/// Check if this is empty.
	fn is_empty(&self) -> bool;
}

pub trait InMemoryValueRef<V>: ValueRef<V> {
	/// Get reference to the value at this state.
	fn get_ref(&self, at: &Self::S) -> Option<&V>;
}

/// Trait for historied value.
pub trait Value<V>: ValueRef<V> {
	/// State to use here.
	/// We use a different state than
	/// for the ref as it can use different
	/// constraints.
	type SE: StateIndex<Self::Index>;

	/// Index a single history item.
	type Index;
	//type SE = Self::S; TODO next nightly and future stable should accept it
	/// GC strategy that can be applied.
	/// GC can be run in parallel, it does not
	/// make query incompatible.
	type GC;
	/// Like gc but operation require a lock on the db
	/// and all pending state are invalidated.
	type Migrate;

	/// Initiate a new value.
	fn new(value: V, at: &Self::SE) -> Self;

	/// Insert or update a value.
	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()>;

	/// Discard history at.
	fn discard(&mut self, at: &Self::SE) -> UpdateResult<Option<V>>;

	fn gc(&mut self, gc: &Self::GC) -> UpdateResult<()>;

	fn is_in_migrate(index: &Self::Index, gc: &Self::Migrate) -> bool;

	fn migrate(&mut self, mig: &Self::Migrate) -> UpdateResult<()>;
}

pub trait InMemoryValue<V>: Value<V> {
	/// Get latest value, can apply updates.
	fn get_mut(&mut self, at: &Self::SE) -> Option<&mut V>;

	/// Similar to value set but returning a pointer on replaced or deleted value.
	/// If the value is change but history is kept (new state), no pointer is returned.
	fn set_mut(&mut self, value: V, at: &Self::SE) -> UpdateResult<Option<V>>;
}

/// An entry at a given history index.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct HistoriedValue<V, S> {
	/// The stored value.
	pub value: V,
	/// The state this value belongs to.
	pub state: S,
}

impl<V, S> From<(V, S)> for HistoriedValue<V, S> {
	fn from(input: (V, S)) -> HistoriedValue<V, S> {
		HistoriedValue { value: input.0, state: input.1 }
	}
}

/// Implementation for plain db.
pub struct BTreeMap<K, V, H>(crate::rstd::BTreeMap<K, H>, PhantomData<V>);

impl<K: Ord, V: Clone, H: ValueRef<V>> StateDBRef<K, V> for BTreeMap<K, V, H> {
	type S = H::S;

	fn get(&self, key: &K, at: &Self::S) -> Option<V> {
		self.0.get(key)
			.and_then(|h| h.get(at))
	}

	fn contains(&self, key: &K, at: &Self::S) -> bool {
		self.0.get(key)
			.map(|h| h.contains(at))
			.unwrap_or(false)
	}
}

// note that the constraint on state db ref for the associated type is bad (forces V as clonable).
impl<K: Ord, V: Clone, H: InMemoryValueRef<V>> InMemoryStateDBRef<K, V> for BTreeMap<K, V, H> {
	fn get_ref(&self, key: &K, at: &Self::S) -> Option<&V> {
		self.0.get(key)
			.and_then(|h| h.get_ref(at))
	}
}

impl<K: Ord + Clone, V: Clone + Eq, H: Value<V>> StateDB<K, V> for BTreeMap<K, V, H> {
	// see inmemory
	type SE = H::SE;
	// see inmemory
	type GC = H::GC;
	// see inmemory
	type Migrate = H::Migrate;

	fn emplace(&mut self, key: K, value: V, at: &Self::SE) {
		if let Some(hist) = self.0.get_mut(&key) {
			hist.set(value, at);
		} else {
			self.0.insert(key, H::new(value, at));
		}
	}

	fn remove(&mut self, key: &K, at: &Self::SE) {
		match self.0.get_mut(&key).map(|h| h.discard(at)) {
			Some(UpdateResult::Cleared(_)) => (),
			_ => return,
		}
		self.0.remove(&key);
	}

	fn gc(&mut self, gc: &Self::GC) {
		// retain for btreemap missing here.
		let mut to_remove = Vec::new();
		for (key, h) in self.0.iter_mut() {
			match h.gc(gc) {
				UpdateResult::Cleared(_) => (),
				_ => break,
			}
			to_remove.push(key.clone());
		}
		for k in to_remove {
			self.0.remove(&k);
		}
	}

	fn migrate(&mut self, mig: &Self::Migrate) {
		// retain for btreemap missing here.
		let mut to_remove = Vec::new();
		for (key, h) in self.0.iter_mut() {
			match h.migrate(mig) {
				UpdateResult::Cleared(_) => (),
				_ => break,
			}
			to_remove.push(key.clone());
		}
		for k in to_remove {
			self.0.remove(&k);
		}
	}
}

/// Implementation for plain db.
pub struct PlainDBState<K, DB, H, S> {
	db: DB,
	touched_keys: crate::rstd::BTreeMap<S, Vec<K>>, // TODO change that by a journal trait!!
	_ph: PhantomData<H>,
}

impl<K, V: Clone, H: ValueRef<V>, DB: PlainDBRef<K, H>, S> StateDBRef<K, V> for PlainDBState<K, DB, H, S> {
	type S = H::S;

	fn get(&self, key: &K, at: &Self::S) -> Option<V> {
		self.db.get(key)
			.and_then(|h| h.get(at))
	}

	fn contains(&self, key: &K, at: &Self::S) -> bool {
		self.db.get(key)
			.map(|h| h.contains(at))
			.unwrap_or(false)
	}
}

impl<
	K: Ord + Clone,
	V: Clone + Eq,
	H: Value<V>,
	DB: PlainDBRef<K, H> + PlainDB<K, H>,
> StateDB<K, V> for PlainDBState<K, DB, H, H::Index>
	where
			H::Index: Clone + Ord,
{
	// see inmemory
	type SE = H::SE;
	// see inmemory
	type GC = H::GC;
	// see inmemory
	type Migrate = H::Migrate;

	fn emplace(&mut self, key: K, value: V, at: &Self::SE) {
		if let Some(mut hist) = <DB as PlainDB<_, _>>::get(&self.db, &key) {
			match hist.set(value, at) {
				UpdateResult::Changed(_) => self.db.emplace(key.clone(), hist),
				UpdateResult::Cleared(_) => self.db.remove(&key),
				UpdateResult::Unchanged => return,
			}
		} else {
			self.db.emplace(key.clone(), H::new(value, at));
		}
		self.touched_keys.entry(at.index()).or_default().push(key);
	}

	fn remove(&mut self, key: &K, at: &Self::SE) {
		if let Some(mut hist) = <DB as PlainDB<_, _>>::get(&self.db, &key) {
			match hist.discard(at) {
				UpdateResult::Changed(_) => self.db.emplace(key.clone(), hist),
				UpdateResult::Cleared(_) => self.db.remove(&key),
				UpdateResult::Unchanged => return,
			}
		}
		self.touched_keys.entry(at.index()).or_default().push(key.clone());
	}

	fn gc(&mut self, gc: &Self::GC) {
		let mut keys: crate::rstd::BTreeSet<_> = Default::default();
		for touched in self.touched_keys.values() {
			for key in touched.iter() {
				keys.insert(key.clone());
			}
		}
		for key in keys {
			if let Some(mut hist) = <DB as PlainDB<_, _>>::get(&self.db, &key) {
				match hist.gc(gc) {
					UpdateResult::Changed(_) => self.db.emplace(key, hist),
					UpdateResult::Cleared(_) => self.db.remove(&key),
					UpdateResult::Unchanged => break,
				}
			}
		}
	}

	fn migrate(&mut self, mig: &Self::Migrate) {
		// TODO this is from old gc but seems ok (as long as touched is complete).
		// retain for btreemap missing here.
		let mut states = Vec::new();
		// TODO do we really want this error prone prefiltering??
		for touched in self.touched_keys.keys() {
			if H::is_in_migrate(touched, mig) {
				states.push(touched.clone());
			}
		}
		let mut keys: crate::rstd::BTreeSet<_> = Default::default();
		for state in states {
			if let Some(touched) = self.touched_keys.remove(&state) {
				for k in touched {
					keys.insert(k);
				}
			}
		}
		self.touched_keys.clear();
		for key in keys {
			if let Some(mut hist) = <DB as PlainDB<_, _>>::get(&self.db, &key) {
				match hist.migrate(mig) {
					UpdateResult::Changed(_) => self.db.emplace(key, hist),
					UpdateResult::Cleared(_) => self.db.remove(&key),
					UpdateResult::Unchanged => break,
				}
			}
		}
	}
}

/// Associate an index for a given state reference
/// TODO this should be removable or rename (I is
/// individual item index when state index is larger).
pub trait StateIndex<I> {
	fn index(&self) -> I;
	fn index_ref(&self) -> &I;
}

impl<S: Clone> StateIndex<S> for Latest<S> {
	fn index(&self) -> S {
		self.latest().clone()
	}
	fn index_ref(&self) -> &S {
		self.latest()
	}
}
