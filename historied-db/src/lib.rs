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

//! Database for key value with history.


// TODO change all ref to S to a borrow similar to map (most
// of the type S is copied so using reference looks pointless).

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "std")]
mod rstd {
	pub use std::{borrow, boxed, cmp, convert, fmt, hash, iter, marker, mem, ops, rc, result, vec};
	pub use std::collections::VecDeque;
	pub use std::collections::{BTreeMap, BTreeSet};
	pub use std::error::Error;
}

#[cfg(not(feature = "std"))]
mod rstd {
	pub use core::{borrow, convert, cmp, iter, fmt, hash, marker, mem, ops, result};
	pub use alloc::{boxed, rc, vec};
	pub use alloc::collections::VecDeque;
	pub use alloc::collections::BTreeMap;
	pub trait Error {}
	impl<T> Error for T {}
}

use core::marker::PhantomData;

/// Implementation of historied-db traits
/// using historied values
pub mod historied;

#[cfg_attr(test, derive(PartialEq, Debug))]
///  result to be able to proceed
/// with further update if the value needs it.
pub enum UpdateResult<T> {
	Unchanged,
	Changed(T),
	Cleared(T),
}

impl<T> UpdateResult<T> {

	pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> UpdateResult<U> {
		match self {
			UpdateResult::Unchanged => UpdateResult::Unchanged,
			UpdateResult::Changed(v) => UpdateResult::Changed(f(v)),
			UpdateResult::Cleared(v) => UpdateResult::Cleared(f(v)),
		}
	}
}

/// Trait for immutable reference of PlainDB.
pub trait StateDBRef<K, V> {
	/// State for this db.
	type S;

	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &K, at: &Self::S) -> Option<V>;

	/// Check for the existance of a hash-key.
	fn contains(&self, key: &K, at: &Self::S) -> bool;
}

/// Variant of `StateDBRef` to return value without copy.
pub trait InMemoryStateDBRef<K, V>: StateDBRef<K, V> {
	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get_ref(&self, key: &K, at: &Self::S) -> Option<&V>;
}

pub trait StateDB<K, V>: StateDBRef<K, V> {
		// TODO associated type from Value??
	/// State to use here.
	/// We use a different state than
	/// for the ref as it can use different
	/// constraints.
	type SE;
	/// GC strategy that can be applied.
	/// GC can be run in parallel, it does not
	/// make query incompatible.
	type GC;
	/// Like gc but operation require a lock on the db
	/// and all pending state are invalidated.
	type Migrate;
	/// Insert a datum item into the DB. Insertions are counted and the equivalent
	/// number of `remove()`s must be performed before the data is considered dead.
	/// The caller should ensure that a key only corresponds to one value.
	fn emplace(&mut self, key: K, value: V, at: &Self::SE);

	/// Remove a datum previously inserted. Insertions can be "owed" such that the
	/// same number of `insert()`s may happen without the data being eventually
	/// being inserted into the DB. It can be "owed" more than once.
	/// The caller should ensure that a key only corresponds to one value.
	fn remove(&mut self, key: &K, at: &Self::SE);
	fn gc(&mut self, gc: &Self::GC);
	fn migrate(&mut self, mig: &Self::Migrate);
}

pub struct Migrate<H, M>(M, PhantomData<H>);

impl<H, M: Management<H>> Migrate<H, M> {
	pub fn capture(m: M) -> Self {
		Migrate(m, PhantomData)
	}
	pub fn applied_migrate(mut self) -> M {
		self.0.applied_migrate();
		self.0
	}
}

pub enum Ref<'a, V> {
	Borrowed(&'a V),
	Owned(V),
}

impl<'a, V> AsRef<V> for Ref<'a, V> {
	fn as_ref(&self) -> &V {
		match self {
			Ref::Borrowed(v) => v,
			Ref::Owned(v) => &v,
		}
	}
}

/// Management maps a state with a db state.
pub trait ManagementRef<H> {
	/// attached db state
	type S;
	/// attached db gc strategy.
	type GC;
	type Migrate;
	fn get_db_state(&self, state: &H) -> Option<Self::S>;
	/// returns optional to avoid holding lock of do nothing GC.
	fn get_gc(&self) -> Option<Ref<Self::GC>>;
}

pub trait Management<H>: ManagementRef<H> + Sized {
	/// attached db state for actual modification
	type SE;
	fn init() -> (Self, Self::S);

	fn get_db_state_mut(&self, state: &H) -> Option<Self::SE>;

	fn latest_state(&self) -> Self::SE;

	fn reverse_lookup(&self, state: &Self::S) -> Option<H>;

	/// see migrate. When running thes making a backup of this management
	/// state is usually a good idea (this method does not manage
	/// backup or rollback).
	fn get_migrate(self) -> Migrate<H, Self>;

	/// report a migration did run successfully, will update management state
	/// accordingly.
	/// All previously fetch states are unvalid.
	/// There is no type constraint of this, because migration is a specific
	/// case the general type should not be complexified.
	/// TODO see if Pin could do something for us.
	fn applied_migrate(&mut self);
}

/// This trait is for mapping a given state to the DB opaque inner state.
pub trait ForkableManagement<H>: Management<H> {
	fn append_external_state(&mut self, state: H, at: &Self::SE) -> Option<Self::S>;

	fn try_append_external_state(&mut self, state: H, at: &H) -> Option<Self::S>;
}

pub trait LinearManagement<H>: ManagementRef<H> {
	fn append_external_state(&mut self, state: H) -> Option<Self::S>;

	// cannot be empty: if at initial state we return initial
	// state and initialise with a new initial state.
	fn drop_last_state(&mut self) -> Self::S;
}

/// Latest from fork only, this is for use case of aggregable
/// data cache: to store the aggregate cache.
/// (it only record a single state per fork!! but still need to resolve
/// if new fork is needed so hold almost as much info as a forkable management).
/// NOTE aggregable data cache is a cache that reply to locality
/// (a byte trie with locks that invalidate cache when set storage is call).
/// get_aggregate(aggregate_key)-> option<StructAggregate>
/// set_aggregate(aggregate_key, struct aggregate, [(child_info, lockprefixes)]).
pub trait ForkableHeadManagement<H>: ManagementRef<H> {
	fn register_external_state_head(&mut self, state: H, at: &Self::S) -> Self::S;
	fn try_register_external_state_head(&mut self, state: H, at: &H) -> Option<Self::S> {
		self.get_db_state(at).map(|at| self.register_external_state_head(state, &at))
	}
}

/// Hybrid that apply linear over a defined fork (usually the root one).
/// So a linear substate is short for these fork.
pub trait CompositeRef<H>: ManagementRef<H> {
	// linear state TODO rewrite as sub state
	type LS;
	// here also report an error non linear state in case it is defined. 
	fn get_db_linear_state(&self, state: &H) -> Result<Self::LS, Option<Self::S>>;
}

pub trait CompositeManagement<H>: Management<H> {
	// join existing linear fork with this state
	// for a canonicalisation it therefore apply before
	// this state
	fn apply_linear(&mut self, at: &Self::S);
	fn try_apply_linear(&mut self, at: &H) {
		self.get_db_state(at).map(|at| self.apply_linear(&at));
	}
}


/*
enum DualState<S1, S2> {
	State1(S1),
	State2(S2),
}

enum DualManagement<'a, M1, M2> {
	Management1(&M1),
	Management2(&M2),
}


/// composite repsesentation, implements a switch between two
/// representation.
pub trait CompositeManagement<H> {
	/// State representation corresponding to external state H.
	type S1;
	type S2;
	type Management1<H>;
	type Management2<H>;

	fn get_db_state(&self, H) -> DualState<Self::S1, Self::S2>;
	fn get_management(&mut self, DualState) -> DualManagement<Self::M1, Self::M2>;
}
*/


// Additional trait that should not be in this crate (deals with collections).

pub trait MultipleDB {
	type Handle;
	fn set_collection(&mut self, h: Self::Handle);
	fn current_collection(&self) -> Self::Handle;
}

pub struct Collection {
	// static handle
	pub top: &'static [u8],
	// dynamic handle
	pub child: Vec<u8>,
}

pub trait ManagedDB {
	type Collection;
	type Handle;
	fn get_collection(&self, collection: &Self::Collection) -> Option<Self::Handle>;
	fn new_collection(&mut self, collection: &Self::Collection) -> Self::Handle;
	fn delete_collection(&mut self, collection: &Self::Collection);
}
