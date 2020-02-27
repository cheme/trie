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

use core::marker::PhantomData;

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

pub trait StateDB<K, V>: StateDBRef<K, V> {
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
	fn emplace(&mut self, key: K, value: V, at: &Self::S);

	/// Remove a datum previously inserted. Insertions can be "owed" such that the
	/// same number of `insert()`s may happen without the data being eventually
	/// being inserted into the DB. It can be "owed" more than once.
	/// The caller should ensure that a key only corresponds to one value.
	fn remove(&mut self, key: &K, at: &Self::S);
	fn gc(&mut self, gc: Self::GC);
	fn migrate(&mut self, mig: Self::Migrate);
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

/// Management maps a state with a db state.
pub trait ManagementRef<H> {
	/// attached db state
	type S;
	/// attached db gc strategy.
	type GC;
	type Migrate;
	fn get_db_state(&self, state: &H) -> Option<Self::S>;
	/// returns optional to avoid holding lock of do nothing GC.
	fn get_gc(&self) -> Option<Self::GC>;
}

pub trait Management<H>: ManagementRef<H> + Sized {
	fn init() -> (Self, Self::S);
	fn latest_state(&self) -> Self::S;
	fn reverse_lookup(&self, state: &Self::S) -> H;
	fn drop_from(&mut self, state: &Self::S);
	fn try_drop_from(&mut self, state: &H) {
		self.get_db_state(state).map(|at| self.drop_from(&at));
	}
	fn drop_before(&mut self, state: &Self::S);
	fn try_drop_before(&mut self, state: &H) {
		self.get_db_state(state).map(|at| self.drop_before(&at));
	}
	fn drop_at(&mut self, state: &Self::S);
	fn try_drop_at(&mut self, state: &H) {
		self.get_db_state(state).map(|at| self.drop_at(&at));
	}

	/// report a gc did run successfully, this only update gc
	/// information to not include these info.
	fn applied_gc(&mut self, gc: Self::GC);
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
pub trait ForkableManagement<H>: ManagementRef<H> {
	fn register_external_state(&mut self, state: H, at: &Self::S) -> Self::S;

	fn try_register_external_state(&mut self, state: H, at: &H) -> Option<Self::S> {
		self.get_db_state(at).map(|at| self.register_external_state(state, &at))
	}
}

pub trait LinearManagement<H>: ManagementRef<H> {
	fn append_external_state(&mut self, state: H) -> Self::S;

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
