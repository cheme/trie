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

#![cfg_attr(not(feature = "std"), no_std)]

pub trait StateDB<K, V> {
	/// State for this db.
	type S;

	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &K, at: &S) -> Option<V>;

	/// Check for the existence of a hash-key.
	fn contains(&self, key: &K, at: &S) -> bool;

	/// Insert a datum item into the DB. Insertions are counted and the equivalent
	/// number of `remove()`s must be performed before the data is considered dead.
	/// The caller should ensure that a key only corresponds to one value.
	fn emplace(&mut self, key: K, value: V, at: &S);

	/// Remove a datum previously inserted. Insertions can be "owed" such that the
	/// same number of `insert()`s may happen without the data being eventually
	/// being inserted into the DB. It can be "owed" more than once.
	/// The caller should ensure that a key only corresponds to one value.
	fn remove(&mut self, key: &K, at: &S);
}

/// Trait for immutable reference of PlainDB.
pub trait StateDBRef<K, V> {
	/// State for this db.
	type S;

	/// Look up a given hash into the bytes that hash to it, returning None if the
	/// hash is not known.
	fn get(&self, key: &K, at: &S) -> Option<V>;

	/// Check for the existance of a hash-key.
	fn contains(&self, key: &K, at: &S) -> bool;
}

pub trait ManagementRef<H> {
	type S;
	fn get_db_state(&self, state: H) -> Option<S>
}

/// This trait is for mapping a given state to the DB opaque inner state.
pub trait ForkableManagement<H> {
	/// State representation corresponding to external state H.
	type S;

	fn get_db_state(&self, state: H)
	fn register_external_state
}

pub trait LinearManagement<H> {
	/// State representation corresponding to external state H.
	type S;

	fn get_db_state(&self, state: H) -> Option<S>
	fn new_initial_state(&mut self) -> S;
	fn append_external_state(&mut self, state: H) -> S;

	// cannot be empty: if at initial state we return initial
	// state and initialise with a new initial state.
	fn drop_last_state(&mut self) -> S;
}

/// Latest from fork only, this is for use case of aggregable
/// data cache: to store the aggregate cache.
/// NOTE aggregable data cache is a cache that reply to locality
/// (a byte trie with locks that invalidate cache when set storage is call).
/// get_aggregate(aggregate_key)-> option<StructAggregate>
/// set_aggregate(aggregate_key, struct aggregate, [(child_info, lockprefixes)]).
pub trait ForkableHeadManagement<H> {
	/// State representation corresponding to external state H.
	type S;

	fn get_db_state(&self, state: H)
	fn register_external_state
}



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
