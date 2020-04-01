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

//! Test implementation that favor minimal non scalable in memory
//! implementation.

use crate::*;
use std::collections::HashMap;
use std::hash::Hash;

/// The main test Db.
pub struct Db<K, V> {
	db: Vec<HashMap<K, V>>,
	management: Mgmt,
}

impl<K: Eq + Hash, V> Db<K, V> {
	pub fn init() -> (Self, Query) {
		let (management, query) = Mgmt::init();
		(Db {
			db: vec![Default::default()],
			management,
		}, query)
	}
}

/// state index.
type StateIndex = usize;

/// State Input (aka hash).
struct StateInput(usize);

/// The mananagement part.
struct Mgmt {
	/// contain a pointer to parent state for tree.
	/// Representation is simply index of the array.
	pub states: Vec<StateIndex>,
	pub latest_state: Latest<StateIndex>,
	// TODO if delete switch to (Option<StateIndex>, bool)
	// where bool indicate if can append + a is last.
}

impl Mgmt {
	fn is_latest(&self, ix: &StateIndex) -> bool {
		!self.states.contains(&ix)
	}
	fn contains(&self, ix: &StateInput) -> bool {
		self.states.len() > ix.0
	}

	fn get_state(&self, state: &StateInput) -> Option<StateIndex> {
		if self.contains(state) {
			Some(state.0)
		} else {
			None
		}
	}
}

/// Query path, ordered by latest state first.
type Query = Vec<StateIndex>;

impl<K: Hash + Eq, V: Clone> StateDBRef<K, V> for Db<K, V> {
	type S = Query;

	fn get(&self, key: &K, at: &Self::S) -> Option<V> {
		self.get_ref(key, at).cloned()
	}

	fn contains(&self, key: &K, at: &Self::S) -> bool {
		self.get(key, at).is_some()
	}
}

impl<K: Hash + Eq, V: Clone> InMemoryStateDBRef<K, V> for Db<K, V> {
	fn get_ref(&self, key: &K, at: &Self::S) -> Option<&V> {
		for s in at.iter() {
			if let Some(v) = self.db.get(*s).and_then(|h| h.get(key)) {
				return Some(v)
			}
		}
		None
	}
}

impl<K: Hash + Eq, V: Clone> StateDB<K, V> for Db<K, V> {
	type SE = Latest<StateIndex>;
	type GC = ();
	type Migrate = ();

	fn emplace(&mut self, key: K, value: V, at: &Self::SE) {
		debug_assert!(self.management.is_latest(at.latest()));
		self.db.get_mut(at.0)
			.expect("no removal and no random SE")
			.insert(key, value);
	}

	fn remove(&mut self, key: &K, at: &Self::SE) {
		debug_assert!(self.management.is_latest(at.latest()));
		self.db.get_mut(at.0)
			.expect("no removal and no random SE")
			.remove(key);
	}

	fn gc(&mut self, _gc: &Self::GC) { }

	fn migrate(&mut self, _mig: &Self::Migrate) { }
}

impl ManagementRef<StateInput> for Mgmt {
	type S = Query;
	type GC = ();
	type Migrate = ();
	fn get_db_state(&self, state: &StateInput) -> Option<Self::S> {
		if let Some(mut ix) = self.get_state(state) {
			let mut query = vec![ix];
			loop {
				let next = self.states[ix];
				if next == ix {
					break;
				} else {
					query.push(next);
					ix = next;
				}
			}
			Some(query)
		} else {
			None
		}
	}

	fn get_gc(&self) -> Option<Ref<Self::GC>> {
		None
	}
}

impl Management<StateInput> for Mgmt {
	type SE = Latest<StateIndex>;

	fn init() -> (Self, Self::S) {
		// 0 is defined
		(Mgmt {
			states: vec![0],
			latest_state: Latest::unchecked_latest(0),
		}, vec![0])
	}

	fn get_db_state_mut(&self, state: &StateInput) -> Option<Self::SE> {
		if let Some(s) = self.get_state(state) {
			if self.is_latest(&s) {
				return Some(Latest::unchecked_latest(s))
			}
		}
		None
	}

	fn latest_state(&self) -> Self::SE {
		self.latest_state.clone()
	}

	fn reverse_lookup(&self, state: &Self::S) -> Option<StateInput> {
		if let Some(state) = state.first() {
			if &self.states.len() > state {
				Some(StateInput(*state))
			} else {
				None
			}
		} else {
			None
		}
	}

	fn get_migrate(self) -> Migrate<StateInput, Self> {
		Migrate::capture(self)
	}

	fn applied_migrate(&mut self) { }
}

impl ForkableManagement<StateInput> for Mgmt {
	type SF = StateIndex;

	fn get_db_state_for_fork(&self, state: &StateInput) -> Option<Self::SF> {
		self.get_state(state)
	}

	fn latest_state_fork(&self) -> Self::SF {
		self.latest_state.latest().clone()
	}

	fn append_external_state(&mut self, state: StateInput, at: &Self::SF) -> Option<Self::S> {
		debug_assert!(state.0 == self.states.len());
		self.states.push(*at);
		self.latest_state = Latest::unchecked_latest(self.states.len() - 1);

		self.get_db_state(&state)
	}

	/// Warning this recurse over children and can be slow for some
	/// implementations.
	fn drop_state(&mut self, state: &Self::SF, return_dropped: bool) -> Option<Vec<StateInput>> {
		unimplemented!("TODO support for drop");
	}
}
