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

//! Linear historied data historied db implementations.
//!
//! Current implementation is limited to a simple array indexing
//! with modification at the end only.

use super::{HistoriedValue, ValueRef, Value, InMemoryValueRef, InMemoryValue};
use crate::{StateDBRef, UpdateResult, InMemoryStateDBRef, StateDB, ManagementRef,
	Management, Migrate, LinearManagement};
use hash_db::{PlainDB, PlainDBRef};
use crate::rstd::marker::PhantomData;
use crate::rstd::convert::{TryFrom, TryInto};
use crate::rstd::ops::{AddAssign, SubAssign, Range};
use crate::rstd::mem::replace;

/// For in memory implementation we expect the state to be `Into<usize>` and
/// `From<usize>` and will not manage failure when converting.
const USIZE_CONVERT: &'static str = "A in memory vec is bounded by usize limit.";

/// Basic usage case should be integers and byte representation, but
/// only integer should really be use.
///
/// This state is a simple ordered sequence.
pub trait LinearState:
	Default
	+ Clone
	+ Ord
	+ PartialOrd
	+ TryFrom<usize>
	+ AddAssign<usize> // TODO can remove ??
	+ PartialEq<usize>
{
	// stored state and query state are
	// the same for linear state.
	fn exists(&self, at: &Self) -> bool {
		self <= at
	}
	fn register_new(latest: &mut Latest<Self>) {
		latest.0 += 1;
	}
}

impl<S> LinearState for S where S:
	Default
	+ Clone
	+ Ord
	+ PartialOrd
	+ TryFrom<usize>
	+ AddAssign<usize>
	+ PartialEq<usize>
{ }

/// This is a rather simple way of managing state, as state should not be
/// invalidated at all (can be change at latest state, also drop but not at 
/// random state).
///
/// Note that it is only informational and does not guaranty the state
/// is the latest.
#[derive(Clone)]
pub struct Latest<S>(S);

impl<S> Latest<S> {
	/// This is only to be use by a `Management` or
	/// a context where the state can be proven as
	/// being the latest.
	pub(crate) fn unchecked_latest(s: S) -> Self {
		Latest(s)
	}
	pub(crate) fn latest(&self) -> &S {
		&self.0
	}
}

/// Size of preallocated history per element.
/// Currently at two for committed and prospective only.
/// It means that using transaction in a module got a direct allocation cost.
const ALLOCATED_HISTORY: usize = 2;

/// Array like buffer for in memory storage.
/// By in memory we expect that this will
/// not required persistence and is not serialized.
pub struct MemoryOnly<V, S>(smallvec::SmallVec<[HistoriedValue<V, S>; ALLOCATED_HISTORY]>);

impl<V, S: Clone> MemoryOnly<V, S> {
	pub fn new(value: V, state: &Latest<S>) -> Self {
		let mut v = smallvec::SmallVec::default();
		let state = state.latest().clone();
		v.push(HistoriedValue{ value, state });
		MemoryOnly(v)
	}
	pub fn remove_start(&mut self, split_off: usize) {
		if self.0.spilled() {
			let new = replace(&mut self.0, Default::default());
			self.0 = smallvec::SmallVec::from_vec(new.into_vec().split_off(split_off));
		} else {
			for i in 0..split_off {
				self.0.remove(i);
			}
		}
	}
}

impl<V: Clone, S: LinearState> ValueRef<V> for MemoryOnly<V, S> {
	type S = S;

	fn get(&self, at: &Self::S) -> Option<V> {
		self.get_ref(at).map(|v| v.clone())
	}

	fn contains(&self, at: &Self::S) -> bool {
		self.get_ref(at).is_some()
	}

	fn is_empty(&self) -> bool {
		self.0.is_empty()
	}
}

impl<V: Clone, S: LinearState> InMemoryValueRef<V> for MemoryOnly<V, S> {
	fn get_ref(&self, at: &Self::S) -> Option<&V> {
		let mut index = self.0.len();
		if index == 0 {
			return None;
		}
		while index > 0 {
			index -= 1;
			if let Some(HistoriedValue { value, state }) = self.0.get(index) {
				if at.exists(state) {
					return Some(value);
				}
			}
		}
		None
	}
}

//impl<V: Clone, S: LinearState, Q: LinearStateLatest<S>> Value<V> for MemoryOnly<V, S> {
impl<V: Clone + Eq, S: LinearState + SubAssign<S>> Value<V> for MemoryOnly<V, S> {
	type SE = Latest<S>;
	/// Removing existing state before.
	/// Optionally a skipable value (for
	/// history containing deletion that is deletion
	/// as a deletion on empty value can be remove).
	type GC = (S, Option<V>);
	/// Migrate will act as GC but also align state to 0.
	/// First index being the number for start state that
	/// will be removed after migration.
	type Migrate = (S, Self::GC);

	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()> {
		let at = at.latest();
		if let Some(last) = self.0.last() {
			// TODO this is rather unsafe: we expact that
			// when changing value we use a state that is
			// the latest from the state management.
			// Their could be ways to enforce that, but nothing
			// good at this point.
			debug_assert!(&last.state <= at); 
			if at == &last.state {
				if last.value == value {
					return UpdateResult::Unchanged;
				}
				self.0.pop();
			}
		}
		self.0.push(HistoriedValue {value, state: at.clone()});
		UpdateResult::Changed(())
	}

	// TODO not sure discard is of any use (revert is most likely
	// using some migrate as it breaks expectations).
	fn discard(&mut self, at: &Self::SE) -> UpdateResult<Option<V>> {
		let at = at.latest();
		if let Some(last) = self.0.last() {
			debug_assert!(&last.state <= at); 
			if at == &last.state {
				if self.0.len() == 1 {
					return UpdateResult::Cleared(self.0.pop().map(|v| v.value));
				} else {
					return UpdateResult::Changed(self.0.pop().map(|v| v.value));
				}
			}
		}
		UpdateResult::Unchanged
	}

	fn gc(&mut self, (start_treshold, start_void): &Self::GC) -> UpdateResult<()> {
		let mut index = 0;
		loop {
			if let Some(HistoriedValue{ value: _, state }) = self.0.get(index) {
				if state >= &start_treshold {
					index = index.saturating_sub(1);
					break;
				}
			} else {
				index = index.saturating_sub(1);
				break;
			}
			index += 1;
		}
		if let Some(start_void) = start_void.as_ref() {
			while let Some(HistoriedValue{ value, state: _ }) = self.0.get(index) {
				if value != start_void {
					break;
				}
				index += 1;
			}
		}
		if index == 0 {
			return UpdateResult::Unchanged
		}
		if index == self.0.len() {
			self.0.clear();
			return UpdateResult::Cleared(());
		}
		self.remove_start(index);
		UpdateResult::Changed(())
	}

	fn migrate(&mut self, (mig, gc): &Self::Migrate) -> UpdateResult<()> {
		let res = self.gc(gc);
		if self.0.len() > 0 {
			for h in self.0.iter_mut() {
				if &h.state > mig {
					h.state -= mig.clone();
				} else {
					h.state = Default::default();
				}
			}
			UpdateResult::Changed(())
		} else {
			res
		}
	}
}


impl<V: Clone + Eq, S: LinearState + SubAssign<S>> InMemoryValue<V> for MemoryOnly<V, S> {
	fn get_mut(&mut self, at: &Self::S) -> Option<&mut V> {
		let mut index = self.0.len();
		if index == 0 {
			return None;
		}
		while index > 0 {
			index -= 1;
			if let Some(HistoriedValue { value: _, state }) = self.0.get(index) {
				if at.exists(state) {
					return self.0.get_mut(index).map(|v| &mut v.value)
				}
			}
		}
		None
	}
}

/// Implementation for plain db.
pub struct PlainDBState<K, DB, S> {
	db: DB,
	touched_keys: crate::rstd::BTreeMap<S, Vec<K>>, // TODO change that by a journal trait!!
	_ph: PhantomData<S>,
}

impl<K, V: Clone, S: LinearState, DB: PlainDBRef<K, MemoryOnly<V, S>>> StateDBRef<K, V> for PlainDBState<K, DB, S> {
	type S = S;

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
	S: LinearState + SubAssign<S>,
	DB: PlainDBRef<K, MemoryOnly<V, S>> + PlainDB<K, MemoryOnly<V, S>>
> StateDB<K, V> for PlainDBState<K, DB, S> {
	// see inmemory
	type SE = Latest<S>;
	// see inmemory
	type GC = (S, Option<V>);
	// see inmemory
	type Migrate = (S, Self::GC);

	fn emplace(&mut self, key: K, value: V, at: &Self::SE) {
		if let Some(mut hist) = <DB as PlainDB<_, _>>::get(&self.db, &key) {
			match hist.set(value, at) {
				UpdateResult::Changed(_) => self.db.emplace(key.clone(), hist),
				UpdateResult::Cleared(_) => self.db.remove(&key),
				UpdateResult::Unchanged => return,
			}
		} else {
			self.db.emplace(key.clone(), MemoryOnly::new(value, at));
		}
		self.touched_keys.entry(at.latest().clone()).or_default().push(key);
	}

	fn remove(&mut self, key: &K, at: &Self::SE) {
		if let Some(mut hist) = <DB as PlainDB<_, _>>::get(&self.db, &key) {
			match hist.discard(at) {
				UpdateResult::Changed(_) => self.db.emplace(key.clone(), hist),
				UpdateResult::Cleared(_) => self.db.remove(&key),
				UpdateResult::Unchanged => return,
			}
		}
		self.touched_keys.entry(at.latest().clone()).or_default().push(key.clone());
	}

	fn gc(&mut self, gc: &Self::GC) {
		// retain for btreemap missing here.
		let mut states = Vec::new();
		for touched in self.touched_keys.keys() {
			if touched < &gc.0 {
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
		// probably a MaybeTrait could be use to implement
		// partially (or damn extract this in its own trait).
		unimplemented!("requires iterator on the full db");
	}
}

/// Implementation for plain db.
pub struct BTreeMap<K, V, S>(crate::rstd::BTreeMap<K, MemoryOnly<V, S>>, PhantomData<S>);

impl<K: Ord, V: Clone, S: LinearState> StateDBRef<K, V> for BTreeMap<K, V, S> {
	type S = S;

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
impl<K: Ord, V: Clone, S: LinearState> InMemoryStateDBRef<K, V> for BTreeMap<K, V, S> {
	fn get_ref(&self, key: &K, at: &Self::S) -> Option<&V> {
		self.0.get(key)
			.and_then(|h| h.get_ref(at))
	}
}

impl<K: Ord + Clone, V: Clone + Eq, S: LinearState + SubAssign<S>> StateDB<K, V> for BTreeMap<K, V, S> {
	// see inmemory
	type SE = Latest<S>;
	// see inmemory
	type GC = (S, Option<V>);
	// see inmemory
	type Migrate = (S, Self::GC);

	fn emplace(&mut self, key: K, value: V, at: &Self::SE) {
		if let Some(hist) = self.0.get_mut(&key) {
			hist.set(value, at);
		} else {
			self.0.insert(key, MemoryOnly::new(value, at));
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

// This is for small state as there is no double
// mapping an some operation goes through full scan.
pub struct LinearInMemoryManagement<H, S, V> {
	mapping: crate::rstd::BTreeMap<H, S>,
	start_treshold: S,
	next_state: S,
	neutral_element: Option<V>,
	changed_treshold: bool,
	can_append: bool,
}

impl<H, S, V> LinearInMemoryManagement<H, S, V> {
	// TODO should use a builder but then we neend
	// to change Management trait
	pub fn define_neutral_element(mut self, n: V) -> Self {
		self.neutral_element = Some(n);
		self
	}
}

impl<H, S: AddAssign<usize>, V> LinearInMemoryManagement<H, S, V> {
	pub fn prune(&mut self, nb: usize) {
		self.changed_treshold = true;
		self.start_treshold += nb
	}
}

impl<H: Ord, S: Clone, V: Clone> ManagementRef<H> for LinearInMemoryManagement<H, S, V> {
	type S = S;
	type GC = (S, Option<V>);
	type Migrate = (S, Self::GC);
	fn get_db_state(&self, state: &H) -> Option<Self::S> {
		self.mapping.get(state).cloned()
	}
	fn get_gc(&self) -> Option<Self::GC> {
		if self.changed_treshold {
			Some((self.start_treshold.clone(), self.neutral_element.clone()))
		} else {
			None
		}
	}
}

impl<
H: Ord + Clone,
S: Default + Clone + AddAssign<usize> + Ord,
V: Clone,
> Management<H> for LinearInMemoryManagement<H, S, V> {
	type SE = Latest<S>;
	fn init() -> (Self, Self::S) {
		let state = S::default();
		let mut next_state = S::default();
		next_state += 1;
		let mapping = Default::default();
		(LinearInMemoryManagement {
			mapping,
			start_treshold: state.clone(),
			next_state,
			neutral_element: None,
			changed_treshold: false,
			can_append: true,
		}, state)
	}
	fn latest_state(&self) -> Self::SE {
		// TODO can use next_state - 1 to avoid this search
		Latest::unchecked_latest(self.mapping.values().max()
			.map(Clone::clone)
			.unwrap_or(S::default()))
	}
	fn reverse_lookup(&self, state: &Self::S) -> Option<H> {
		// TODO could be the closest valid and return non optional!!!! TODO
		self.mapping.iter()
			.find(|(_k, v)| v == &state)
			.map(|(k, _v)| k.clone())
	}

	fn applied_gc(&mut self, gc: Self::GC) {
		self.changed_treshold = false;
		self.start_treshold = gc.0;
	}

	fn get_migrate(self) -> Migrate<H, Self> {
		unimplemented!()
	}

	fn applied_migrate(&mut self) {
		unimplemented!()
	}
}

impl<
H: Ord + Clone,
S: Default + Clone + SubAssign<S> + AddAssign<usize> + Ord,
V: Clone,
> LinearManagement<H> for LinearInMemoryManagement<H, S, V> {
	fn append_external_state(&mut self, state: H) -> Option<Self::S> {
		if !self.can_append {
			return None;
		}
		self.mapping.insert(state, self.next_state.clone());
		let result = self.next_state.clone();
		self.next_state += 1;
		Some(result)
	}

	fn drop_last_state(&mut self) -> Self::S {
		if self.next_state != S::default() {
			let mut dec = S::default();
			dec += 1;
			self.next_state -= dec;
		}
		self.can_append = true;
		self.next_state.clone()
	}
}
