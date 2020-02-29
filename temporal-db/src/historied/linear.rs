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

//! Linear historied data temporal db implementations.

use super::{HistoriedValue, ValueRef, Value, InMemoryValueRef, InMemoryValue};
use crate::{StateDBRef, UpdateResult};
use hash_db::PlainDBRef;
use crate::rstd::marker::PhantomData;
use crate::rstd::convert::{TryFrom, TryInto};
use crate::rstd::ops::{Add, SubAssign, Range};
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
	+ SubAssign<usize> // TODO can remove ??
	+ Add<usize> // TODO can remove ??
	+ PartialEq<usize>
{
	fn previous(&mut self) -> bool {
		if *self == 0 {
			false
		} else {
			*self -= 1;
			true
		}
	}

	// stored state and query state are
	// the same for linear state.
	fn exists(&self, at: &Self) -> bool {
		self <= at
	}
}

impl<S> LinearState for S where S:
	Default
	+ Clone
	+ Ord
	+ PartialOrd
	+ TryFrom<usize>
	+ SubAssign<usize>
	+ Add<usize>
	+ PartialEq<usize>
{ }


/// This is a rather simple way of managing state, as state should not be
/// invalidated at all (can be change at latest state, also drop but not at 
/// random state).
///
/// Note that this does not guaranty the state
/// is the latest, but it shows the intention.
pub struct Latest<S>(S);

impl<S> Latest<S> {
	fn latest(&self) -> &S {
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

impl<V, S> MemoryOnly<V, S> {
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
impl<V: Clone + Eq, S: LinearState> Value<V> for MemoryOnly<V, S> {
	type SE = Latest<S>;
	/// Removing existing state before.
	/// Optionally a skipable value (for
	/// history containing deletion that is deletion
	/// as a deletion on empty value can be remove).
	type GC = (S, Option<V>);
	/// Migrate will act as GC but also align state to 0.
	type Migrate = Range<S>;

	fn set(&mut self, value: V, at: &Self::SE) {
		let at = at.latest();
		if let Some(last) = self.0.last() {
			// TODO this is rather unsafe: we expact that
			// when changing value we use a state that is
			// the latest from the state management.
			// Their could be ways to enforce that, but nothing
			// good at this point.
			debug_assert!(&last.state <= at); 
			if at == &last.state {
				self.0.pop();
			}
		}
		self.0.push(HistoriedValue {value, state: at.clone()});
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

	fn gc(&mut self, (start_treshold, start_void): Self::GC) -> UpdateResult<()> {
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

	fn migrate(&mut self, mig: Self::Migrate) -> UpdateResult<()> {
		unimplemented!()
	}
}


impl<V: Clone + Eq, S: LinearState> InMemoryValue<V> for MemoryOnly<V, S> {
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
pub struct PlainDBState<DB, S>(DB, PhantomData<S>);

impl<K, V, S: LinearState, DB: PlainDBRef<K, MemoryOnly<V, S>>> StateDBRef<K, V> for PlainDBState<DB, S> {
	type S = S;

	fn get(&self, key: &K, at: &Self::S) -> Option<V> {
		unimplemented!()
	}

	fn contains(&self, key: &K, at: &Self::S) -> bool {
		unimplemented!()
	}
}
