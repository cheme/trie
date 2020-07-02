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

use super::{HistoriedValue, ValueRef, Value, InMemoryValueRef, InMemoryValue, StateIndex};
use crate::{StateDBRef, UpdateResult, InMemoryStateDBRef, StateDB, ManagementRef,
	Management, Migrate, LinearManagement, Latest};
use crate::rstd::marker::PhantomData;
use crate::rstd::convert::{TryFrom, TryInto};
use crate::rstd::ops::{AddAssign, SubAssign, Range};
use crate::rstd::mem::replace;
use codec::{Encode, Decode, Codec};

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
	+ TryFrom<u32>
	+ AddAssign<u32> // TODO can remove ??
	+ PartialEq<u32>
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
	+ TryFrom<u32>
	+ AddAssign<u32>
	+ PartialEq<u32>
{ }

/// Size of preallocated history per element.
/// Currently at two for committed and prospective only.
/// It means that using transaction in a module got a direct allocation cost.
const ALLOCATED_HISTORY: usize = 2;

/// Array like buffer for in memory storage.
/// By in memory we expect that this will
/// not required persistence and is not serialized.
#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct Linear<V, S>(smallvec::SmallVec<[HistoriedValue<V, S>; ALLOCATED_HISTORY]>);

/// Array like buffer for in memory storage.
/// By in memory we expect that this will
/// not required persistence and is not serialized.
#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct MemoryOnly<V, S>(smallvec::SmallVec<[HistoriedValue<V, S>; ALLOCATED_HISTORY]>);

/// Backend for linear storage with inmemory reference.
pub trait LinearStorageRef<V, S> {
}

/// Backend for linear storage.
pub trait LinearStorage<V, S> {
}

impl<V, S> LinearStorageRef<V, S> for MemoryOnly<V, S> {
}

impl<V, S> LinearStorage<V, S> for MemoryOnly<V, S> {
}

/// Implementation of linear value history storage.
#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct LinearNext<V, S, D>(D, PhantomData<(V, S)>);

impl<V, S: Clone> Linear<V, S> {
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

impl<V: Clone, S: LinearState> ValueRef<V> for Linear<V, S> {
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

impl<V: Clone, S: LinearState> InMemoryValueRef<V> for Linear<V, S> {
	fn get_ref(&self, at: &Self::S) -> Option<&V> {
		let mut index = self.0.len();
		if index == 0 {
			return None;
		}
		while index > 0 {
			index -= 1;
			if let Some(HistoriedValue { value, state }) = self.0.get(index) {
				if state.exists(at) {
					return Some(value);
				}
			}
		}
		None
	}
}

//impl<V: Clone, S: LinearState, Q: LinearStateLatest<S>> Value<V> for Linear<V, S> {
impl<V: Clone + Eq, S: LinearState + SubAssign<S>> Value<V> for Linear<V, S> {
	type SE = Latest<S>;
	type Index = S;
	type GC = LinearGC<S, V>;
	/// Migrate will act as GC but also align state to 0.
	/// First index being the number for start state that
	/// will be removed after migration.
	type Migrate = (S, Self::GC);

	fn new(value: V, at: &Self::SE) -> Self {
		let mut v = smallvec::SmallVec::default();
		let state = at.latest().clone();
		v.push(HistoriedValue{ value, state });
		Linear(v)
	}

	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()> {
		self.set_mut(value, at).map(|v| ())
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

	fn gc(&mut self, gc: &Self::GC) -> UpdateResult<()> {
		if gc.new_start.is_some() && gc.new_start == gc.new_end {
			self.0.clear();
			return UpdateResult::Cleared(());
		}

		let mut end_result = UpdateResult::Unchanged;
		if let Some(new_end) = gc.new_end.as_ref() {

			let mut index = self.0.len();
			while index > 0 {
				if let Some(HistoriedValue{ value: _, state }) = self.0.get(index - 1) {
					if state < new_end {
						break;
					}
				} else {
					break;
				}
				index -= 1;
			}

			if index == 0 {
				self.0.clear();
				return UpdateResult::Cleared(());
			} else if index != self.0.len() {
				self.0.truncate(index);
				end_result = UpdateResult::Changed(());
			}
		}

		if let Some(start_treshold) = gc.new_start.as_ref() {
			let mut index = 0;
			loop {
				if let Some(HistoriedValue{ value: _, state }) = self.0.get(index) {
					if state >= start_treshold {
						index = index.saturating_sub(1);
						break;
					}
				} else {
					index = index.saturating_sub(1);
					break;
				}
				index += 1;
			}
			if let Some(neutral) = gc.neutral_element.as_ref() {
				while let Some(HistoriedValue{ value, state: _ }) = self.0.get(index) {
					if value != neutral {
						break;
					}
					index += 1;
				}
			}
			if index == 0 {
				return end_result;
			}
			if index == self.0.len() {
				self.0.clear();
				return UpdateResult::Cleared(());
			}
			self.remove_start(index);
			UpdateResult::Changed(())
		} else {
			return end_result;
		}
	}

	fn migrate(&mut self, (mig, gc): &mut Self::Migrate) -> UpdateResult<()> {
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

	fn is_in_migrate(index: &Self::Index, gc: &Self::Migrate) -> bool {
		gc.1.new_start.as_ref().map(|s| index < s).unwrap_or(false)
			|| gc.1.new_end.as_ref().map(|s| index >= s).unwrap_or(false)
	}
}


impl<V: Clone + Eq, S: LinearState + SubAssign<S>> InMemoryValue<V> for Linear<V, S> {
	fn get_mut(&mut self, at: &Self::SE) -> Option<&mut V> {
		let mut index = self.0.len();
		if index == 0 {
			return None;
		}
		let at = at.latest();
		while index > 0 {
			index -= 1;
			if let Some(HistoriedValue { value: _, state }) = self.0.get(index) {
				if at == state {
					return self.0.get_mut(index).map(|v| &mut v.value)
				}
				if at < state {
					break;
				}
			}
		}
		None
	}

	fn set_mut(&mut self, value: V, at: &Self::SE) -> UpdateResult<Option<V>> {
		let mut result = None;
		let at = at.latest();
		loop {
			if let Some(last) = self.0.last() {
				// TODO this is rather unsafe: we expect that
				// when changing value we use a state that is
				// the latest from the state management.
				// Their could be ways to enforce that, but nothing
				// good at this point.
				if &last.state > at {
					self.0.pop();
					continue;
				} 
				if at == &last.state {
					if last.value == value {
						return UpdateResult::Unchanged;
					}
					result = self.0.pop();
				}
			}
			break;
		}
		self.0.push(HistoriedValue {value, state: at.clone()});
		UpdateResult::Changed(result.map(|r| r.value))
	}


}

#[derive(Debug, Clone, Encode, Decode)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct LinearGC<S, V> {
	// inclusive
	pub(crate) new_start: Option<S>,
	// exclusive
	pub(crate) new_end: Option<S>,
	// TODO use reference??
	pub(crate) neutral_element: Option<V>,
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

impl<H, S: AddAssign<u32>, V> LinearInMemoryManagement<H, S, V> {
	pub fn prune(&mut self, nb: usize) {
		self.changed_treshold = true;
		self.start_treshold += nb as u32
	}
}

impl<H: Ord, S: Clone, V: Clone> ManagementRef<H> for LinearInMemoryManagement<H, S, V> {
	type S = S;
	type GC = (S, Option<V>);
	type Migrate = (S, Self::GC);
	fn get_db_state(&mut self, state: &H) -> Option<Self::S> {
		self.mapping.get(state).cloned()
	}
	fn get_gc(&self) -> Option<crate::Ref<Self::GC>> {
		if self.changed_treshold {
			Some(crate::Ref::Owned((self.start_treshold.clone(), self.neutral_element.clone())))
		} else {
			None
		}
	}
}

impl<
H: Ord + Clone,
S: Default + Clone + AddAssign<u32> + Ord,
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

	fn get_db_state_mut(&mut self, state: &H) -> Option<Self::SE> {
		if let Some(state) = self.mapping.get(state) {
			let latest = self.mapping.values().max()
				.map(Clone::clone)
				.unwrap_or(S::default());
			if state == &latest {
				return Some(Latest::unchecked_latest(latest))
			}
		}
		None
	}

	fn latest_state(&mut self) -> Self::SE {
		// TODO can use next_state - 1 to avoid this search
		Latest::unchecked_latest(self.mapping.values().max()
			.map(Clone::clone)
			.unwrap_or(S::default()))
	}

	fn reverse_lookup(&mut self, state: &Self::S) -> Option<H> {
		// TODO could be the closest valid and return non optional!!!! TODO
		self.mapping.iter()
			.find(|(_k, v)| v == &state)
			.map(|(k, _v)| k.clone())
	}

	fn get_migrate(self) -> Migrate<H, Self> {
		unimplemented!()
	}

	fn applied_migrate(&mut self) {
		self.changed_treshold = false;
		//self.start_treshold = gc.0; // TODO from backed inner state

		unimplemented!()
	}
}

impl<
H: Ord + Clone,
S: Default + Clone + SubAssign<S> + AddAssign<u32> + Ord,
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

impl Linear<Option<Vec<u8>>, u32> {
	/// Temporary function to get occupied stage.
	/// TODO replace by heapsizeof
	pub fn temp_size(&self) -> usize {
		let mut size = 0;
		for h in self.0.iter() {
			size += 4; // usize as u32 for index
			size += 1; // optional
			size += h.value.as_ref().map(|v| v.len()).unwrap_or(0);
		}
		size
	}
}
