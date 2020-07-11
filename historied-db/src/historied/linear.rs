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

use super::{HistoriedValue, ValueRef, Value, InMemoryValueRange, InMemoryValueRef, InMemoryValueSlice, InMemoryValue};
use crate::{UpdateResult, Latest};
use crate::rstd::marker::PhantomData;
use crate::rstd::convert::TryFrom;
use crate::rstd::ops::{AddAssign, SubAssign, Range};
use codec::{Encode, Decode};
use crate::backend::{LinearStorage, LinearStorageMem, LinearStorageSlice, LinearStorageRange};
use crate::backend::encoded_array::EncodedArrayValue;

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

/// Implementation of linear value history storage.
#[derive(Debug, Encode, Decode)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct Linear<V, S, D>(D, PhantomData<(V, S)>);

impl<V, S, D: AsRef<[u8]>> AsRef<[u8]> for Linear<V, S, D> {
	fn as_ref(&self) -> &[u8] {
		self.0.as_ref()
	}
}

impl<V, S, D: AsMut<[u8]>> AsMut<[u8]> for Linear<V, S, D> {
	fn as_mut(&mut self) -> &mut [u8] {
		self.0.as_mut()
	}
}

impl<V, S, D: Clone> Clone for Linear<V, S, D> {
	fn clone(&self) -> Self {
		Linear(self.0.clone(), PhantomData)
	}
}

impl<V, S, D: EncodedArrayValue> EncodedArrayValue for Linear<V, S, D> {
	fn from_slice(slice: &[u8]) -> Self {
		let v = D::from_slice(slice);
		Linear(v, PhantomData)
	}
}

impl<V, S, D: Default> Default for Linear<V, S, D> {
	fn default() -> Self {
		let v = D::default();
		Linear(v, PhantomData)
	}
}

impl<V, S, D: LinearStorage<V, S>> LinearStorage<V, S> for Linear<V, S, D> {
	fn truncate_until(&mut self, split_off: usize) {
		self.0.truncate_until(split_off)
	}
	fn len(&self) -> usize {
		self.0.len()
	}
	fn st_get(&self, index: usize) -> Option<HistoriedValue<V, S>> {
		self.0.st_get(index)
	}
	fn get_state(&self, index: usize) -> Option<S> {
		self.0.get_state(index)
	}
	fn push(&mut self, value: HistoriedValue<V, S>) {
		self.0.push(value)
	}
	fn insert(&mut self, index: usize, value: HistoriedValue<V, S>) {
		self.0.insert(index, value)
	}
	fn remove(&mut self, index: usize) {
		self.0.remove(index)
	}
	fn last(&self) -> Option<HistoriedValue<V, S>> {
		self.0.last()
	}
	fn pop(&mut self) -> Option<HistoriedValue<V, S>> {
		self.0.pop()
	}
	fn clear(&mut self) {
		self.0.clear()
	}
	fn truncate(&mut self, at: usize) {
		self.0.truncate(at)
	}
	fn emplace(&mut self, at: usize, value: HistoriedValue<V, S>) {
		self.0.emplace(at, value)
	}
}

/// Adapter for storage (allows to factor code while keeping simple types).
/// VR is the reference to value that is used, and I the initial state.
pub trait StorageAdapter<'a, S, VR, I> {
	fn get_adapt(inner: I, index: usize) -> Option<HistoriedValue<VR, S>>;
}

struct RefVecAdapter;
impl<'a, S, V, D: LinearStorageMem<V, S>> StorageAdapter<
	'a,
	S,
	&'a V,
	&'a D,
> for RefVecAdapter {
	fn get_adapt(inner: &'a D, index: usize) -> Option<HistoriedValue<&'a V, S>> {
		inner.get_ref(index)
	}
}
struct RefVecAdapterMut;
impl<'a, S, V, D: LinearStorageMem<V, S>> StorageAdapter<
	'a,
	S,
	&'a mut V,
	&'a mut D,
> for RefVecAdapterMut {
	fn get_adapt(inner: &'a mut D, index: usize) -> Option<HistoriedValue<&'a mut V, S>> {
		inner.get_ref_mut(index)
	}
}
struct ValueVecAdapter;
impl<'a, S, V, D: LinearStorage<V, S>> StorageAdapter<
	'a,
	S,
	V,
	&'a D,
> for ValueVecAdapter {
	fn get_adapt(inner: &'a D, index: usize) -> Option<HistoriedValue<V, S>> {
		inner.st_get(index)
	}
}
struct SliceAdapter;
impl<'a, S, D: LinearStorageSlice<Vec<u8>, S>> StorageAdapter<
	'a,
	S,
	&'a [u8],
	&'a D,
> for SliceAdapter {
	fn get_adapt(inner: &'a D, index: usize) -> Option<HistoriedValue<&'a [u8], S>> {
		inner.get_slice(index)
	}
}
struct SliceAdapterMut;
impl<'a, S, D: LinearStorageSlice<Vec<u8>, S>> StorageAdapter<
	'a,
	S,
	&'a mut [u8],
	&'a mut D,
> for SliceAdapter {
	fn get_adapt(inner: &'a mut D, index: usize) -> Option<HistoriedValue<&'a mut [u8], S>> {
		inner.get_slice_mut(index)
	}
}

impl<V: Clone, S: LinearState, D: LinearStorage<V, S>> ValueRef<V> for Linear<V, S, D> {
	type S = S;

	fn get(&self, at: &Self::S) -> Option<V> {
		self.get_adapt::<_, ValueVecAdapter>(at)
	}

	fn contains(&self, at: &Self::S) -> bool {
		self.pos(at).is_some()
	}

	fn is_empty(&self) -> bool {
		self.0.len() == 0
	}
}

impl<V, S: LinearState, D: LinearStorageRange<V, S>> InMemoryValueRange<S> for Linear<V, S, D> {
	fn get_range(slice: &[u8], at: &S) -> Option<Range<usize>> {
		if let Some(inner) = D::from_slice(slice) {
			let inner = Linear(inner, PhantomData);
			let mut index = inner.len();
			if index == 0 {
				return None;
			}
			while index > 0 {
				index -= 1;
				if let Some(HistoriedValue { value, state }) = D::get_range(slice, index) {
					if state.exists(at) {
						return Some(value);
					}
				}
			}
			None
		} else {
			None
		}
	}
}

impl<V, S: LinearState, D: LinearStorage<V, S>> Linear<V, S, D> {
	fn get_adapt<'a, VR, A: StorageAdapter<'a, S, VR, &'a D>>(&'a self, at: &S) -> Option<VR> {
		let mut index = self.0.len();
		if index == 0 {
			return None;
		}
		while index > 0 {
			index -= 1;
			if let Some(HistoriedValue { value, state }) = A::get_adapt(&self.0, index) {
				if state.exists(at) {
					return Some(value);
				}
			}
		}
		None
	}

	fn get_adapt_mut<'a, VR, A: StorageAdapter<'a, S, VR, &'a mut D>>(&'a mut self, at: &S)
		-> Option<HistoriedValue<VR, S>> {
		let pos = self.pos(at);
		let self_mut: &'a mut D = &mut self.0;
		if let Some(index) = pos {
			A::get_adapt(self_mut, index)
		} else {
			None
		}
	}
}

impl<V, S: LinearState, D: LinearStorage<V, S>> Linear<V, S, D> {
	fn pos(&self, at: &S) -> Option<usize> {
		let mut index = self.0.len();
		if index == 0 {
			return None;
		}
		let mut pos = None;
		while index > 0 {
			index -= 1;
			if let Some(vr) = self.0.get_state(index) {
				if at == &vr {
					pos = Some(index);
					break
				}
				if at < &vr {
					break;
				}
			}
		}
		pos
	}

/*	fn clean_to_pos(&self, at: &S, value: &V) -> UpdateResult<()> {
		loop {
			if let Some(last) = self.0.last() {
				// TODO this is rather unsafe: we expect that
				// when changing value we use a state that is
				// the latest from the state management.
				// Their could be ways to enforce that, but nothing
				// good at this point.
				if &last.state > at {
					self.0.pop();
					if self.0.len() == 0 {
						return UpdateResult::Cleared(());
					}
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
		UpdateResult::Changed(())
	}*/
	
}

impl<V: Clone, S: LinearState, D: LinearStorageMem<V, S>> InMemoryValueRef<V> for Linear<V, S, D> {
	fn get_ref(&self, at: &Self::S) -> Option<&V> {
		self.get_adapt::<_, RefVecAdapter>(at)
	}
}

impl<S: LinearState, D: LinearStorageSlice<Vec<u8>, S>> InMemoryValueSlice<Vec<u8>> for Linear<Vec<u8>, S, D> {
	fn get_slice(&self, at: &Self::S) -> Option<&[u8]> {
		self.get_adapt::<_, SliceAdapter>(at)
	}
}

//impl<V: Clone, S: LinearState, Q: LinearStateLatest<S>> Value<V> for Linear<V, S> {
impl<V: Clone + Eq, S: LinearState + SubAssign<S>, D: LinearStorage<V, S>> Value<V> for Linear<V, S, D> {
	type SE = Latest<S>;
	type Index = S;
	type GC = LinearGC<S, V>;
	/// Migrate will act as GC but also align state to 0.
	/// First index being the number for start state that
	/// will be removed after migration.
	type Migrate = (S, Self::GC);

	fn new(value: V, at: &Self::SE) -> Self {
		let mut v = D::default();
		let state = at.latest().clone();
		v.push(HistoriedValue{ value, state });
		Linear(v, PhantomData)
	}

	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()> {
		// Warn DUPLICATED code with set mut : see refactoring to include 'a for in mem
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
					self.0.pop();
				}
			}
			break;
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

	fn gc(&mut self, gc: &Self::GC) -> UpdateResult<()> {
		if gc.new_start.is_some() && gc.new_start == gc.new_end {
			self.0.clear();
			return UpdateResult::Cleared(());
		}

		let mut end_result = UpdateResult::Unchanged;
		if let Some(new_end) = gc.new_end.as_ref() {

			let mut index = self.0.len();
			while index > 0 {
				if let Some(HistoriedValue{ value: _, state }) = self.0.st_get(index - 1) {
					if &state < new_end {
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
				if let Some(HistoriedValue{ value: _, state }) = self.0.st_get(index) {
					if &state >= start_treshold {
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
				while let Some(HistoriedValue{ value, state: _ }) = self.0.st_get(index) {
					if &value != neutral {
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
			self.0.truncate_until(index);
			UpdateResult::Changed(())
		} else {
			return end_result;
		}
	}

	fn migrate(&mut self, (mig, gc): &mut Self::Migrate) -> UpdateResult<()> {
		let res = self.gc(gc);
		let len = self.0.len();
		if len > 0 {
			/* TODO write iter_mut that iterate on a HandleMut as in simple_db
			for h in self.0.iter_mut() {
				if &h.state > mig {
					h.state -= mig.clone();
				} else {
					h.state = Default::default();
				}
			}
			*/
		
			for i in 0..len {
				if let Some(mut h) = self.0.st_get(i) {
					if &h.state > mig {
						h.state -= mig.clone();
					} else {
						h.state = Default::default();
					}
					self.0.emplace(i, h);
				} else {
					unreachable!("len checked")
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

impl<V: Clone + Eq, S: LinearState + SubAssign<S>, D: LinearStorageMem<V, S>> InMemoryValue<V> for Linear<V, S, D> {
	fn get_mut(&mut self, at: &Self::SE) -> Option<&mut V> {
		let at = at.latest();
		self.get_adapt_mut::<_, RefVecAdapterMut>(at).map(|h| h.value)
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

impl Linear<Option<Vec<u8>>, u32, crate::backend::in_memory::MemoryOnly<Option<Vec<u8>>, u32>> {
	/// Temporary function to get occupied stage.
	/// TODO replace by heapsizeof
	pub fn temp_size(&self) -> usize {
		let mut size = 0;
		for h in (self.0).0.iter() {
			size += 4; // usize as u32 for index
			size += 1; // optional
			size += h.value.as_ref().map(|v| v.len()).unwrap_or(0);
		}
		size
	}
}
