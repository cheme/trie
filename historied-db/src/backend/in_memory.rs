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

//! In memory backend structure.

use crate::historied::HistoriedValue;
use codec::{Encode, Decode, Input as CodecInput};
use super::{LinearStorage, LinearStorageMem};
use crate::rstd::mem::replace;

/// Size of preallocated history per element.
/// Currently at two for committed and prospective only.
/// It means that using transaction in a module got a direct allocation cost.
const ALLOCATED_HISTORY: usize = 2;

/// Array like buffer for in memory storage.
/// By in memory we expect that this will
/// not required persistence and is not serialized.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MemoryOnly<V, S>(pub(crate) smallvec::SmallVec<[HistoriedValue<V, S>; ALLOCATED_HISTORY]>);


impl<V: Encode, S: Encode> Encode for MemoryOnly<V, S> {

	fn size_hint(&self) -> usize {
		self.0.as_slice().size_hint()
	}

	fn encode(&self) -> Vec<u8> {
		self.0.as_slice().encode()
	}

/*	fn using_encoded<R, F: FnOnce(&[u8]) -> R>(&self, f: F) -> R {
		f(&self.0)
	}*/
}

impl<V: Decode, S: Decode> Decode for MemoryOnly<V, S> {
	fn decode<I: CodecInput>(value: &mut I) -> Result<Self, codec::Error> {
		// TODO make a variant when len < ALLOCATED_HISTORY
		let v = Vec::decode(value)?;
		Ok(MemoryOnly(smallvec::SmallVec::from_vec(v)))
	}
}

impl<V, S> Default for MemoryOnly<V, S> {
	fn default() -> Self {
		MemoryOnly(smallvec::SmallVec::default())
	}
}

impl<V: Clone, S: Clone> LinearStorageMem<V, S> for MemoryOnly<V, S> {
	fn get_ref(&self, index: usize) -> Option<HistoriedValue<&V, S>> {
		if let Some(HistoriedValue { value, state }) = self.0.get(index) {
			Some(HistoriedValue { value: &value, state: state.clone() })
		} else {
			None
		}
	}

	fn get_ref_mut(&mut self, index: usize) -> Option<HistoriedValue<&mut V, S>> {
		if let Some(HistoriedValue { value, state }) = self.0.get_mut(index) {
			Some(HistoriedValue { value, state: state.clone() })
		} else {
			None
		}
	}
}

impl<V: Clone, S: Clone> LinearStorage<V, S> for MemoryOnly<V, S> {
	fn truncate_until(&mut self, split_off: usize) {
		if self.0.spilled() {
			let new = replace(&mut self.0, Default::default());
			self.0 = smallvec::SmallVec::from_vec(new.into_vec().split_off(split_off));
		} else {
			for i in 0..split_off {
				self.0.remove(i);
			}
		}
	}
	fn len(&self) -> usize {
		self.0.len()
	}
	fn st_get(&self, index: usize) -> Option<HistoriedValue<V, S>> {
		self.0.get(index).cloned()
	}
	fn get_state(&self, index: usize) -> Option<S> {
		self.0.get(index).map(|h| h.state.clone())
	}
	fn push(&mut self, value: HistoriedValue<V, S>) {
		self.0.push(value)
	}
	fn insert(&mut self, index: usize, value: HistoriedValue<V, S>) {
		self.0.insert(index, value)
	}
	fn remove(&mut self, index: usize) {
		self.0.remove(index);
	}
	fn last(&self) -> Option<HistoriedValue<V, S>> {
		self.0.last().cloned()
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
		self.0[at] = value;
	}
}

