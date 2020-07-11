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

//! Linear backend structures for historied data storage.

use crate::historied::HistoriedValue;
use crate::rstd::ops::Range;

/// Data stored as rust structs in memory.
pub mod in_memory;

/// Data encoded in a byte buffer, no unserialized
/// stractures.
pub mod encoded_array;

/// Backend for linear storage.
pub trait LinearStorage<V, S>: Default {
	/// This does not need to be very efficient as it is mainly for
	/// garbage collection.
	fn truncate_until(&mut self, split_off: usize);
	/// Number of element for different S.
	fn len(&self) -> usize;
	/// Array like get.
	fn st_get(&self, index: usize) -> Option<HistoriedValue<V, S>>;
	/// Array like get.
	fn get_state(&self, index: usize) -> Option<S>;
	/// Vec like push.
	fn push(&mut self, value: HistoriedValue<V, S>);
	/// Vec like insert, this is mainly use in tree implementation.
	/// So when used as tree branch container, a efficient implementation
	/// shall be use.
	fn insert(&mut self, index: usize, value: HistoriedValue<V, S>);
	/// Vec like remove, this is mainly use in tree branch implementation.
	fn remove(&mut self, index: usize);
	/// TODO put 'a and return read type that can be &'a S and where S is AsRef<S>.
	/// TODO put 'a and return read type that can be &'a [u8] and where Vec<u8> is AsRef<[u8]>.
	fn last(&self) -> Option<HistoriedValue<V, S>> {
		if self.len() > 0 {
			self.st_get(self.len() - 1)
		} else {
			None
		}
	}
	fn pop(&mut self) -> Option<HistoriedValue<V, S>>;
	fn clear(&mut self);
	fn truncate(&mut self, at: usize);
	/// This can be slow, only define in migrate.
	/// TODO consider renaming.
	fn emplace(&mut self, at: usize, value: HistoriedValue<V, S>);
}

/// Backend for linear storage with inmemory reference.
pub trait LinearStorageSlice<V: AsRef<[u8]> + AsMut<[u8]>, S>: LinearStorage<V, S> {
	/// Array like get.
	fn get_slice(&self, index: usize) -> Option<HistoriedValue<&[u8], S>>;
	fn last_slice(&self) -> Option<HistoriedValue<&[u8], S>> {
		if self.len() > 0 {
			self.get_slice(self.len() - 1)
		} else {
			None
		}
	}
	/// Array like get mut.
	fn get_slice_mut(&mut self, index: usize) -> Option<HistoriedValue<&mut [u8], S>>;
}

/// Backend for linear storage with inmemory reference.
pub trait LinearStorageMem<V, S>: LinearStorage<V, S> {
	/// Array like get.
	fn get_ref(&self, index: usize) -> Option<HistoriedValue<&V, S>>;
	fn last_ref(&self) -> Option<HistoriedValue<&V, S>> {
		if self.len() > 0 {
			self.get_ref(self.len() - 1)
		} else {
			None
		}
	}
	/// Array like get mut.
	fn get_ref_mut(&mut self, index: usize) -> Option<HistoriedValue<&mut V, S>>;
}

pub trait LinearStorageRange<V, S>: LinearStorage<V, S> {
	/// Array like get. TODO consider not returning option (same for from_slice), inner
	/// implementation being unsafe.
	fn get_range(slice: &[u8], index: usize) -> Option<HistoriedValue<Range<usize>, S>>;

	fn from_slice(slice: &[u8]) -> Option<Self>;
}

