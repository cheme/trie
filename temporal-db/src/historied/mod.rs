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

//! Linear historied data.

#[cfg(not(feature = "std"))]
use crate::rstd::{vec::Vec, vec};
use crate::UpdateResult;

pub mod linear;
pub mod encoded_array;

/// Trait for historied value
pub trait ValueRef<V> {
	/// State to query for this value.
	type S;

	/// Get value at this state.
	fn get(&self, at: &Self::S) -> Option<V>;

	/// Check if a value exists at this state.
	fn contains(&self, at: &Self::S) -> bool;

	/// Check if this is empty.
	fn is_empty(&self) -> bool;
}

pub trait InMemoryValueRef<V>: ValueRef<V> {
	/// Get reference to the value at this state.
	fn get_ref(&self, at: &Self::S) -> Option<&V>;
}

/// Trait for historied value.
pub trait Value<V>: ValueRef<V> {
	/// State to use here.
	/// We use a different state than
	/// for the ref as it can use different
	/// constraints.
	type SE;
	//type SE = Self::S; TODO next nightly and future stable should accept it
	/// GC strategy that can be applied.
	/// GC can be run in parallel, it does not
	/// make query incompatible.
	type GC;
	/// Like gc but operation require a lock on the db
	/// and all pending state are invalidated.
	type Migrate;

	/// Insert or update a value.
	fn set(&mut self, value: V, at: &Self::SE);


	/// Discard history at.
	fn discard(&mut self, at: &Self::SE) -> UpdateResult<Option<V>>;

	fn gc(&mut self, gc: Self::GC) -> UpdateResult<()>;
	fn migrate(&mut self, mig: Self::Migrate) -> UpdateResult<()>;
}

pub trait InMemoryValue<V>: Value<V> {
	/// Get latest value, can apply updates.
	fn get_mut(&mut self, at: &Self::S) -> Option<&mut V>;
}

/// An entry at a given history index.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct HistoriedValue<V, S> {
	/// The stored value.
	pub value: V,
	/// The state this value belongs to.
	pub state: S,
}

impl<V, S> From<(V, S)> for HistoriedValue<V, S> {
	fn from(input: (V, S)) -> HistoriedValue<V, S> {
		HistoriedValue { value: input.0, state: input.1 }
	}
}
