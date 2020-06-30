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

//! Tree historied data historied db implementations.

use super::{HistoriedValue, ValueRef, Value, InMemoryValueRef, InMemoryValue, UpdateResult};
use crate::historied::linear::{MemoryOnly as MemoryOnlyLinear, LinearState, LinearGC};
use crate::historied::tree_management::{ForkPlan, BranchesContainer, TreeMigrate, TreeStateGc};
use crate::rstd::ops::{AddAssign, SubAssign, Range};
use crate::Latest;

// TODO for not in memory we need some direct or indexed api, returning value
// and the info if there can be lower value index (not just a direct index).
// -> then similar to those reverse iteration with possible early exit.
// -> Also need to attach some location index (see enumerate use here)

#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct MemoryOnly<I, BI, V> {
	branches: Vec<MemoryOnlyBranch<I, BI, V>>,
	// TODO add optional range indexing.
	// Indexing is over couple (I, BI), runing on fix size batches (aka max size).
	// First try latest, then try indexing, (needs 3 methods
	// get_latest, get_index, iter; currently we use directly iter).

	// TODO add an optional pointer to deeper branch
	// to avoid iterating to much over latest fork when the
	// deepest most usefull branch manage to keep a linear history
	// but a lower branch index.
	// (conf needed in state also with optional indexing).
}

impl<I, BI, V> Default for MemoryOnly<I, BI, V> {
	fn default() -> Self {
		MemoryOnly{
			branches: Vec::new(),
		}
	}
}

#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct MemoryOnlyBranch<I, BI, V> {
	branch_index: I,
	history: MemoryOnlyLinear<V, BI>,
}

impl<I: Clone, BI: LinearState + SubAssign<BI>, V: Clone + Eq> MemoryOnlyBranch<I, BI, V> {
	pub fn new(value: V, state: &Latest<(I, BI)>) -> Self {
		let (branch_index, index) = state.latest().clone();
		let index = Latest::unchecked_latest(index); // TODO cast ptr?
		let history = MemoryOnlyLinear::new(value, &index);
		MemoryOnlyBranch{
			branch_index,
			history,
		}
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32>, // TODO consider subassing usize or minus one trait...
	V: Clone,
> ValueRef<V> for MemoryOnly<I, BI, V> {
	type S = ForkPlan<I, BI>;

	fn get(&self, at: &Self::S) -> Option<V> {
		self.get_ref(at).map(|v| v.clone())
	}

	fn contains(&self, at: &Self::S) -> bool {
		self.get_ref(at).is_some()
	}

	fn is_empty(&self) -> bool {
		// This implies remove from linear clean directly the parent vec.
		self.branches.is_empty()
	}
}
impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32>,
	V: Clone,
> InMemoryValueRef<V> for MemoryOnly<I, BI, V> {
	fn get_ref(&self, at: &<Self as ValueRef<V>>::S) -> Option<&V> {
		let mut index = self.branches.len();
		// note that we expect branch index to be linearily set
		// along a branch (no state containing unordered branch_index
		// and no history containing unorderd branch_index).
		if index == 0 {
			return None;
		}

		for (state_branch_range, state_branch_index) in at.iter() {
			while index > 0 {
				let branch_index = &self.branches[index - 1].branch_index;
				if branch_index < &state_branch_index {
					break;
				} else if branch_index == &state_branch_index {
					// TODO add a lower bound check (maybe debug_assert it only).
					let mut upper_bound = state_branch_range.end.clone();
					upper_bound -= 1;
					if let Some(result) = self.branches[index - 1].history.get_ref(&upper_bound) {
						return Some(result)
					}
				}
				index -= 1;
			}
		}

		// composite part.
		while index > 0 {
			let branch_index = &self.branches[index - 1].branch_index;
			if branch_index <= &at.composite_treshold.0 {
				if let Some(result) = self.branches[index - 1].history.get_ref(&at.composite_treshold.1) {
					return Some(result)
				}
			}
			index -= 1;
		}
	
		None
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32> + SubAssign<BI>,
	V: Clone + Eq,
> Value<V> for MemoryOnly<I, BI, V> {
	type SE = Latest<(I, BI)>;
	type Index = (I, BI);
	type GC = TreeStateGc<I, BI, V>;
	type Migrate = (BI, TreeMigrate<I, BI, V>);

	fn new(value: V, at: &Self::SE) -> Self {
		let mut v = Vec::new();
		v.push(MemoryOnlyBranch::new(value, at));
		MemoryOnly {
			branches: v,
		}
	}

	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()> {
		self.set_mut(value, at).map(|v| ())
	}

	fn discard(&mut self, at: &Self::SE) -> UpdateResult<Option<V>> {
		let (branch_index, index) = at.latest();
		for branch in self.branches.iter_mut().rev() {
			if &branch.branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.history.discard(&index);
			}
			if &branch.branch_index < branch_index {
				break;
			}
		}
		UpdateResult::Unchanged
	}

	fn gc(&mut self, gc: &mut Self::GC) -> UpdateResult<()> {

		let neutral = &gc.neutral_element;
		let mut result = UpdateResult::Unchanged;
		let start_len = self.branches.len();
		let mut to_remove = Vec::new(); // if switching to hash map retain usage is way better.
		let mut gc_iter = gc.storage.iter().rev();
		let start_composite = gc.composite_treshold.1.clone();
		let mut branch_iter = self.branches.iter_mut().enumerate().rev();
		let mut o_gc = gc_iter.next();
		let mut o_branch = branch_iter.next();
		while let (Some(gc), Some((index, branch))) = (o_gc.as_ref(), o_branch.as_mut()) {
			if gc.0 == &branch.branch_index {
				// TODO using linear gc does not make sense here (no sense of delta: TODO change
				// linear to use a simple range with neutral).
				let (start, end) = gc.1.range();
				let start = if start < start_composite {
					start_composite.clone()
				} else {
					start
				};
				let mut gc = LinearGC {
					new_start: Some(start),
					new_end:  Some(end),
					neutral_element: neutral.clone(),
				};

				match branch.history.gc(&mut gc) {
					UpdateResult::Unchanged => (),
					UpdateResult::Changed(_) => { result = UpdateResult::Changed(()); },
					UpdateResult::Cleared(_) => to_remove.push(*index),
				}

				o_gc = gc_iter.next();
				o_branch = branch_iter.next();
			} else if gc.0 < &branch.branch_index {
				to_remove.push(*index);
				o_branch = branch_iter.next();
			} else {
				o_gc = gc_iter.next();
			}
		}

		for i in to_remove.into_iter() {
			self.branches.remove(i);
		}

		if self.branches.len() == 0 {
			result = UpdateResult::Cleared(());
		} else if self.branches.len() != start_len {
			result = UpdateResult::Changed(());
		}

		result
	}

	// TODO this is rather costy and would run in a loop, consider using btreemap instead of vec in
	// treegc
	fn is_in_migrate((index, linear_index) : &Self::Index, gc: &Self::Migrate) -> bool {
		for branch in gc.1.changes.iter().rev() {
			let bi = &gc.0;
			if &branch.branch_index == index {
				return branch.new_range.as_ref()
					.map(|gc| {
						let linear_migrate = (bi.clone(), gc.clone());
						MemoryOnlyLinear::is_in_migrate(linear_index, &linear_migrate)
					}).unwrap_or(true);
			}
			if &branch.branch_index < &index {
				break;
			}
		}
		false
	}

	fn migrate(&mut self, mig: &mut Self::Migrate) -> UpdateResult<()> {
		// This is basis from old gc, it does not do indexing change as it could
		// be possible. TODO start migrate too (mig.0)
		let mut result = UpdateResult::Unchanged;
		let start_len = self.branches.len();
		let mut to_remove = Vec::new(); // if switching to hash map retain usage is way better.
		let mut gc_iter = mig.1.changes.iter_mut().rev();
		let mut branch_iter = self.branches.iter_mut().enumerate().rev();
		let mut o_gc = gc_iter.next();
		let mut o_branch = branch_iter.next();
		// TODO ref mut remove (not previously works fine with as ref so refact serialize to inner mut.
		while let (Some(ref mut gc), Some((index, branch))) = (o_gc.as_mut(), o_branch.as_mut()) {
			if gc.branch_index == branch.branch_index {
				if let Some(gc) = gc.new_range.as_mut() {
					match branch.history.gc(gc) {
						UpdateResult::Unchanged => (),
						UpdateResult::Changed(_) => { result = UpdateResult::Changed(()); },
						UpdateResult::Cleared(_) => to_remove.push(*index),
					}
				} else {
					to_remove.push(*index);
				}
				o_gc = gc_iter.next();
				o_branch = branch_iter.next();
			} else if gc.branch_index < branch.branch_index {
				o_branch = branch_iter.next();
			} else {
				o_gc = gc_iter.next();
			}
		}

		for i in to_remove.into_iter() {
			self.branches.remove(i);
		}

		if self.branches.len() == 0 {
			result = UpdateResult::Cleared(());
		} else if self.branches.len() != start_len {
			result = UpdateResult::Changed(());
		}

		result
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32> + SubAssign<BI>,
	V: Clone + Eq,
> InMemoryValue<V> for MemoryOnly<I, BI, V> {
	fn get_mut(&mut self, at: &Self::SE) -> Option<&mut V> {
		let (branch_index, index) = at.latest();
		for branch in self.branches.iter_mut().rev() {
			if &branch.branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.history.get_mut(&index);
			}
			if &branch.branch_index < branch_index {
				break;
			}
		}
		None
	}

	fn set_mut(&mut self, value: V, at: &Self::SE) -> UpdateResult<Option<V>> {
		let (branch_index, index) = at.latest();
		let mut insert_at = self.branches.len();
		for (iter_index, branch) in self.branches.iter_mut().enumerate().rev() {
			if &branch.branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.history.set_mut(value, &index);
			}
			if &branch.branch_index < branch_index {
				insert_at = iter_index + 1;
				break;
			} else {
				insert_at = iter_index;
			}
		}
		let branch = MemoryOnlyBranch::new(value, at);
		if insert_at == self.branches.len() {
			self.branches.push(branch);
		} else {
			self.branches.insert(insert_at, branch);
		}
		UpdateResult::Changed(None)
	}
}

impl MemoryOnly<u32, u32, Option<Vec<u8>>> {
	/// Temporary function to get occupied stage.
	/// TODO replace by heapsizeof
	pub fn temp_size(&self) -> usize {
		let mut size = 0;
		for b in self.branches.iter() {
			size += 4; // branch index (using u32 as usize)
			size += b.history.temp_size();
		}
		size
	}
}

#[cfg(test)]
mod test {
	use super::*;
	use crate::historied::tree_management::test::test_states;

	#[test]
	fn test_set_get() {
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _
		let mut states = test_states();
		let mut item: MemoryOnly<u32, u32, u32> = Default::default();

		for i in 0..6 {
			assert_eq!(item.get(&states.query_plan(i)), None);
		}

		// setting value respecting branch build order
		for i in 1..6 {
			item.set(i, &states.unchecked_latest_at(i).unwrap());
		}

		for i in 1..6 {
			assert_eq!(item.get_ref(&states.query_plan(i)), Some(&i));
		}

		let ref_1 = states.query_plan(1);
		assert_eq!(Some(false), states.branch_state_mut(&1).map(|ls| ls.drop_state()));

		let ref_1_bis = states.query_plan(1);
		assert_eq!(item.get(&ref_1), Some(1));
		assert_eq!(item.get(&ref_1_bis), None);
		item.set(11, &states.unchecked_latest_at(1).unwrap());
		// lazy linear clean of drop state on insert
		assert_eq!(item.get(&ref_1), Some(11));
		assert_eq!(item.get(&ref_1_bis), Some(11));

		item = Default::default();

		// need fresh state as previous modification leaves unattached branches
		let mut states = test_states();
		// could rand shuffle if rand get imported later.
		let disordered = [
			[1,2,3,5,4],
			[2,5,1,3,4],
			[5,3,2,4,1],
		];
		for r in disordered.iter() {
			for i in r {
				item.set(*i, &states.unchecked_latest_at(*i).unwrap());
			}
			for i in r {
				assert_eq!(item.get_ref(&states.query_plan(*i)), Some(i));
			}
		}
	}
}
