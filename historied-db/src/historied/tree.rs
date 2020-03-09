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
use crate::historied::linear::{MemoryOnly as MemoryOnlyLinear, Latest, LinearState};
use crate::historied::tree_management::{ForkPlan, BranchesContainer, TreeGc};
use crate::rstd::ops::{AddAssign, SubAssign, Range};

#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct MemoryOnly<I, BI, V>(Vec<MemoryOnlyBranch<I, BI, V>>);

impl<I, BI, V> Default for MemoryOnly<I, BI, V> {
	fn default() -> Self {
		MemoryOnly(Vec::new())
	}
}

#[derive(Debug, Clone)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct MemoryOnlyBranch<I, BI, V> {
	branch_index: I,
	history: MemoryOnlyLinear<V, BI>,
}

impl<I: Clone, BI: Clone, V> MemoryOnlyBranch<I, BI, V> {
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

impl<I: Clone, BI: Clone, V> MemoryOnly<I, BI, V> {
	pub fn new(value: V, state: &Latest<(I, BI)>) -> Self {
		let mut v = Vec::new();
		v.push(MemoryOnlyBranch::new(value, state));
		MemoryOnly(v)
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<usize>, // TODO consider subassing usize or minus one trait...
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
		self.0.is_empty()
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<usize>,
	V: Clone,
> MemoryOnly<I, BI, V> {
	fn get_ref(&self, at: &<Self as ValueRef<V>>::S) -> Option<&V> {
		let mut index = self.0.len();
		// note that we expect branch index to be linearily set
		// along a branch (no state containing unordered branch_index
		// and no history containing unorderd branch_index).
		if index == 0 {
			return None;
		}

		for (state_branch_range, state_branch_index) in at.iter() {
			while index > 0 {
				let branch_index = &self.0[index - 1].branch_index;
				if branch_index < &state_branch_index {
					break;
				} else if branch_index == &state_branch_index {
					// TODO add a lower bound check (maybe debug_assert it only).
					let mut upper_bound = state_branch_range.end.clone();
					upper_bound -= 1;
					if let Some(result) = self.0[index - 1].history.get_ref(&upper_bound) {
						return Some(result)
					}
				}
				index -= 1;
			}
		}
		None
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<usize> + SubAssign<BI>,
	V: Clone + Eq,
> Value<V> for MemoryOnly<I, BI, V> {
	type SE = Latest<(I, BI)>;
	type GC = TreeGc<I, BI, V>;
	type Migrate = TreeGc<I, BI, V>;

	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()> {
		let (branch_index, index) = at.latest();
		let mut insert_at = self.0.len();
		for (iter_index, branch) in self.0.iter_mut().enumerate().rev() {
			if &branch.branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.history.set(value, &index);
			}
			if &branch.branch_index < branch_index {
				insert_at = iter_index + 1;
				break;
			}
		}
		let branch = MemoryOnlyBranch::new(value, at);
		if insert_at == self.0.len() {
			self.0.push(branch);
		} else {
			self.0.insert(insert_at, branch);
		}
		UpdateResult::Changed(())
	}

	fn discard(&mut self, at: &Self::SE) -> UpdateResult<Option<V>> {
		let (branch_index, index) = at.latest();
		for branch in self.0.iter_mut().rev() {
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

	fn gc(&mut self, gc: &Self::GC) -> UpdateResult<()> {
		unimplemented!("TODO needs impl");
	}

	fn migrate(&mut self, mig: &Self::Migrate) -> UpdateResult<()> {
		unimplemented!()
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<usize> + SubAssign<BI>,
	V: Clone + Eq,
> InMemoryValue<V> for MemoryOnly<I, BI, V> {
	fn get_mut(&mut self, at: &Self::SE) -> Option<&mut V> {
		let (branch_index, index) = at.latest();
		for branch in self.0.iter_mut().rev() {
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
		let mut item: MemoryOnly<usize, usize, usize> = Default::default();

		for i in 0..6 {
			assert_eq!(item.get(&states.query_plan(i)), None);
		}

		// setting value respecting branch build order
		for i in 1..6 {
			item.set(i, &states.latest_at(i).unwrap());
		}

		for i in 1..6 {
			assert_eq!(item.get_ref(&states.query_plan(i)), Some(&i));
		}

		let ref_1 = states.query_plan(1);
		assert_eq!(Some(false), states.branch_state_mut(&1).map(|ls| ls.drop_state()));

		let ref_1_bis = states.query_plan(1);
		assert_eq!(item.get(&ref_1), Some(1));
		assert_eq!(item.get(&ref_1_bis), None);
		item.set(11, &states.latest_at(1).unwrap());
		// lazy linear clean of drop state on insert
		assert_eq!(item.get(&ref_1), Some(11));
		assert_eq!(item.get(&ref_1_bis), Some(11));

		item = Default::default();

		// need fresh state as previous modification leaves unattached branches
		let states = test_states();
		// could rand shuffle if rand get imported later.
		let disordered = [
			[1,2,3,5,4],
			[2,5,1,3,4],
			[5,3,2,4,1],
		];
		for r in disordered.iter() {
			for i in r {
				item.set(*i, &states.latest_at(*i).unwrap());
			}
			for i in r {
				assert_eq!(item.get_ref(&states.query_plan(*i)), Some(i));
			}
		}
	}

/* TODO move in tree
	#[test]
	fn test_gc() {
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _
		let states = test_states();
		let mut item: MemoryOnly<usize, usize, usize> = Default::default();
		// setting value respecting branch build order
		for i in 1..6 {
			item.set(&states.query_plan(i), i);
		}

		let mut states1 = states.branches.clone();
		let action = [(1, true), (2, false), (3, false), (4, true), (5, false)];
		for a in action.iter() {
			if !a.1 {
				states1.remove(&a.0);
			}
		}
		// makes invalid tree (detaches 4)
		states1.get_mut(&1).map(|br| br.state.len = 1);
		let states1: BTreeMap<_, _> = states1.iter().map(|(k,v)| (k, v.branch_ref())).collect();
		let mut item1 = item.clone();
		item1.gc(states1.iter().map(|(k, v)| ((&v.state, None), **k)).rev());
		assert_eq!(item1.get(&states.query_plan(1)), None);
		for a in action.iter().skip(1) {
			if a.1 {
				assert_eq!(item1.get(&states.query_plan(a.0)), Some(&a.0));
			} else {
				assert_eq!(item1.get(&states.query_plan(a.0)), None);
			}
		}
	}
*/

}
