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

//! Implementation of state management for tree like
//! state.
//!
//! State changes are limited so resulting tree is rather unbalance.
//! This is best when there is not to many branch (fork)

use crate::rstd::ops::{AddAssign, SubAssign, Range};
use crate::rstd::BTreeMap;

/// Trait defining a state for querying or modifying a tree.
/// This is a collection of branches index, corresponding
/// to a tree path.
pub trait BranchesStateTrait<I, BI> {
	type Branch: BranchStateTrait<BI>;
	type Iter: Iterator<Item = (Self::Branch, I)>;

	/// Get branch state for node at a given index.
	fn get_branch(self, index: I) -> Option<Self::Branch>;

	/// Get the last state, inclusive.
	fn last_index(self) -> I;

	/// Iterator over the branch states.
	fn iter(self) -> Self::Iter;
}

/// Trait defining a state for querying or modifying a branch.
/// This is therefore the representation of a branch state.
pub trait BranchStateTrait<I> {
	/// Get state for node at a given index.
	fn exists(&self, i: I) -> bool;

	/// Get the last index for the state, inclusive.
	fn last_index(&self) -> I;
}

/// Stored states for a branch, it contains branch reference information,
/// structural information (index of parent branch) and fork tree building
/// information (is branch appendable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchState<I, BI> {
	state: BranchStateRef<BI>,
	can_append: bool,
	parent_branch_index: I,
}

/// This is a simple range, end non inclusive.
/// TODO type alias or use ops::Range? see next todo?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchStateRef<I> {
	// TODO rewrite this to use as single linear index?
	// we could always start at 0 but the state could not
	// be compared between branch which is sad.
	// Still start info is not that relevant, this is probably
	// removable.
	pub start: I,
	pub end: I,
}


/// Current branches range definition, indexed by branch
/// numbers.
///
/// New branches index are using `last_index`.
///
/// Also acts as a cache, storage can store
/// unknown db value as `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeSet<I, BI> {
	storage: BTreeMap<I, BranchState<I, BI>>,
	last_index: I,
	/// treshold for possible node value, correspond
	/// roughly to last cannonical block branch index.
	/// If at default state value, we go through simple storage.
	composite_treshold: I,
}

impl<I: Default + SubAssign<usize> + Ord, BI> Default for RangeSet<I, BI> {
	fn default() -> Self {
		let mut composite_treshold = I::default();
		composite_treshold -= 1;
		RangeSet {
			storage: BTreeMap::new(),
			last_index: I::default(),
			composite_treshold,
		}
	}
}

#[derive(Clone, Default, Debug)]
/// State needed for query updates.
/// That is a subset of the full branch ranges set.
///
/// Values are ordered by branch_ix,
/// and only a logic branch path should be present.
///
/// Note that an alternative could be a pointer to the full state
/// a branch index corresponding to the leaf for the fork.
/// Here we use an in memory copy of the path because it seems
/// to fit query at a given state with multiple operations
/// (block processing), that way we iterate on a vec rather than 
/// hoping over linked branches.
/// TODO small vec that ??
pub struct BranchRanges<I, BI>(Vec<BranchStatesRef<I, BI>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchStatesRef<I, BI> {
	// TODO rename to index
	pub branch_index: I,
	pub state: BranchStateRef<BI>,
}

impl<'a, I: Default + Eq + Ord + Clone, BI: SubAssign<usize> + Ord + Clone> BranchesStateTrait<I, BI> for &'a BranchRanges<I, BI> {
	type Branch = &'a BranchStateRef<BI>;
	type Iter = BranchRangesIter<'a, I, BI>;

	fn get_branch(self, i: I) -> Option<Self::Branch> {
		for (b, bi) in self.iter() {
			if bi == i {
				return Some(b);
			} else if bi < i {
				break;
			}
		}
		None
	}

	fn last_index(self) -> I {
		let l = self.0.len();
		if l > 0 {
			self.0[l - 1].branch_index.clone()
		} else {
			I::default()
		}
	}

	fn iter(self) -> Self::Iter {
		BranchRangesIter(self, self.0.len())
	}
}

/// Iterator, contains index of last inner struct.
pub struct BranchRangesIter<'a, I, BI>(&'a BranchRanges<I, BI>, usize);

impl<'a, I: Clone, BI> Iterator for BranchRangesIter<'a, I, BI> {
	type Item = (&'a BranchStateRef<BI>, I);

	fn next(&mut self) -> Option<Self::Item> {
		if self.1 > 0 {
			Some((
				&(self.0).0[self.1 - 1].state,
				(self.0).0[self.1 - 1].branch_index.clone(),
			))
		} else {
			None
		}
	}
}

impl<'a, I: Ord + SubAssign<usize> + Clone> BranchStateTrait<I> for &'a BranchStateRef<I> {

	fn exists(&self, i: I) -> bool {
		i >= self.start && i < self.end
	}

	fn last_index(&self) -> I {
		let mut r = self.end.clone();
		// underflow should not happen as long as branchstateref are not allowed to be empty.
		r -= 1;
		r
	}

}

/// u64 is use a a state target so it is implemented as
/// a upper bound.
impl<'a, I: Clone + Ord> BranchStateTrait<I> for I {

	fn exists(&self, i: I) -> bool {
		&i <= self
	}

	fn last_index(&self) -> I {
		self.clone()
	}

}

impl<I: Default, BI: Default + AddAssign<usize>> Default for BranchState<I, BI> {
	// initialize with one element
	fn default() -> Self {
		let mut end = BI::default();
		end += 1;
		BranchState {
			state: BranchStateRef{
				start: Default::default(),
				end,
			},
			can_append: true,
			parent_branch_index: Default::default(),
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	fn test_states() -> TestStates {
		let mut states = TestStates::default();
		assert_eq!(states.create_branch(1, 0, None), Some(1));
		// root branching.
		assert_eq!(states.create_branch(1, 0, None), Some(2));
		assert_eq!(Some(true), states.branch_state_mut(1).map(|ls| ls.add_state()));
		assert_eq!(states.create_branch(2, 1, None), Some(3));
		assert_eq!(states.create_branch(1, 1, Some(0)), Some(5));
		assert_eq!(states.create_branch(1, 1, Some(2)), None);
		assert_eq!(Some(true), states.branch_state_mut(1).map(|ls| ls.add_state()));
		assert_eq!(Some(Some(2)), states.branch_state_mut(1).map(|ls| ls.drop_state()));
		// cannot create when dropped happen on branch
		assert_eq!(Some(false), states.branch_state_mut(1).map(|ls| ls.add_state()));

		assert!(states.get(1, 1));
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _

		states
	}

	#[test]
	fn test_remove_attached() {
		let mut states = test_states();
		assert_eq!(Some(Some(1)), states.branch_state_mut(1).map(|ls| ls.drop_state()));
		assert!(states.get(3, 0));
		assert!(states.get(4, 0));
		states.apply_drop_state(1, 1);
		assert!(!states.get(3, 0));
		assert!(!states.get(4, 0));
	}

	#[test]
	fn test_state_refs() {
		let states = test_states();
		let ref_3 = vec![
			BranchStatesRef {
				branch_index: 1,
				state: BranchStateRef { start: 0, end: 2 },
			},
			BranchStatesRef {
				branch_index: 3,
				state: BranchStateRef { start: 0, end: 1 },
			},
		];
		assert_eq!(*states.state_ref(3).history, ref_3);

		let mut states = states;

		assert_eq!(states.create_branch(1, 1, Some(0)), Some(6));
		let ref_6 = vec![
			BranchStatesRef {
				branch_index: 1,
				state: BranchStateRef { start: 0, end: 1 },
			},
			BranchStatesRef {
				branch_index: 6,
				state: BranchStateRef { start: 0, end: 1 },
			},
		];
		assert_eq!(*states.state_ref(6).history, ref_6);

		states.valid_treshold = 3;
		let mut ref_6 = ref_6;
		ref_6.remove(0);
		assert_eq!(*states.state_ref(6).history, ref_6);
	}

	#[test]
	fn test_set_get() {
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _
		let states = test_states();
		let mut item: History<u64> = Default::default();

		for i in 0..6 {
			assert_eq!(item.get(&states.state_ref(i)), None);
		}

		// setting value respecting branch build order
		for i in 1..6 {
			item.set(&states.state_ref(i), i);
		}

		for i in 1..6 {
			assert_eq!(item.get(&states.state_ref(i)), Some(&i));
		}

		let mut ref_3 = states.state_ref(3);
		ref_3.limit_branch(1, None);
		assert_eq!(item.get(&ref_3), Some(&1));

		let mut ref_1 = states.state_ref(1);
		ref_1.limit_branch(1, Some(0));
		assert_eq!(item.get(&ref_1), None);
		item.set(&ref_1, 11);
		assert_eq!(item.get(&ref_1), Some(&11));

		item = Default::default();

		// could rand shuffle if rand get imported later.
		let disordered = [
			[1,2,3,5,4],
			[2,5,1,3,4],
			[5,3,2,4,1],
		];
		for r in disordered.iter() {
			for i in r {
				item.set(&states.state_ref(*i), *i);
			}
			for i in r {
				assert_eq!(item.get(&states.state_ref(*i)), Some(i));
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
		let mut item: History<u64> = Default::default();
		// setting value respecting branch build order
		for i in 1..6 {
			item.set(&states.state_ref(i), i);
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
		assert_eq!(item1.get(&states.state_ref(1)), None);
		for a in action.iter().skip(1) {
			if a.1 {
				assert_eq!(item1.get(&states.state_ref(a.0)), Some(&a.0));
			} else {
				assert_eq!(item1.get(&states.state_ref(a.0)), None);
			}
		}
	}
*/
}
