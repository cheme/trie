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
	fn get_branch(self, index: &I) -> Option<Self::Branch>;

	/// Get the last state, inclusive.
	fn last_index(self) -> I;

	/// Iterator over the branch states.
	fn iter(self) -> Self::Iter;
}

/// Trait defining a state for querying or modifying a branch.
/// This is therefore the representation of a branch state.
pub trait BranchStateTrait<I> {
	/// Get state for node at a given index.
	fn exists(&self, i: &I) -> bool;

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

impl<I: Default + Ord, BI> Default for RangeSet<I, BI> {
	fn default() -> Self {
		RangeSet {
			storage: BTreeMap::new(),
			last_index: I::default(),
			composite_treshold: I::default(),
		}
	}
}

impl<I: Clone + Default + SubAssign<usize> + AddAssign<usize> + Ord, BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone> RangeSet<I, BI> {
	/// Return anchor index for this branch history:
	/// - same index as input if the branch was modifiable
	/// - new index in case of branch range creation
	pub fn add_state(
		&mut self,
		branch_index: I,
		number: BI,
	) -> Option<I> {
		if branch_index < self.composite_treshold {
			return None;
		}
		let mut create_new = false;
		if branch_index == I::default()  {
			create_new = true;
		} else {
			let branch_state = self.storage.get_mut(&branch_index)
				.expect("Inconsistent state on new block");
			if branch_state.can_append && branch_state.can_add(&number) {
				branch_state.add_state();
			} else {
				if !branch_state.can_fork(&number) {
					return None;
				} else {
					create_new = true;
				}
			}
		}

		Some(if create_new {
			self.last_index += 1;

			let state = BranchState::new(number, branch_index);
			self.storage.insert(self.last_index.clone(), state);
			self.last_index.clone()
		} else {
			branch_index
		})
	}

	/// Get the branch reference for a given branch index if it exists.
	pub fn state_ref(&self, mut branch_index: I) -> BranchRanges<I, BI> {
		let mut history = Vec::new();
		while branch_index > self.composite_treshold {
			if let Some(branch) = self.storage.get(&branch_index) {
				let branch_ref = branch.state_ref();
				// vecdeque would be better suited
				history.insert(0, BranchStatesRef {
					state: branch_ref,
					branch_index: branch_index.clone(),
				});
				branch_index = branch.parent_branch_index.clone();
			} else {
				break;
			}
		}
		BranchRanges {
			history
		}
	}

	/// Return anchor index for this branch history:
	/// - same index as input if branch is not empty
	/// - parent index if branch is empty
	pub fn drop_state(
		&mut self,
		branch_index: &I,
	) -> Option<I> {
		let mut do_remove = None;
		match self.storage.get_mut(branch_index) {
			Some(branch_state) => {
				if branch_state.drop_state() {
					do_remove = Some(branch_state.parent_branch_index.clone());
				}
			},
			None => return None,
		}

		Some(if let Some(parent_index) = do_remove {
			self.storage.remove(branch_index);
			parent_index
		} else {
			branch_index.clone()
		})
	}

	pub fn branch_state(&self, branch_index: &I) -> Option<&BranchState<I, BI>> {
		self.storage.get(branch_index)
	}

	pub fn branch_state_mut(&mut self, branch_index: &I) -> Option<&mut BranchState<I, BI>> {
		self.storage.get_mut(branch_index)
	}

	/// this function can go into deep recursion with full scan, it indicates
	/// that the tree model use here should only be use for small data or
	/// tests.
	pub fn apply_drop_state(&mut self, branch_index: &I, node_index: &BI) {
		let mut to_delete = Vec::new();
		let mut remove = false;
		if let Some(branch) = self.storage.get_mut(branch_index) {
			while &branch.state.end > node_index {
				if branch.drop_state() {
					remove = true;
					break;
				}
			}
		}
		if remove {
			self.storage.remove(branch_index);
		}
		for (i, s) in self.storage.iter() {
			if &s.parent_branch_index == branch_index && &s.state.start >= node_index {
				to_delete.push(i.clone());
			}
		}
		for i in to_delete.into_iter() {
			self.apply_drop_state(&i, node_index)
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
pub struct BranchRanges<I, BI> {
	history: Vec<BranchStatesRef<I, BI>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchStatesRef<I, BI> {
	// TODO rename to index
	pub branch_index: I,
	pub state: BranchStateRef<BI>,
}

impl<'a, I: Default + Eq + Ord + Clone, BI: SubAssign<usize> + Ord + Clone> BranchesStateTrait<I, BI> for &'a BranchRanges<I, BI> {
	type Branch = &'a BranchStateRef<BI>;
	type Iter = BranchRangesIter<'a, I, BI>;

	fn get_branch(self, i: &I) -> Option<Self::Branch> {
		for (b, bi) in self.iter() {
			if &bi == i {
				return Some(b);
			} else if &bi < i {
				break;
			}
		}
		None
	}

	fn last_index(self) -> I {
		let l = self.history.len();
		if l > 0 {
			self.history[l - 1].branch_index.clone()
		} else {
			I::default()
		}
	}

	fn iter(self) -> Self::Iter {
		BranchRangesIter(self, self.history.len())
	}
}

/// Iterator, contains index of last inner struct.
pub struct BranchRangesIter<'a, I, BI>(&'a BranchRanges<I, BI>, usize);

impl<'a, I: Clone, BI> Iterator for BranchRangesIter<'a, I, BI> {
	type Item = (&'a BranchStateRef<BI>, I);

	fn next(&mut self) -> Option<Self::Item> {
		if self.1 > 0 {
			Some((
				&(self.0).history[self.1 - 1].state,
				(self.0).history[self.1 - 1].branch_index.clone(),
			))
		} else {
			None
		}
	}
}

impl<I: Ord + SubAssign<usize> + Clone> BranchStateTrait<I> for BranchStateRef<I> {

	fn exists(&self, i: &I) -> bool {
		i >= &self.start && i < &self.end
	}

	fn last_index(&self) -> I {
		let mut r = self.end.clone();
		// underflow should not happen as long as branchstateref are not allowed to be empty.
		r -= 1;
		r
	}
}



impl<'a, I, B: BranchStateTrait<I>> BranchStateTrait<I> for &'a B {

	fn exists(&self, i: &I) -> bool {
		(*self).exists(i)
	}

	fn last_index(&self) -> I {
		(*self).last_index()
	}

}
/*
/// u64 is use a a state target so it is implemented as
/// a upper bound.
impl<'a, I: Clone + Ord> BranchStateTrait<I> for I {

	fn exists(&self, i: &I) -> bool {
		i <= self
	}

	fn last_index(&self) -> I {
		self.clone()
	}

}
*/
impl<I: Default, BI: Default + AddAssign<usize>> Default for BranchState<I, BI> {

	// initialize with one element
	fn default() -> Self {
		let mut end = BI::default();
		end += 1;
		BranchState {
			state: BranchStateRef {
				start: Default::default(),
				end,
			},
			can_append: true,
			parent_branch_index: Default::default(),
		}
	}
}

impl<I, BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone> BranchState<I, BI> {

	pub fn state_ref(&self) -> BranchStateRef<BI> {
		self.state.clone()
	}

	pub fn new(offset: BI, parent_branch_index: I) -> Self {
		let mut end = offset.clone();
		end += 1;
		BranchState {
			state: BranchStateRef {
				start: offset,
				end,
			},
			can_append: true,
			parent_branch_index,
		}
	}

	/// Return true if you can add this index.
	pub fn can_add(&self, index: &BI) -> bool {
		index == &self.state.end
	}

 	pub fn can_fork(&self, index: &BI) -> bool {
		index <= &self.state.end && index >= &self.state.start
	}
 
	pub fn add_state(&mut self) -> bool {
		if self.can_append {
			self.state.end += 1;
			true
		} else {
			false
		}
	}

	/// Return true if resulting branch is empty.
	fn drop_state(&mut self) -> bool {
		if self.state.end > self.state.start {
			self.state.end -= 1;
			self.can_append = false;
			if self.state.end == self.state.start {
				true
			} else {
				false
			}
		} else {
			true
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	fn test_states() -> RangeSet<usize, usize> {
		let mut states = RangeSet::default();
		assert_eq!(states.add_state(0, 1), Some(1));
		// root branching.
		assert_eq!(states.add_state(1, 1), Some(2));
		assert_eq!(Some(true), states.branch_state_mut(&1).map(|ls| ls.add_state()));
		assert_eq!(states.add_state(1, 2), Some(3));
		assert_eq!(states.add_state(1, 2), Some(4));
		assert_eq!(states.add_state(1, 1), Some(5));
		assert_eq!(states.add_state(2, 2), Some(2));
		assert_eq!(Some(true), states.branch_state_mut(&1).map(|ls| ls.add_state()));
		assert_eq!(Some(1), states.drop_state(&1));
		// cannot create when dropped happen on branch
		assert_eq!(Some(false), states.branch_state_mut(&1).map(|ls| ls.add_state()));

		assert!(states.branch_state(&1).unwrap().state.exists(&1));
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
		assert_eq!(Some(false), states.branch_state_mut(&1).map(|ls| ls.drop_state()));
		assert!(states.branch_state(&3).unwrap().state.exists(&2));
		assert!(states.branch_state(&4).unwrap().state.exists(&2));
		states.apply_drop_state(&1, &1);
		assert_eq!(states.branch_state(&3), None);
		assert_eq!(states.branch_state(&4), None);
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
				state: BranchStateRef { start: 2, end: 3 },
			},
		];
		assert_eq!(states.state_ref(3).history, ref_3);

		let mut states = states;

		assert_eq!(states.add_state(1, 1), Some(6));
		let ref_6 = vec![
			BranchStatesRef {
				branch_index: 1,
				state: BranchStateRef { start: 0, end: 1 },
			},
			BranchStatesRef {
				branch_index: 6,
				state: BranchStateRef { start: 1, end: 2 },
			},
		];
		assert_eq!(states.state_ref(6).history, ref_6);

		states.composite_treshold = 1;
		let mut ref_6 = ref_6;
		ref_6.remove(0);
		assert_eq!(states.state_ref(6).history, ref_6);
	}

/* TODO enable it on tree history
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
*/
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
