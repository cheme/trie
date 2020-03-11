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
use crate::historied::linear::{Latest, LinearGC};
use crate::{Management, ManagementRef, Migrate, ForkableManagement};

/// Trait defining a state for querying or modifying a tree.
/// This is a collection of branches index, corresponding
/// to a tree path.
pub trait BranchesContainer<I, BI> {
	type Branch: BranchContainer<BI>;
	type Iter: Iterator<Item = (Self::Branch, I)>;

	/// Get branch state for node at a given index.
	fn get_branch(self, index: &I) -> Option<Self::Branch>;

	/// Get the last branch, inclusive.
	fn last_index(self) -> I;

	/// Iterator over the branch states in query order
	/// (more recent first).
	fn iter(self) -> Self::Iter;
}

/// Trait defining a state for querying or modifying a branch.
/// This is therefore the representation of a branch state.
pub trait BranchContainer<I> {
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
	state: BranchRange<BI>,
	can_append: bool,
	parent_branch_index: I,
}

/// This is a simple range, end non inclusive.
/// TODO type alias or use ops::Range? see next todo?
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchRange<I> {
	// TODO rewrite this to use as single linear index?
	// we could always start at 0 but the state could not
	// be compared between branch which is sad.
	// Still start info is not that relevant, this is probably
	// removable.
	pub start: I,
	pub end: I,
}


/// Full state of current tree layout.
/// It contains all layout inforamation for branches
/// states.
/// Branches are indexed by a sequential index.
/// Element of branches are indexed by a secondary
/// sequential indexes.
///
/// New branches index are defined by using `last_index`.
///
/// Also acts as a cache, storage can store
/// unknown db value as `None`.
///
/// NOTE that the single element branch at default index
/// containing the default branch index element does always
/// exist by convention.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct Tree<I, BI> {
	storage: BTreeMap<I, BranchState<I, BI>>,
	last_index: I,
	/// treshold for possible node value, correspond
	/// roughly to last cannonical block branch index.
	/// If at default state value, we go through simple storage.
	/// TODO move in tree management??
	composite_treshold: I,
	// TODO some strategie to close a long branch that gets
	// behind multiple fork? This should only be usefull
	// for high number of modification, small number of
	// fork. The purpose is to avoid history where meaningfull
	// value is always in a low number branch behind a few fork.
	// A longest branch pointer per history is also a viable
	// strategy and avoid fragmenting the history to much.
}

impl<I: Default + Ord, BI> Default for Tree<I, BI> {
	fn default() -> Self {
		Tree {
			storage: BTreeMap::new(),
			last_index: I::default(),
			composite_treshold: I::default(),
		}
	}
}


#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct TreeManagement<H, I, BI, V> {
	tree: Tree<I, BI>,
	mapping: crate::rstd::BTreeMap<H, (I, BI)>,
	touched_gc: bool,
	current_gc: TreeGc<I, BI, V>,
	neutral_element: Option<V>,
	last_in_use_index: (I, BI), // TODO rename to last inserted as we do not rebase on query
}

impl<H: Ord, I: Default + Ord, BI: Default, V> Default for TreeManagement<H, I, BI, V> {
	fn default() -> Self {
		TreeManagement {
			tree: Tree::default(),
			mapping: crate::rstd::BTreeMap::new(),
			neutral_element: None,
			touched_gc: false,
			current_gc: Default::default(),
			last_in_use_index: Default::default(),
		}
	}
}

impl<
	H,
	I,
	BI,
	V,
> TreeManagement<H, I, BI, V> {
	pub fn define_neutral_element(mut self, n: V) -> Self {
		self.neutral_element = Some(n);
		self
	}
}

impl<
	I: Clone + Default + SubAssign<usize> + AddAssign<usize> + Ord,
	BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone,
> Tree<I, BI> {
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

	pub fn latest_at(&self, branch_index : I) -> Option<Latest<(I, BI)>> {
		self.storage.get(&branch_index).map(|branch| {
			let mut end = branch.state.end.clone();
			end -= 1;
			Latest::unchecked_latest((branch_index, end))
		})
	}
	
	/// TODO doc & switch to &I
	pub fn query_plan_at(&self, (branch_index, mut index) : (I, BI)) -> ForkPlan<I, BI> {
		// make index exclusive
		index += 1;
		self.query_plan_inner(branch_index, Some(index))
	}
	/// TODO doc & switch to &I
	pub fn query_plan(&self, branch_index: I) -> ForkPlan<I, BI> {
		self.query_plan_inner(branch_index, None)
	}

	fn query_plan_inner(&self, mut branch_index: I, mut parent_fork_branch_index: Option<BI>) -> ForkPlan<I, BI> {
		let mut history = Vec::new();
		while branch_index > self.composite_treshold {
			if let Some(branch) = self.storage.get(&branch_index) {
				let branch_ref = if let Some(end) = parent_fork_branch_index.take() {
					branch.query_plan_to(end)
				} else {
					branch.query_plan()
				};
				parent_fork_branch_index = Some(branch_ref.start.clone());
				if branch_ref.end > branch_ref.start {
					// vecdeque would be better suited
					history.insert(0, BranchPlan {
						state: branch_ref,
						branch_index: branch_index.clone(),
					});
				}
				branch_index = branch.parent_branch_index.clone();
			} else {
				break;
			}
		}
		ForkPlan {
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

#[derive(Clone, Debug, Eq, PartialEq)]
/// Query plane needed for operation for a given
/// fork.
/// This is a subset of the full branch set definition.
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
/// TODO add I treshold (everything valid starting at this one)?
pub struct ForkPlan<I, BI> {
	history: Vec<BranchPlan<I, BI>>,
}

impl<I, BI> Default for ForkPlan<I, BI> {
	fn default() -> Self {
		ForkPlan{ history: Vec::new() }
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Query plan element for a single branch.
pub struct BranchPlan<I, BI> {
	// TODO rename to index
	pub branch_index: I,
	pub state: BranchRange<BI>,
}

impl<'a, I: Default + Eq + Ord + Clone, BI: SubAssign<usize> + Ord + Clone> BranchesContainer<I, BI> for &'a ForkPlan<I, BI> {
	type Branch = &'a BranchRange<BI>;
	type Iter = ForkPlanIter<'a, I, BI>;

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
		ForkPlanIter(self, self.history.len())
	}
}

/// Iterator, contains index of last inner struct.
pub struct ForkPlanIter<'a, I, BI>(&'a ForkPlan<I, BI>, usize);

impl<'a, I: Clone, BI> Iterator for ForkPlanIter<'a, I, BI> {
	type Item = (&'a BranchRange<BI>, I);

	fn next(&mut self) -> Option<Self::Item> {
		if self.1 > 0 {
			self.1 -= 1;
			Some((
				&(self.0).history[self.1].state,
				(self.0).history[self.1].branch_index.clone(),
			))
		} else {
			None
		}
	}
}

impl<I: Ord + SubAssign<usize> + Clone> BranchContainer<I> for BranchRange<I> {

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



impl<'a, I, B: BranchContainer<I>> BranchContainer<I> for &'a B {

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
impl<'a, I: Clone + Ord> BranchContainer<I> for I {

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
			state: BranchRange {
				start: Default::default(),
				end,
			},
			can_append: true,
			parent_branch_index: Default::default(),
		}
	}
}

impl<I, BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone> BranchState<I, BI> {

	pub fn query_plan(&self) -> BranchRange<BI> {
		self.state.clone()
	}

	pub fn query_plan_to(&self, end: BI) -> BranchRange<BI> {
		debug_assert!(self.state.end >= end);
		BranchRange {
			start: self.state.start.clone(),
			end,
		}
	}

	pub fn new(offset: BI, parent_branch_index: I) -> Self {
		let mut end = offset.clone();
		end += 1;
		BranchState {
			state: BranchRange {
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
		index <= &self.state.end && index > &self.state.start
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
	pub fn drop_state(&mut self) -> bool {
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

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct BranchGC<I, BI, V> {
	pub branch_index: I,
	/// A new start - end limit for the branch or a removed
	/// branch.
	pub new_range: Option<LinearGC<BI, V>>,
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct TreeGc<I, BI, V> {
	/// Every modified branch.
	/// Ordered by branch index.
	pub changes: Vec<BranchGC<I, BI, V>>,
	// TODO is also in every lineargc of branchgc.
	pub neutral_element: Option<V>,
	// TODO add the key elements (as option to trigger registration or not).
}

impl<I, BI, V> Default for TreeGc<I, BI, V> {
	fn default() -> Self {
		TreeGc {
			changes: Vec::new(),
			neutral_element: None,
		}
	}
}

impl<I, BI, V> TreeGc<I, BI, V> {
	fn applied(&mut self, gc_applied: TreeGc<I, BI, V>) {
		unimplemented!("TODO run a delta to keep possible updates in between");
	}
}

impl<
	H: Ord,
	I: Clone + Default + SubAssign<usize> + AddAssign<usize> + Ord,
	BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone,
	V: Clone,
> ManagementRef<H> for TreeManagement<H, I, BI, V> {
	type S = ForkPlan<I, BI>;
	/// Start treshold and neutral element
	type GC = TreeGc<I, BI, V>;
	/// TODO this needs some branch index mappings.
	type Migrate = TreeGc<I, BI, V>;

	fn get_db_state(&self, state: &H) -> Option<Self::S> {
		self.mapping.get(state).cloned().map(|i| self.tree.query_plan_at(i))
	}

	fn get_gc(&self) -> Option<Self::GC> {
		if self.touched_gc {
			Some(self.current_gc.clone())
		} else {
			None
		}
	}
}

impl<
	H: Clone + Ord,
	I: Clone + Default + SubAssign<usize> + AddAssign<usize> + Ord,
	BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone + Default,
	V: Clone,
> Management<H> for TreeManagement<H, I, BI, V> {
	// TODO attach gc infos to allow some lazy cleanup (make it optional)
	// on set and on get_mut
	type SE = Latest<(I, BI)>;

	fn get_db_state_mut(&self, state: &H) -> Option<Self::SE> {
		self.mapping.get(state).cloned().and_then(|(i, bi)| {
			// enforce only latest
			if let Some(result) = self.tree.latest_at(i) {
				if result.latest().1 == bi {
					return Some(result)
				}
			}
			None
		})
	}

	fn init() -> (Self, Self::S) {
		let management = Self::default();
		let init_plan = management.tree.query_plan(I::default());
		(management, init_plan)
	}

	fn latest_state(&self) -> Self::SE {
		Latest::unchecked_latest(self.last_in_use_index.clone())
	}

	fn reverse_lookup(&self, state: &Self::S) -> Option<H> {
		// TODO should be the closest valid and return non optional!!!! TODO
		let state = state.history.last()
			.map(|b| (b.branch_index.clone(), b.state.end.clone()))
			.map(|mut b| {
				b.1 -= 1;
				b
			})
			.unwrap_or((Default::default(), Default::default()));
		self.mapping.iter()
			.find(|(_k, v)| v == &&state)
			.map(|(k, _v)| k.clone())
	}

	fn applied_gc(&mut self, gc: Self::GC) {
		self.current_gc.applied(gc);
		self.touched_gc = false;
	}

	fn get_migrate(self) -> Migrate<H, Self> {
		unimplemented!()
	}

	fn applied_migrate(&mut self) {
		unimplemented!()
	}
}

impl<
	H: Clone + Ord,
	I: Clone + Default + SubAssign<usize> + AddAssign<usize> + Ord,
	BI: Ord + Eq + SubAssign<usize> + AddAssign<usize> + Clone + Default,
	V: Clone,
> ForkableManagement<H> for TreeManagement<H, I, BI, V> {
	// note that se must be valid.
	fn append_external_state(&mut self, state: H, at: &Self::SE) -> Option<Self::S> {
		let (branch_index, index) = at.latest();
		let mut index = index.clone();
		index += 1;
		if let Some(branch_index) = self.tree.add_state(branch_index.clone(), index.clone()) {
			let result = self.tree.query_plan(branch_index.clone());
			self.last_in_use_index = (branch_index.clone(), index);
			self.mapping.insert(state, self.last_in_use_index.clone());
			Some(result)
		} else {
			None
		}
	}

	fn try_append_external_state(&mut self, state: H, at: &H) -> Option<Self::S> {
		self.mapping.get(at).and_then(|(branch_index, _index)| {
			self.tree.branch_state(branch_index).map(|branch| {
				let mut index = branch.state.end.clone();
				// TODO factor append_external state at +1 index
				index -= 1;
				Latest::unchecked_latest((branch_index.clone(), index))
			 })
		})
		.and_then(|at| self.append_external_state(state, &at))
	}
}



#[cfg(test)]
pub(crate) mod test {
	use super::*;

	// TODO switch to management function?
	pub(crate) fn test_states() -> Tree<usize, usize> {
		let mut states = Tree::default();
		assert_eq!(states.add_state(0, 1), Some(1));
		// root branching.
		assert_eq!(states.add_state(0, 1), Some(2));
		assert_eq!(Some(true), states.branch_state_mut(&1).map(|ls| ls.add_state()));
		assert_eq!(Some(true), states.branch_state_mut(&1).map(|ls| ls.add_state()));
		assert_eq!(states.add_state(1, 3), Some(3));
		assert_eq!(states.add_state(1, 3), Some(4));
		assert_eq!(states.add_state(1, 2), Some(5));
		assert_eq!(states.add_state(2, 2), Some(2));
		assert_eq!(Some(1), states.drop_state(&1));
		// cannot create when dropped happen on branch
		assert_eq!(Some(false), states.branch_state_mut(&1).map(|ls| ls.add_state()));

		assert!(states.branch_state(&1).unwrap().state.exists(&1));
		assert!(states.branch_state(&1).unwrap().state.exists(&2));
		assert!(!states.branch_state(&1).unwrap().state.exists(&3));
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _ _

		states
	}

	#[test]
	fn test_remove_attached() {
		let mut states = test_states();
		assert_eq!(Some(false), states.branch_state_mut(&1).map(|ls| ls.drop_state()));
		assert!(states.branch_state(&3).unwrap().state.exists(&3));
		assert!(states.branch_state(&4).unwrap().state.exists(&3));
		states.apply_drop_state(&1, &2);
		assert_eq!(states.branch_state(&3), None);
		assert_eq!(states.branch_state(&4), None);
	}

	#[test]
	fn test_query_plans() {
		let states = test_states();
		let ref_3 = vec![
			BranchPlan {
				branch_index: 1,
				state: BranchRange { start: 1, end: 3 },
			},
			BranchPlan {
				branch_index: 3,
				state: BranchRange { start: 3, end: 4 },
			},
		];
		assert_eq!(states.query_plan(3).history, ref_3);

		let mut states = states;

		assert_eq!(states.add_state(1, 2), Some(6));
		let ref_6 = vec![
			BranchPlan {
				branch_index: 1,
				state: BranchRange { start: 1, end: 2 },
			},
			BranchPlan {
				branch_index: 6,
				state: BranchRange { start: 2, end: 3 },
			},
		];
		assert_eq!(states.query_plan(6).history, ref_6);

		states.composite_treshold = 1;
		let mut ref_6 = ref_6;
		ref_6.remove(0);
		assert_eq!(states.query_plan(6).history, ref_6);
	}
}
