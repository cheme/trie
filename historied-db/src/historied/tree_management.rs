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
use crate::rstd::fmt::Debug;
use crate::historied::linear::LinearGC;
use crate::{Management, ManagementRef, Migrate, ForkableManagement, Latest};
use codec::{Codec, Encode, Decode};
use crate::simple_db::{SerializeDB, SerializeMap, SerializeVariable, SerializeInstance, SerializeInstanceVariable};
use derivative::Derivative;

pub trait TreeManagementStorage: Sized {
	type Storage: SerializeDB;
	type Mapping: SerializeInstance;
	type TouchedGC: SerializeInstanceVariable;
	type CurrentGC: SerializeInstanceVariable;
	type LastIndex: SerializeInstanceVariable;
	type NeutralElt: SerializeInstanceVariable;
	type TreeMeta: SerializeInstanceVariable;
	type TreeState: SerializeInstance;

	fn init() -> Self::Storage;
}

impl TreeManagementStorage for () {
	type Storage = ();
	type Mapping = ();
	type TouchedGC = ();
	type CurrentGC = ();
	type LastIndex = ();
	type NeutralElt = ();
	type TreeMeta = ();
	type TreeState = ();

	fn init() -> Self { }
}

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
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct BranchState<I, BI> {
	state: BranchRange<BI>,
	/// does a state get rollback.
	can_append: bool,
	/// is the branch latest.
	is_latest: bool,
	parent_branch_index: I,
}

impl<I, BI: Clone> BranchState<I, BI> {
	pub(crate) fn range(&self) -> (BI, BI) {
		(self.state.start.clone(), self.state.end.clone())
	}
}

/// This is a simple range, end non inclusive.
/// TODO type alias or use ops::Range? see next todo?
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
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
/// It contains all layout information for branches
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
#[derive(Derivative)]
#[derivative(Debug(bound="I: Debug, BI: Debug, S::Storage: Debug"))]
#[derivative(Clone(bound="I: Clone, BI: Clone, S::Storage: Clone"))]
#[cfg_attr(test, derivative(PartialEq(bound="I: PartialEq, BI: PartialEq, S::Storage: PartialEq")))]
pub struct Tree<I: Ord, BI, S: TreeManagementStorage> {
	// TODO this could probably be cleared depending on S::ACTIVE.
	// -> on gc ?
	pub(crate) storage: SerializeMap<I, BranchState<I, BI>, S::Storage, S::TreeState>,
	pub(crate) meta: SerializeVariable<TreeMeta<I, BI>, S::Storage, S::TreeMeta>,
	/// serialize implementation
	pub(crate) serialize: S::Storage,
	// TODO some strategie to close a long branch that gets
	// behind multiple fork? This should only be usefull
	// for high number of modification, small number of
	// fork. The purpose is to avoid history where meaningfull
	// value is always in a low number branch behind a few fork.
	// A longest branch pointer per history is also a viable
	// strategy and avoid fragmenting the history to much.
}

#[derive(Derivative, Encode, Decode)]
#[derivative(Debug(bound="I: Debug, BI: Debug"))]
#[derivative(Clone(bound="I: Clone, BI: Clone"))]
#[cfg_attr(test, derivative(PartialEq(bound="I: PartialEq, BI: PartialEq")))]
pub(crate) struct TreeMeta<I, BI> {
	// TODO pub(crate) storage: SerializeMap<I, BranchState<I, BI>>,
	pub(crate) last_index: I,
	/// treshold for possible node value, correspond
	/// roughly to last cannonical block branch index.
	/// If at default state value, we go through simple storage.
	/// TODO move in tree management??
	pub(crate) composite_treshold: (I, BI),
	/// Is composite latest, so can we write its last state (only
	/// possible on new or after a migration).
	pub(crate) composite_latest: bool,
}

impl<I: Default, BI: Default> Default for TreeMeta<I, BI> {
	fn default() -> Self {
		TreeMeta {
			last_index: I::default(),
			composite_treshold: Default::default(),
			composite_latest: true,
		}
	}
}

impl<I: Ord + Default, BI: Default, S: TreeManagementStorage> Default for Tree<I, BI, S> {
	fn default() -> Self {
		Tree {
			storage: Default::default(),
			meta: Default::default(),
			serialize: S::init(),
		}
	}
}

impl<I: Ord + Default + Codec, BI: Default + Codec, S: TreeManagementStorage> Tree<I, BI, S> {
	pub fn from_ser(mut serialize: S::Storage) -> Self {
		Tree {
			storage: Default::default(),
			meta: SerializeVariable::from_ser(&mut serialize),
			serialize,
		}
	}
}

#[derive(Derivative)]
#[derivative(Debug(bound="V: Debug, I: Debug, BI: Debug, S::Storage: Debug"))]
#[derivative(Clone(bound="V: Clone, I: Clone, BI: Clone, S::Storage: Clone"))]
#[cfg_attr(test, derivative(PartialEq(bound="V: PartialEq, I: PartialEq, BI: PartialEq, S::Storage: PartialEq")))]
pub struct TreeState<I: Ord, BI, V, S: TreeManagementStorage> {
	pub(crate) tree: Tree<I, BI, S>,
	pub(crate) neutral_element: Option<V>,
}

#[derive(Clone, Debug)]
pub struct TreeStateGc<I, BI, V> {
	/// see Tree `storage`
	pub(crate) storage: BTreeMap<I, BranchState<I, BI>>,
	/// see TreeMeta `composite_treshold`
	pub(crate) composite_treshold: (I, BI),
	/// see TreeManagement `neutral_element`
	pub(crate) neutral_element: Option<V>,
}

impl<I: Ord, BI, V, S: TreeManagementStorage> TreeState<I, BI, V, S> {
	pub fn ser(&mut self) -> &mut S::Storage {
		&mut self.tree.serialize
	}
}

#[derive(Derivative)]
#[derivative(Debug(bound="H: Debug, V: Debug, I: Debug, BI: Debug, S::Storage: Debug"))]
#[derivative(Clone(bound="H: Clone, V: Clone, I: Clone, BI: Clone, S::Storage: Clone"))]
#[cfg_attr(test, derivative(PartialEq(bound="H: PartialEq, V: PartialEq, I: PartialEq, BI: PartialEq, S::Storage: PartialEq")))]
pub struct TreeManagement<H: Ord, I: Ord, BI, V, S: TreeManagementStorage> {
	state: TreeState<I, BI, V, S>,
	mapping: SerializeMap<H, (I, BI), S::Storage, S::Mapping>,
	touched_gc: SerializeVariable<bool, S::Storage, S::TouchedGC>,
	current_gc: SerializeVariable<TreeMigrate<I, BI, V>, S::Storage, S::CurrentGC>,
	last_in_use_index: SerializeVariable<(I, BI), S::Storage, S::LastIndex>, // TODO rename to last inserted as we do not rebase on query
	neutral_element: SerializeVariable<Option<V>, S::Storage, S::NeutralElt>,
}

impl<H: Ord, I: Default + Ord, BI: Default, V, S: TreeManagementStorage> Default for TreeManagement<H, I, BI, V, S> {
	fn default() -> Self {
		TreeManagement {
			state: TreeState {
				tree: Tree::default(),
				neutral_element: Default::default(),
			},
			mapping: Default::default(),
			touched_gc: Default::default(),
			current_gc: Default::default(),
			last_in_use_index: Default::default(),
			neutral_element: Default::default(),
		}
	}
}

impl<H: Ord, I: Default + Ord + Codec, BI: Default + Codec, V: Codec + Clone, S: TreeManagementStorage> TreeManagement<H, I, BI, V, S> {
	/// Initialize from a default ser
	pub fn from_ser(mut serialize: S::Storage) -> Self {
		let mut neutral_element_ser = SerializeVariable::<Option<V>, S::Storage, S::NeutralElt>::from_ser(&serialize);
		let neutral_element = neutral_element_ser.handle(&mut serialize).get().clone();
		TreeManagement {
			mapping: Default::default(),
			touched_gc: SerializeVariable::from_ser(&serialize),
			current_gc: SerializeVariable::from_ser(&serialize),
			last_in_use_index: SerializeVariable::from_ser(&serialize),
			neutral_element: neutral_element_ser,
			state: TreeState {
				neutral_element,
				tree: Tree::from_ser(serialize),
			},
		}
	}

	/// Also should guaranty to flush change (but currently implementation
	/// writes synchronously).
	pub fn extract_ser(self) -> S::Storage {
		self.state.tree.serialize
	}

/*	pub fn ser(&mut self) -> &mut S::Storage {
		&mut self.state.tree.serialize
	}*/
}

impl<
	H: Ord,
	I: Ord,
	BI,
	V: Codec,
	S: TreeManagementStorage,
> TreeManagement<H, I, BI, V, S> {
	pub fn define_neutral_element(mut self, n: V) -> Self {
		self.neutral_element.handle(self.state.ser()).set(Some(n));
		self
	}
}

impl<
	H: Clone + Ord + Codec,
	I: Clone + Default + SubAssign<u32> + AddAssign<u32> + Ord + Debug + Codec,
	BI: Ord + Eq + SubAssign<u32> + AddAssign<u32> + Clone + Default + Debug + Codec,
	V,
	S: TreeManagementStorage,
> TreeManagement<H, I, BI, V, S> {
	/// Associate a state for the initial root (default index).
	pub fn map_root_state(&mut self, root: H) {
		self.mapping.handle(self.state.ser()).insert(root, Default::default());
	}

	// TODO consider removing drop_mapping argument (is probably default)
	pub fn apply_drop_state(
		&mut self,
		state: &(I, BI),
		mut drop_mapping: bool,
		mut collect_dropped: Option<&mut Vec<H>>,
	) {
		drop_mapping |= collect_dropped.is_some();
		let mut tree_meta = self.state.tree.meta.handle(&mut self.state.tree.serialize).get().clone();
		// TODO optimized drop from I, BI == 0, 0 and ignore x, 0
		let mapping = &mut self.mapping;
		let collect_dropped = &mut collect_dropped;
		let mut call_back = move |i: &I, bi: &BI, ser: &mut S::Storage| {
			let mut mapping = mapping.handle(ser);
			if drop_mapping {
				let state = (i.clone(), bi.clone());
				// TODO again cost of reverse lookup: consider double mapping
				if let Some(h) = mapping.iter() 
					.find(|(_k, v)| v == &state)
					.map(|(k, _v)| k.clone()) {
					mapping.remove(&h);
					collect_dropped.as_mut().map(|collect| collect.push(h));
				}
			}
		};
		// Less than composite do not contain a 
		if state.1 <= tree_meta.composite_treshold.1 {
			// No branch delete (the implementation guaranty branch 0 is a single element)
			self.state.tree.apply_drop_state_rec_call(&state.0, &state.1, &mut call_back, true);
			let treshold = tree_meta.composite_treshold.clone();
			self.last_in_use_index.handle(self.state.ser()).set(treshold);

			if tree_meta.composite_latest == false {
				tree_meta.composite_latest = true;
				self.state.tree.meta.handle(&mut self.state.tree.serialize).set(tree_meta);
			}
			return;
		}
		let mut previous_index = state.1.clone();
		previous_index -= 1;
		if let Some((parent, branch_end)) = self.state.tree.branch_state(&state.0)
			.map(|s| if s.state.start <= previous_index {
				((state.0.clone(), previous_index), s.state.end)
			} else {
				((s.parent_branch_index.clone(), previous_index), s.state.end)
			}) {
			let mut bi = state.1.clone();
			while bi < branch_end { // TODO should be < branch_end - 1
				call_back(&state.0, &bi, self.state.ser());
				bi += 1;
			}
			call_back(&state.0, &state.1, self.state.ser());
			self.state.tree.apply_drop_state(&state.0, &state.1, &mut call_back);
			self.last_in_use_index.handle(self.state.ser()).set(parent);
		}
	}

	// TODO rename to canonicalize or similar naming
	// TOdO update last_in_use_index
	pub fn apply_drop_from_latest(&mut self, back: usize) -> bool {
		let latest = self.last_in_use_index.handle(self.state.ser()).get().clone();
		let mut switch_index = latest.1.clone();
		switch_index -= back as u32;
		let qp = self.state.tree.query_plan_at(latest);
		let mut branch_index = self.state.tree.meta.handle(&mut self.state.tree.serialize).get().composite_treshold.0.clone();
		for b in qp.iter() {
			if b.0.end <= switch_index {
				branch_index = b.1;
				break;
			}
		}
		// this is the actual operation that should go in a trait TODO EMCH
		self.canonicalize(qp, (branch_index, switch_index))
	}

	// TODO subfunction in tree (more tree related)? This is a migrate (we change
	// composite_treshold).
	pub fn canonicalize(&mut self, branch: ForkPlan<I, BI>, switch_index: (I, BI)) -> bool {

		println!("cano : {:?} {:?}", &branch, &switch_index);
		// TODO makes last index the end of this canonicalize branch

		// TODO move fork plan resolution in?? -> wrong fork plan usage can result in incorrect
		// latest.

		// TODO EMCH keep if branch start index is before switch index, keep
		// only if part of fork plan (putting fork plan in a map as branch index are
		// unrelated to their start).
		// For branch that are in fork plan, if end index is more than the fork plan one (and less than
		// switch index), align.

		// TODO it may be reasonable most of the time to use forkplan index lookup up to some
		// treshold: may need variant depending on number of branch in the forkplan, or
		// have state trait and change name of `filter` to `cache` as it is a particular
		// use case.
		let mut filter: BTreeMap<_, _> = Default::default();
		for h in branch.history.into_iter() {
			if h.state.end > switch_index.1 {
				println!("ins {:?}", h.branch_index);
				filter.insert(h.branch_index, h.state);
			}
		}
		let mut change = false;
		let mut to_change = Vec::new();
		let mut to_remove = Vec::new();
		for (branch_ix, mut branch) in self.state.tree.storage.handle(&mut self.state.tree.serialize).iter() {
				println!("it {:?}", branch_ix);
			if branch.state.start < switch_index.1 {
				if let Some(ref_range) = filter.get(&branch_ix) {
					debug_assert!(ref_range.start == branch.state.start);
					debug_assert!(ref_range.end <= branch.state.end);
					if ref_range.end < branch.state.end {
						branch.state.end = ref_range.end.clone();
						branch.can_append = false;
						to_change.push((branch_ix, branch));
						// TODO EMCH clean mapping for ends shifts
					}
				} else {
				println!("rem {:?}", branch_ix);
					to_remove.push(branch_ix.clone());
				}
			}
		}
		if to_remove.len() > 0 {
			change = true;
			for to_remove in to_remove {
				self.state.tree.storage.handle(&mut self.state.tree.serialize).remove(&to_remove);
				// TODO EMCH clean mapping for range
			}
		}
		if to_change.len() > 0 {
			change = true;
			for (branch_ix, branch) in to_change {
				self.state.tree.storage.handle(&mut self.state.tree.serialize).insert(branch_ix, branch);
				// TODO EMCH clean mapping for range
			}
		}

		let mut handle = self.state.tree.meta.handle(&mut self.state.tree.serialize);
		let tree_meta = handle.get();
		println!("new ct: {:?}", switch_index);
		if switch_index != tree_meta.composite_treshold {
			let mut tree_meta = tree_meta.clone();
			tree_meta.composite_treshold = switch_index;
			handle.set(tree_meta);
			change = true;
		}
		change
	}
}

impl<
	I: Clone + Default + SubAssign<u32> + AddAssign<u32> + Ord + Debug + Codec,
	BI: Ord + Eq + SubAssign<u32> + AddAssign<u32> + Clone + Default + Debug + Codec,
	S: TreeManagementStorage,
> Tree<I, BI, S> {
	/// Return anchor index for this branch history:
	/// - same index as input if the branch was modifiable
	/// - new index in case of branch range creation
	pub fn add_state(
		&mut self,
		branch_index: I,
		number: BI,
	) -> Option<I> {
		let mut meta = self.meta.handle(&mut self.serialize).get().clone();
		if number < meta.composite_treshold.1 {
			return None;
		}
		let mut create_new = false;
		if branch_index <= meta.composite_treshold.0 {
			// only allow terminal append
			let mut next = meta.composite_treshold.1.clone();
			next += 1;
			if number == next {
				if meta.composite_latest {
					meta.composite_latest = false;
				}
				create_new = true;
			} else {
				return None;
			}
		} else {
			let mut handle = self.storage.handle(&mut self.serialize);
			assert!(handle.get(&branch_index).is_some(),
				"Inconsistent state on new block: {:?} {:?}, {:?}",
				branch_index,
				number,
				meta.composite_treshold,
			);
			let branch_state = handle.entry(&branch_index);
	
			let mut can_fork = true;
			branch_state.and_modify(|branch_state| {
				if branch_state.can_append && branch_state.can_add(&number) {
					branch_state.add_state();
				} else {
					if !branch_state.can_fork(&number) {
						can_fork = false;
					} else {
						if branch_state.state.end == number {
							branch_state.is_latest = false;
						}
						create_new = true;
					}
				}
			});
			if !can_fork {
				return None;
			}
		}
		Some(if create_new {
			meta.last_index += 1;
			let state = BranchState::new(number, branch_index);
			self.storage.handle(&mut self.serialize).insert(meta.last_index.clone(), state);
			let result = meta.last_index.clone();

			self.meta.handle(&mut self.serialize).set(meta);
			result
		} else {
			branch_index
		})
	}

	#[cfg(test)]
	pub fn unchecked_latest_at(&mut self, branch_index : I) -> Option<Latest<(I, BI)>> {
		{
			let mut handle = self.meta.handle(&mut self.serialize);
			let meta = handle.get();
			if meta.composite_latest {
				// composite
				if branch_index <= meta.composite_treshold.0 {
					return Some(Latest::unchecked_latest(meta.composite_treshold.clone()));
				} else {
					return None;
				}
			}
		}
		self.storage.handle(&mut self.serialize).get(&branch_index).map(|branch| {
			let mut end = branch.state.end.clone();
			end -= 1;
			Latest::unchecked_latest((branch_index, end))
		})
	}
	
	// TODO this and is_latest is borderline useless, for management implementation only.
	pub fn if_latest_at(&mut self, branch_index: I, seq_index: BI) -> Option<Latest<(I, BI)>> {
		{
			let mut handle = self.meta.handle(&mut self.serialize);
			let meta = handle.get();
			if meta.composite_latest {
				// composite
				if branch_index <= meta.composite_treshold.0 && seq_index == meta.composite_treshold.1 {
					return Some(Latest::unchecked_latest(meta.composite_treshold.clone()));
				} else {
					return None;
				}
			}
		}
		self.storage.handle(&mut self.serialize).get(&branch_index).and_then(|branch| {
			if !branch.is_latest {
				None
			} else {
				let mut end = branch.state.end.clone();
				end -= 1;
				if seq_index == end {
					Some(Latest::unchecked_latest((branch_index, end)))
				} else {
					None
				}
			}
		})
	}
	
	/// TODO doc & switch to &I
	pub fn query_plan_at(&mut self, (branch_index, mut index) : (I, BI)) -> ForkPlan<I, BI> {
		// make index exclusive
		index += 1;
		self.query_plan_inner(branch_index, Some(index))
	}
	/// TODO doc & switch to &I
	pub fn query_plan(&mut self, branch_index: I) -> ForkPlan<I, BI> {
		self.query_plan_inner(branch_index, None)
	}

	fn query_plan_inner(&mut self, mut branch_index: I, mut parent_fork_branch_index: Option<BI>) -> ForkPlan<I, BI> {
		let composite_treshold = self.meta.handle(&mut self.serialize).get().composite_treshold.clone();
		let mut history = Vec::new();
		while branch_index >= composite_treshold.0 {
			if let Some(branch) = self.storage.handle(&mut self.serialize).get(&branch_index) {
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
			history,
			composite_treshold: composite_treshold,
		}
	}

	/// Return anchor index for this branch history:
	/// - same index as input if branch is not empty
	/// - parent index if branch is empty
	/// TODO is it of any use, we probably want to recurse.
	pub fn drop_state(
		&mut self,
		branch_index: &I,
	) -> Option<I> {
		let mut do_remove = None;
		{
			let mut handle = self.storage.handle(&mut self.serialize);
			let mut has_state = false;
			handle.entry(branch_index).and_modify(|branch_state| {
				has_state = true;
				if branch_state.drop_state() {
					do_remove = Some(branch_state.parent_branch_index.clone());
				}
			});
			if !has_state {
				return None;
			}
		}

		Some(if let Some(parent_index) = do_remove {
			self.storage.handle(&mut self.serialize).remove(branch_index);
			parent_index
		} else {
			branch_index.clone()
		})
	}

	pub fn branch_state(&mut self, branch_index: &I) -> Option<BranchState<I, BI>> {
		self.storage.handle(&mut self.serialize).get(branch_index).cloned()
	}

	pub fn branch_state_mut<R, F: FnOnce(&mut BranchState<I, BI>) -> R>(&mut self, branch_index: &I, f: F) -> Option<R> {
		let mut result: Option<R> = None;
		self.storage.handle(&mut self.serialize)
			.entry(branch_index)
			.and_modify(|s: &mut BranchState<I, BI>| {
				result = Some(f(s));
			});
		result
	}

	/// this function can go into deep recursion with full scan, it indicates
	/// that the tree model use here should only be use for small data or
	/// tests. TODO should apply call back here and remove from caller!!
	pub fn apply_drop_state(&mut self,
		branch_index: &I,
		node_index: &BI,
		call_back: &mut impl FnMut(&I, &BI, &mut S::Storage),
	) {
		// Never remove default
		let mut remove = false;
		let mut last = Default::default();
		let mut has_branch = false;
		let mut handle = self.storage.handle(&mut self.serialize);
		let branch_entry = handle.entry(branch_index);
		branch_entry.and_modify(|branch| {
			has_branch = true;
			branch.is_latest = true;
			last = branch.state.end.clone();
			while &branch.state.end > node_index {
				// TODO a function to drop multiple state in linear.
				if branch.drop_state() {
					remove = true;
					break;
				}
			}
		});
		if !has_branch {
			return;
		}
		if remove {
			self.storage.handle(&mut self.serialize).remove(branch_index);
		}
		while &last > node_index {
			last -= 1;
			self.apply_drop_state_rec_call(branch_index, &last, call_back, false);
		}
	}

	pub fn apply_drop_state_rec_call(&mut self,
		branch_index: &I,
		node_index: &BI,
		call_back: &mut impl FnMut(&I, &BI, &mut S::Storage),
		composite: bool,
	) {
		let mut to_delete = Vec::new();
		if composite {
			for (i, s) in self.storage.handle(&mut self.serialize).iter() {
				if &s.state.start >= node_index {
					to_delete.push((i, s));
				}
			}
		} else {
			for (i, s) in self.storage.handle(&mut self.serialize).iter() {
				if &s.parent_branch_index == branch_index && &s.state.start > node_index {
					to_delete.push((i, s));
				}
			}
		}
		for (i, s) in to_delete.into_iter() {
			// TODO these drop is a full branch drop: we could recurse on ourselves
			// into calling function and this function rec on itself and do its own drop
			let mut bi = s.state.start.clone();
			while bi < s.state.end {
				call_back(&i, &bi, &mut self.serialize);
				bi += 1;
			}
			// TODO the store and remove patern is ugly (could use a retain implementation)
			self.storage.handle(&mut self.serialize).remove(&i);
			// composite to false, as no in composite branch are stored.
			self.apply_drop_state_rec_call(&i, &s.state.start, call_back, false);
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
	pub composite_treshold: (I, BI),
}

impl<I: Clone, BI: Clone + SubAssign<u32>> ForkPlan<I, BI> {
	fn latest(&self) -> (I, BI) {
		if let Some(branch_plan) = self.history.last() {
			let mut index = branch_plan.state.end.clone();
			index -= 1;
			(branch_plan.branch_index.clone(), index)
		} else {
			self.composite_treshold.clone()
		}
	}
}

impl<I: Default, BI: Default> Default for ForkPlan<I, BI> {
	fn default() -> Self {
		ForkPlan {
			history: Vec::new(),
			composite_treshold: Default::default(),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Query plan element for a single branch.
pub struct BranchPlan<I, BI> {
	// TODO rename to index
	pub branch_index: I,
	pub state: BranchRange<BI>,
}

impl<'a, I: Default + Eq + Ord + Clone, BI: SubAssign<u32> + Ord + Clone> BranchesContainer<I, BI> for &'a ForkPlan<I, BI> {
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

impl<I: Ord + SubAssign<u32> + Clone> BranchContainer<I> for BranchRange<I> {

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
impl<I: Default, BI: Default + AddAssign<u32>> Default for BranchState<I, BI> {

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
			is_latest: true,
			parent_branch_index: Default::default(),
		}
	}
}

impl<I, BI: Ord + Eq + SubAssign<u32> + AddAssign<u32> + Clone> BranchState<I, BI> {

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
			is_latest: true,
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

#[derive(Debug, Clone, Encode, Decode)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct BranchGC<I, BI, V> {
	pub branch_index: I,
	/// A new start - end limit for the branch or a removed
	/// branch.
	pub new_range: Option<LinearGC<BI, V>>,
}

#[derive(Debug, Clone, Encode, Decode)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct TreeMigrate<I, BI, V> {
	/// Every modified branch.
	/// Ordered by branch index.
	pub changes: Vec<BranchGC<I, BI, V>>,
	// TODO is also in every lineargc of branchgc.
	pub neutral_element: Option<V>,
	// TODO add the key elements (as option to trigger registration or not).
}

impl<I, BI, V> Default for TreeMigrate<I, BI, V> {
	fn default() -> Self {
		TreeMigrate {
			changes: Vec::new(),
			neutral_element: None,
		}
	}
}

impl<I, BI, V> TreeMigrate<I, BI, V> {
	fn applied(&mut self, gc_applied: TreeMigrate<I, BI, V>) {
		unimplemented!("TODO run a delta to keep possible updates in between");
	}
}

impl<
	H: Ord + Clone + Codec,
	I: Clone + Default + SubAssign<u32> + AddAssign<u32> + Ord + Debug + Codec,
	BI: Ord + Eq + SubAssign<u32> + AddAssign<u32> + Clone + Default + Debug + Codec,
	V: Clone,
	S: TreeManagementStorage,
> ManagementRef<H> for TreeManagement<H, I, BI, V, S> {
	type S = ForkPlan<I, BI>;
	/// Start treshold and neutral element
	type GC = TreeState<I, BI, V, S>;
	/// TODO this needs some branch index mappings.
	type Migrate = TreeMigrate<I, BI, V>;

	fn get_db_state(&mut self, state: &H) -> Option<Self::S> {
		self.mapping.handle(self.state.ser()).get(state).cloned().map(|i| self.state.tree.query_plan_at(i))
	}

	fn get_gc(&self) -> Option<crate::Ref<Self::GC>> {
		Some(crate::Ref::Borrowed(&self.state))
	}
}

impl<
	H: Clone + Ord + Codec,
	I: Clone + Default + SubAssign<u32> + AddAssign<u32> + Ord + Debug + Codec,
	BI: Ord + Eq + SubAssign<u32> + AddAssign<u32> + Clone + Default + Debug + Codec,
	V: Clone,
	S: TreeManagementStorage,
> Management<H> for TreeManagement<H, I, BI, V, S> {
	// TODO attach gc infos to allow some lazy cleanup (make it optional)
	// on set and on get_mut
	type SE = Latest<(I, BI)>;

	fn get_db_state_mut(&mut self, state: &H) -> Option<Self::SE> {
		self.mapping.handle(self.state.ser()).get(state).cloned().and_then(|(i, bi)| {
			// enforce only latest
			self.state.tree.if_latest_at(i, bi)
		})
	}

	fn init() -> (Self, Self::S) {
		let mut management = Self::default();
		let init_plan = management.state.tree.query_plan(I::default());
		(management, init_plan)
	}

	fn latest_state(&mut self) -> Self::SE {
		let latest = self.last_in_use_index.handle(self.state.ser()).get().clone();
		Latest::unchecked_latest(latest)
	}

	// TODO the state parameter may not be the correct one.
	fn reverse_lookup(&mut self, state: &Self::S) -> Option<H> {
		// TODO should be the closest valid and return non optional!!!! TODO
		let state = state.history.last()
			.map(|b| (b.branch_index.clone(), b.state.end.clone()))
			.map(|mut b| {
				b.1 -= 1;
				b
			})
			.unwrap_or((Default::default(), Default::default()));
		self.mapping.handle(self.state.ser()).iter()
			.find(|(_k, v)| v == &state)
			.map(|(k, _v)| k.clone())
	}

	fn get_migrate(self) -> Migrate<H, Self> {
		unimplemented!()
	}

	fn applied_migrate(&mut self) {
		
	//	self.current_gc.applied(gc); TODO pass back this reference: put it in buf more likely
	//	(remove the associated type)
		self.touched_gc.handle(self.state.ser()).set(false);
	}
}

impl<
	H: Clone + Ord + Codec,
	I: Clone + Default + SubAssign<u32> + AddAssign<u32> + Ord + Debug + Codec,
	BI: Ord + Eq + SubAssign<u32> + AddAssign<u32> + Clone + Default + Debug + Codec,
	V: Clone,
	S: TreeManagementStorage,
> ForkableManagement<H> for TreeManagement<H, I, BI, V, S> {

	type SF = (I, BI);

	fn inner_fork_state(&self, s: Self::SE) -> Self::SF {
		s.0
	}

	fn ref_state_fork(&self, s: &Self::S) -> Self::SF {
		s.latest()
	}

	fn get_db_state_for_fork(&mut self, state: &H) -> Option<Self::SF> {
		self.mapping.handle(self.state.ser()).get(state).cloned()
	}

	// note that se must be valid.
	fn append_external_state(&mut self, state: H, at: &Self::SF) -> Option<Self::S> {
		let (branch_index, index) = at;
		let mut index = index.clone();
		index += 1;
		if let Some(branch_index) = self.state.tree.add_state(branch_index.clone(), index.clone()) {
			let result = self.state.tree.query_plan(branch_index.clone());
			let last_in_use_index = (branch_index.clone(), index);
			self.last_in_use_index.handle(self.state.ser()).set(last_in_use_index.clone());
			self.mapping.handle(self.state.ser()).insert(state, last_in_use_index);
			Some(result)
		} else {
			None
		}
	}

	fn drop_state(&mut self, state: &Self::SF, return_dropped: bool) -> Option<Vec<H>> {
		let mut result = if return_dropped {
			Some(Vec::new())
		} else {
			None
		};
		self.apply_drop_state(state, true, result.as_mut());
		result
	}
}



#[cfg(test)]
pub(crate) mod test {
	use super::*;

	pub(crate) fn test_states() -> Tree<u32, u32, ()> {
		test_states_inner()
	}
	// TODO switch to management function?
	pub(crate) fn test_states_inner<T: TreeManagementStorage>() -> Tree<u32, u32, T> {
		let mut states = Tree::default();
		assert_eq!(states.add_state(0, 1), Some(1));
		// root branching.
		assert_eq!(states.add_state(0, 1), Some(2));
		assert_eq!(Some(true), states.branch_state_mut(&1, |ls| ls.add_state()));
		assert_eq!(Some(true), states.branch_state_mut(&1, |ls| ls.add_state()));
		assert_eq!(states.add_state(1, 3), Some(3));
		assert_eq!(states.add_state(1, 3), Some(4));
		assert_eq!(states.add_state(1, 2), Some(5));
		assert_eq!(states.add_state(2, 2), Some(2));
		assert_eq!(Some(1), states.drop_state(&1));
		// cannot create when dropped happen on branch
		assert_eq!(Some(false), states.branch_state_mut(&1, |ls| ls.add_state()));

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
		assert_eq!(Some(false), states.branch_state_mut(&1, |ls| ls.drop_state()));
		// does not recurse
		assert!(states.branch_state(&3).unwrap().state.exists(&3));
		assert!(states.branch_state(&4).unwrap().state.exists(&3));
		assert!(states.branch_state(&5).unwrap().state.exists(&2));
		let mut states = test_states();
		states.apply_drop_state(&1, &2, &mut |_i, _bi, _ser| {});
		// does recurse
		assert_eq!(states.branch_state(&3), None);
		assert_eq!(states.branch_state(&4), None);
		assert!(states.branch_state(&5).unwrap().state.exists(&2));
	}

	#[test]
	fn test_query_plans() {
		let mut states = test_states();
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

		let mut meta = states.meta.handle(&mut states.serialize).get().clone();
		meta.composite_treshold = (2, 1);
		states.meta.handle(&mut states.serialize).set(meta);

		let mut ref_6 = ref_6;
		ref_6.remove(0);
		assert_eq!(states.query_plan(6).history, ref_6);
	}
}
