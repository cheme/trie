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

// TODO remove "previous code" expect.

use super::{HistoriedValue, ValueRef, Value, InMemoryValueRef, InMemoryValue, InMemoryValueSlice, InMemoryValueRange, UpdateResult};
use crate::backend::{LinearStorage, LinearStorageRange, LinearStorageSlice, LinearStorageMem};
use crate::historied::linear::{Linear, LinearState, LinearGC};
use crate::management::tree::{ForkPlan, BranchesContainer, TreeStateGc, DeltaTreeStateGc, MultipleGc, MultipleMigrate};
use crate::rstd::ops::SubAssign;
use crate::rstd::marker::PhantomData;
use crate::Latest;
use codec::{Encode, Decode};
use derivative::Derivative;

// TODO for not in memory we need some direct or indexed api, returning value
// and the info if there can be lower value index (not just a direct index).
// -> then similar to those reverse iteration with possible early exit.
// -> Also need to attach some location index (see enumerate use here)

// strategy such as in linear are getting too complex for tree, just using
// macros to remove duplicated code.

// get from tree
macro_rules! tree_get {
	($fn_name: ident, $return_type: ty, $branch_query: ident, $value_query: expr, $post_process: expr) => {
	fn $fn_name<'a>(&'a self, at: &<Self as ValueRef<V>>::S) -> Option<$return_type> {
		let mut index = self.branches.len();
		// note that we expect branch index to be linearily set
		// along a branch (no state containing unordered branch_index
		// and no history containing unorderd branch_index).
		if index == 0 {
			return None;
		}

		for (state_branch_range, state_branch_index) in at.iter() {
			while index > 0 {
				let branch_index = &self.branches.get_state(index - 1).expect("previous code");
				if branch_index < &state_branch_index {
					break;
				} else if branch_index == &state_branch_index {
					// TODO add a lower bound check (maybe debug_assert it only).
					let mut upper_bound = state_branch_range.end.clone();
					upper_bound -= 1;
					let branch = self.branches.$branch_query(index - 1).expect("previous code").value;
					if let Some(result) = $value_query(&branch, &upper_bound) {
						return Some($post_process(result, branch))
					}
				}
				index -= 1;
			}
		}

		// composite part.
		while index > 0 {
			let branch_index = &self.branches.get_state(index - 1).expect("previous code");
			if branch_index <= &at.composite_treshold.0 {
				let branch = self.branches.$branch_query(index - 1).expect("previous code").value;
				if let Some(result) = $value_query(&branch, &at.composite_treshold.1) {
					return Some($post_process(result, branch))
				}
			}
			index -= 1;
		}
	
		None
	}
	}
}

#[derive(Derivative, Debug, Clone, Encode, Decode)]
#[derivative(PartialEq(bound="D: PartialEq"))]
pub struct Tree<I, BI, V, D, BD> {
	branches: D,
	_ph: PhantomData<(I, BI, V, BD)>,
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

impl<I, BI, V, D: Default, BD> Default for Tree<I, BI, V, D, BD> {
	fn default() -> Self {
		Tree {
			branches: D::default(),
			_ph: PhantomData,
		}
	}
}

/*#[derive(Debug, Clone, Encode, Decode)]
#[cfg_attr(any(test, feature = "test"), derive(PartialEq))]
pub struct Branch<I, BI, V, BD> {
	branch_index: I,
	history: Linear<V, BI, BD>,
}*/
type Branch<I, BI, V, BD> = HistoriedValue<Linear<V, BI, BD>, I>;

impl<
	I: Clone,
	BI: LinearState + SubAssign<BI>,
	V: Clone + Eq,
	BD: LinearStorage<V, BI>,
> Branch<I, BI, V, BD>
{
	pub fn new(value: V, state: &Latest<(I, BI)>) -> Self {
		let (branch_index, index) = state.latest().clone();
		let index = Latest::unchecked_latest(index); // TODO cast ptr?
		let history = Linear::new(value, &index);
		Branch {
			state: branch_index,
			value: history,
		}
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32>, // TODO consider subassing usize or minus one trait...
	V: Clone,
	D: LinearStorage<Linear<V, BI, BD>, I>, // TODO rewrite to be linear storage of BD only.
	BD: LinearStorage<V, BI>,
> ValueRef<V> for Tree<I, BI, V, D, BD> {
	type S = ForkPlan<I, BI>;

	tree_get!(get, V, st_get, |b: &Linear<V, BI, BD>, ix| b.get(ix), |r, _| r);

	fn contains(&self, at: &Self::S) -> bool {
		self.get(at).is_some() // TODO avoid clone??
	}

	fn is_empty(&self) -> bool {
		// This implies remove from linear clean directly the parent vec.
		self.branches.len() == 0
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32>,
	V: Clone,
	D: LinearStorageMem<Linear<V, BI, BD>, I>,
	BD: LinearStorageMem<V, BI>,
> InMemoryValueRef<V> for Tree<I, BI, V, D, BD> {
	tree_get!(get_ref, &V, get_ref, |b: &'a Linear<V, BI, BD>, ix| b.get_ref(ix), |r, _| r );
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32> + SubAssign<BI>,
	V: Clone + Eq,
	D: LinearStorage<Linear<V, BI, BD>, I>,
	BD: LinearStorage<V, BI>,
> Value<V> for Tree<I, BI, V, D, BD> {
	type SE = Latest<(I, BI)>;
	type Index = (I, BI);
	type GC = MultipleGc<I, BI, V>;
	//type Migrate = (BI, TreeMigrate<I, BI, V>);
	type Migrate = MultipleMigrate<I, BI, V>;

	fn new(value: V, at: &Self::SE) -> Self {
		let mut v = D::default();
		v.push(Branch::new(value, at));
		Tree {
			branches: v,
			_ph: PhantomData,
		}
	}

	fn set(&mut self, value: V, at: &Self::SE) -> UpdateResult<()> {
		// Warn dup code, can be merge if change set to return previ value: with
		// ref refact will be costless
		let (branch_index, index) = at.latest();
		let mut insert_at = self.branches.len();
		/* TODO write iter_mut that iterate on a HandleMut as in simple_db */
/*		for (iter_index, branch) in self.branches.iter_mut().enumerate().rev() {
			if &branch.branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.history.set(value, &index);
			}
			if &branch.branch_index < branch_index {
				insert_at = iter_index + 1;
				break;
			} else {
				insert_at = iter_index;
			}
		}*/
		let len = insert_at;
		for ix in 0..len {
			let iter_index = len - 1 - ix;
			let iter_branch_index = self.branches.get_state(iter_index).expect("previous code");
			if &iter_branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());
				let mut branch = self.branches.st_get(iter_index).expect("previous code");
				return match branch.value.set(value, &index) {
					UpdateResult::Changed(_) => {
						self.branches.emplace(iter_index, branch);
						UpdateResult::Changed(())
					},
					UpdateResult::Cleared(_) => {
						self.branches.remove(iter_index);
						if self.branches.len() == 0 {
							UpdateResult::Cleared(())
						} else {
							UpdateResult::Changed(())
						}
					},
					UpdateResult::Unchanged => UpdateResult::Unchanged,
				};
			}
			if &iter_branch_index < branch_index {
				insert_at = iter_index + 1;
				break;
			} else {
				insert_at = iter_index;
			}
		}

		let branch = Branch::new(value, at);
		if insert_at == self.branches.len() {
			self.branches.push(branch);
		} else {
			self.branches.insert(insert_at, branch);
		}
		UpdateResult::Changed(())
	}

	fn discard(&mut self, at: &Self::SE) -> UpdateResult<Option<V>> {
		let (branch_index, index) = at.latest();
		/* TODO write iter_mut_rev that iterate on a HandleMut as in simple_db */
/*		for branch in self.branches.iter_mut().rev() {
			if &branch.branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.history.discard(&index);
			}
			if &branch.branch_index < branch_index {
				break;
			}
		}*/
		let len = self.branches.len();
		for ix in 0..len {
			let iter_index = len - 1 - ix;
			let iter_branch_index = self.branches.get_state(iter_index).expect("previous code");
			if &iter_branch_index == branch_index {
				let index = Latest::unchecked_latest(index.clone());
				let mut branch = self.branches.st_get(iter_index).expect("previous code");
				return match branch.value.discard(&index) {
					UpdateResult::Changed(v) => {
						self.branches.emplace(iter_index, branch);
						UpdateResult::Changed(v)
					},
					UpdateResult::Cleared(v) => {
						self.branches.remove(iter_index);
						if self.branches.len() == 0 {
							UpdateResult::Cleared(v)
						} else {
							UpdateResult::Changed(v)
						}
					},
					UpdateResult::Unchanged => UpdateResult::Unchanged,
				};
			}
			if &iter_branch_index < branch_index {
				break;
			}
		}
	
		UpdateResult::Unchanged
	}

	fn gc(&mut self, gc: &Self::GC) -> UpdateResult<()> {
		match gc {
			MultipleGc::Journaled(gc) => self.journaled_gc(gc),
			MultipleGc::State(gc) => self.state_gc(gc),
		}
	}

	// TODO is it really usefull
	fn is_in_migrate((index, linear_index) : &Self::Index, gc: &Self::Migrate) -> bool {
		match gc {
			MultipleMigrate::Noops => (),
			MultipleMigrate::JournalGc(gc) => {
				if let Some(new_start) = gc.composite_treshold_new_start.as_ref() {
					if linear_index <= &new_start {
						return true;
					}
				}
				if let Some(br) = gc.storage.get(&index) {
					return if let Some(bi) = br.as_ref() {
						bi <= linear_index
					} else {
						true
					};
				}
			},
			MultipleMigrate::Rewrite(_gc) => {
				unimplemented!()
			},
		}
/*		for branch in gc.1.changes.iter().rev() {
			let bi = &gc.0;
			if &branch.branch_index == index {
				return branch.new_range.as_ref()
					.map(|gc| {
						let linear_migrate = (bi.clone(), gc.clone());
						Linear::<_, _, crate::backend::in_memory::MemoryOnly<V, BI>>::is_in_migrate(linear_index, &linear_migrate)
					}).unwrap_or(true);
			}
			if &branch.branch_index < &index {
				break;
			}
		}*/
		false
	}

	fn migrate(&mut self, mig: &mut Self::Migrate) -> UpdateResult<()> {
		let mut result = UpdateResult::Unchanged;

		match mig {
			MultipleMigrate::JournalGc(gc) => {
				result = self.journaled_gc(gc);
				if let UpdateResult::Cleared(()) = result {
					return UpdateResult::Cleared(());
				}
				let mut new_branch: Option<Branch<I, BI, V, BD>> = None;
				let mut i = 0;
				// TODO merge all less than composite treshold in composite treshold index branch.
				loop {
					if let Some(mut branch) = self.branches.st_get(i) {
						if branch.state <= gc.composite_treshold.0 {
								if let Some(new_branch) = new_branch.as_mut() {
								for i in 0.. {
									if let Some(h) = branch.value.st_get(i) {
										new_branch.value.push(h);
									} else {
										break;
									}
								}
							} else {
								branch.state = gc.composite_treshold.0.clone();
								new_branch = Some(branch);
							}
						} else {
							break;
						}
					} else {
						break;
					}
					i += 1;
				}
				if let Some(new_branch) = new_branch {
					if i == self.branches.len() {
						self.branches.clear();
						self.branches.push(new_branch);
					} else {
						self.branches.truncate_until(i - 1);
						self.branches.emplace(0, new_branch);
					}
				}
			},
			MultipleMigrate::Rewrite(_gc) => unimplemented!(),
			MultipleMigrate::Noops => (),
		}

		/*// This is basis from old gc, it does not do indexing change as it could
		// be possible. TODO start migrate too (mig.0)
		let start_len = self.branches.len();
		let mut gc_iter = mig.1.changes.iter_mut().rev();
		let mut branch_iter = Some(start_len - 1);
		//let mut branch_iter = self.branches.iter_mut().enumerate().rev();
		let mut o_gc = gc_iter.next();
		let mut o_branch = branch_iter.and_then(|i| self.branches.get_state(i).map(|s| (i, s)));
		//let mut o_branch = branch_iter.next();
		// TODO ref mut remove (not previously works fine with as ref so refact serialize to inner mut.
		while let (Some(gc), Some((index, branch_index))) = (o_gc.as_mut(), o_branch.clone()) {
			branch_iter = branch_iter.and_then(|v| if v > 0 {
				Some(v - 1)
			} else {
				None
			});

			if gc.branch_index == branch_index {
				if let Some(gc) = gc.new_range.as_mut() {
					let mut branch = self.branches.st_get(index).expect("previous code");
					match branch.value.gc(gc) {
						UpdateResult::Unchanged => (),
						UpdateResult::Changed(_) => { 
							self.branches.emplace(index, branch);
							result = UpdateResult::Changed(());
						},
						UpdateResult::Cleared(_) => {
							self.branches.remove(index);
							result = UpdateResult::Changed(());
						},
					}
				} else {
					self.branches.remove(index);
					result = UpdateResult::Changed(());
				}
				o_gc = gc_iter.next();
				o_branch = branch_iter.and_then(|i| self.branches.get_state(i).map(|s| (i, s)));
			} else if gc.branch_index < branch_index {
				o_branch = branch_iter.and_then(|i| self.branches.get_state(i).map(|s| (i, s)));
			} else {
				o_gc = gc_iter.next();
			}
		}
		*/

		if let UpdateResult::Changed(()) = result {
			if self.branches.len() == 0 {
				result = UpdateResult::Cleared(());
			}
		}
		result
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32> + SubAssign<BI> + Clone,
	V: Clone + Eq,
	D: LinearStorage<Linear<V, BI, BD>, I>,
	BD: LinearStorage<V, BI>,
> Tree<I, BI, V, D, BD> {
	fn state_gc(&mut self, gc: &TreeStateGc<I, BI, V>) -> UpdateResult<()> {
		let neutral = &gc.neutral_element;
		let mut result = UpdateResult::Unchanged;
		let start_composite = &gc.composite_treshold_new_start;
		let mut gc_iter = gc.storage.iter().rev();
		let len = self.branches.len();
		// TODO use rev iter mut implementation
		let mut branch_iter = Some(len - 1);
//		let mut branch_iter = self.branches.iter_mut().enumerate().rev();
//			let iter_branch_index = self.branches.get_state(iter_index).expect("previous code");
	
		let mut o_gc = gc_iter.next();
		let mut o_branch = branch_iter.and_then(|i| self.branches.get_state(i).map(|s| (i, s)));
		while let (Some(gc), Some((index, branch_index))) = (o_gc.as_ref(), o_branch.as_ref()) {
			branch_iter = branch_iter.and_then(|v| if v > 0 {
				Some(v - 1)
			} else {
				None
			});
			if gc.0 == branch_index {
				// TODO using linear gc does not make sense here (no sense of delta: TODO change
				// linear to use a simple range with neutral).
				let (start, end) = gc.1.range();
				let start = start_composite.as_ref().and_then(|start_composite| if &start < start_composite {
					Some(start_composite.clone())
				} else {
					None
				}).unwrap_or(start);
				let mut gc = LinearGC {
					new_start: Some(start),
					new_end:  Some(end),
					neutral_element: neutral.clone(),
				};

				let mut branch = self.branches.st_get(*index).expect("previous code");
				match branch.value.gc(&mut gc) {
					UpdateResult::Unchanged => (),
					UpdateResult::Changed(_) => { 
						self.branches.emplace(*index, branch);
						result = UpdateResult::Changed(());
					},
					UpdateResult::Cleared(_) => {
						self.branches.remove(*index);
						result = UpdateResult::Changed(());
					}
				}

				o_gc = gc_iter.next();

				o_branch = branch_iter.and_then(|i| self.branches.get_state(i).map(|s| (i, s)));
			} else if gc.0 < &branch_index {
				self.branches.remove(*index);
				result = UpdateResult::Changed(());
				o_branch = branch_iter.and_then(|i| self.branches.get_state(i).map(|s| (i, s)));
			} else {
				o_gc = gc_iter.next();
			}
		}

		if let UpdateResult::Changed(()) = result {
			if self.branches.len() == 0 {
				result = UpdateResult::Cleared(());
			}
		}

		result
	}

	fn journaled_gc(&mut self, gc: &DeltaTreeStateGc<I, BI, V>) -> UpdateResult<()> {
		// for all branch check if in deleted.
		// Also apply new start on all.
		let neutral = &gc.neutral_element;
		let mut result = UpdateResult::Unchanged;
		let start_composite = gc.composite_treshold_new_start.as_ref();
		let len = self.branches.len();
		let mut first_new_start = false;
		for index in (0..len).rev() {

			let mut branch = self.branches.st_get(index).expect("We already fetch its state.");
			let new_start = if branch.state <= gc.composite_treshold.0 {
				match start_composite.as_ref() {
					None => None,
					Some(n_start) => {
						if first_new_start {
							self.branches.remove(index);
							result = UpdateResult::Changed(());
							continue;
						} else {
							if let Some(b) = branch.value.st_get(0) {
								if &b.state < n_start {
									first_new_start = true;
								}
							}
							start_composite.cloned()
						}
					},
				}
			} else {
				None
			};
			if let Some(mut gc) = if let Some(change) = gc.storage.get(&branch.state) {
				if change.is_none() {
					self.branches.remove(index);
					result = UpdateResult::Changed(());
					None
				} else {
					Some(LinearGC {
						new_start,
						new_end: change.clone(),
						neutral_element: neutral.clone(),
					})
				}
			} else {
				if first_new_start {
					Some(LinearGC {
						new_start,
						new_end: None,
						neutral_element: neutral.clone(),
					})
				} else {
					None
				}
			} {
				match branch.value.gc(&mut gc) {
					UpdateResult::Unchanged => (),
						UpdateResult::Changed(_) => { 
						self.branches.emplace(index, branch);
						result = UpdateResult::Changed(());
					},
					UpdateResult::Cleared(_) => {
						self.branches.remove(index);
						result = UpdateResult::Changed(());
					}
				}
			}
		}

		if let UpdateResult::Changed(()) = result {
			if self.branches.len() == 0 {
				result = UpdateResult::Cleared(());
			}
		}

		result
	}


	#[cfg(test)]
	fn nb_internal_history(&self) -> usize {
		let mut nb = 0;
		for i in 0..self.branches.len() {
			let branch = self.branches.st_get(i).unwrap();
			nb += branch.value.len();
		}
		nb
	}
	#[cfg(test)]
	fn nb_internal_branch(&self) -> usize {
		self.branches.len()
	}
}


impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32> + SubAssign<BI>,
	V: Clone + Eq,
	D: LinearStorageMem<Linear<V, BI, BD>, I>,
	BD: LinearStorageMem<V, BI>,
> InMemoryValue<V> for Tree<I, BI, V, D, BD> {
	fn get_mut(&mut self, at: &Self::SE) -> Option<&mut V> {
		let (branch_index, index) = at.latest();
		let len = self.branches.len();
		for ix in 0..len {
			let iter_index = len - 1 - ix;
			let branch_state = self.branches.get_state(iter_index).expect("previous code");
//		for branch in self.branches.iter_mut().rev() {
			if &branch_state == branch_index {
				let branch = self.branches.get_ref_mut(iter_index).expect("previous code");
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.value.get_mut(&index);
			}
			if &branch_state < branch_index {
				break;
			}
		}
		None
	}

	fn set_mut(&mut self, value: V, at: &Self::SE) -> UpdateResult<Option<V>> {
		// Warn dup code, can be merge if change set to return previ value: with
		// ref refact will be costless
		let (branch_index, index) = at.latest();
		let mut insert_at = self.branches.len();
		let len = insert_at;
		for ix in 0..len {
			let iter_index = len - 1 - ix;
			let branch = self.branches.get_ref_mut(iter_index).expect("previous code");
	
//		for (iter_index, branch) in self.branches.iter_mut().enumerate().rev() {
			if &branch.state == branch_index {
				let index = Latest::unchecked_latest(index.clone());// TODO reftransparent &
				return branch.value.set_mut(value, &index);
			}
			if &branch.state < branch_index {
				insert_at = iter_index + 1;
				break;
			} else {
				insert_at = iter_index;
			}
		}
		let branch = Branch::new(value, at);
		if insert_at == self.branches.len() {
			self.branches.push(branch);
		} else {
			self.branches.insert(insert_at, branch);
		}
		UpdateResult::Changed(None)
	}
}

type LinearBackendTempSize = crate::backend::in_memory::MemoryOnly<Option<Vec<u8>>, u32>;
type TreeBackendTempSize = crate::backend::in_memory::MemoryOnly<Linear<Option<Vec<u8>>, u32, LinearBackendTempSize>, u32>;

impl Tree<u32, u32, Option<Vec<u8>>, TreeBackendTempSize, LinearBackendTempSize> {
	/// Temporary function to get occupied stage.
	/// TODO replace by heapsizeof
	pub fn temp_size(&self) -> usize {
		let mut size = 0;
		for i in 0 .. self.branches.len() {
			if let Some(b) = self.branches.get_ref(i) {
				size += 4; // branch index (using u32 as usize)
				size += b.value.temp_size();
			}
		}
		size
	}
}

impl<
	I: Default + Eq + Ord + Clone,
	BI: LinearState + SubAssign<u32>,
	V: Clone + AsRef<[u8]> + AsMut<[u8]>,
	D: LinearStorageSlice<Linear<V, BI, BD>, I>,
	BD: AsRef<[u8]> + AsMut<[u8]> + LinearStorageRange<V, BI>,
> InMemoryValueSlice<V> for Tree<I, BI, V, D, BD> {
	tree_get!(
		get_slice,
		&[u8],
		get_slice,
		|b: &'a [u8], ix| <Linear<V, BI, BD>>::get_range(b, ix),
		|result, b: &'a [u8]| &b[result]
	);
}


#[cfg(test)]
mod test {
	use super::*;
	use crate::management::tree::test::{test_states, test_states_st};

	#[test]
	fn compile_double_encoded() {
		use crate::backend::encoded_array::{EncodedArray, NoVersion};
		use crate::historied::ValueRef;

		type BD<'a> = EncodedArray<'a, Vec<u8>, NoVersion>;
//		type D<'a> = crate::historied::linear::MemoryOnly<
		type D<'a> = EncodedArray<'a,
			crate::historied::linear::Linear<Vec<u8>, u32, BD<'a>>,
			NoVersion,
//			u32
		>;
		let item: Tree<u32, u32, Vec<u8>, D, BD> = Default::default();
		let at: ForkPlan<u32, u32> = Default::default();
		item.get(&at);
		item.get_slice(&at);
		let latest = Latest::unchecked_latest((0, 0));
		let _item: Tree<u32, u32, Vec<u8>, D, BD> = Tree::new(b"dtd".to_vec(), &latest);
//		let slice = &b"dtdt"[..];
//		use crate::backend::encoded_array::{EncodedArrayValue};
//		let bd = crate::historied::linear::Linear::<Vec<u8>, u32, BD>::from_slice(slice);
//		let bd = BD::from_slice(slice);
		let bd = D::default();
		use crate::backend::LinearStorage;
		bd.st_get(1usize);
	}

	#[test]
	fn test_set_get() {
		// TODO EMCH parameterize test
		type BD = crate::backend::in_memory::MemoryOnly<u32, u32>;
		type D = crate::backend::in_memory::MemoryOnly<
			crate::historied::linear::Linear<u32, u32, BD>,
			u32,
		>;
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _
		let mut states = test_states();
		let mut item: Tree<u32, u32, u32, D, BD> = Default::default();

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
		assert_eq!(Some(false), states.branch_state_mut(&1, |ls| ls.drop_state()));

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

	#[test]
	fn test_migrate() {
		use crate::{Management, ManagementRef, ForkableManagement};
		use crate::test::simple_impl::StateInput;
		type BD = crate::backend::in_memory::MemoryOnly<u16, u32>;
		type D = crate::backend::in_memory::MemoryOnly<
			crate::historied::linear::Linear<u16, u32, BD>,
			u32,
		>;
		let mut states = crate::test::fuzz::InMemoryMgmtSer::default()
			.define_neutral_element(0);
		let s0 = states.latest_state_fork();

		let mut item1: Tree<u32, u32, u16, D, BD> = Default::default();
		let mut item2: Tree<u32, u32, u16, D, BD> = Default::default();
		let s1 = states.append_external_state(StateInput(1), &s0).unwrap();
		item1.set(1, &states.get_db_state_mut(&StateInput(1)).unwrap());
		item2.set(1, &states.get_db_state_mut(&StateInput(1)).unwrap());
		// fusing cano
		let _ = states.append_external_state(StateInput(101), &s1).unwrap();
		item1.set(2, &states.get_db_state_mut(&StateInput(101)).unwrap());
		item2.set(2, &states.get_db_state_mut(&StateInput(101)).unwrap());
		let s1 = states.append_external_state(StateInput(102), &s1).unwrap();
		item1.set(3, &states.get_db_state_mut(&StateInput(102)).unwrap());
		let s1 = states.append_external_state(StateInput(103), &s1).unwrap();
		item1.set(4, &states.get_db_state_mut(&StateInput(103)).unwrap());
		let _ = states.append_external_state(StateInput(104), &s1).unwrap();
		item1.set(5, &states.get_db_state_mut(&StateInput(104)).unwrap());
		let s1 = states.append_external_state(StateInput(105), &s1).unwrap();
		item1.set(6, &states.get_db_state_mut(&StateInput(105)).unwrap());
		item2.set(6, &states.get_db_state_mut(&StateInput(105)).unwrap());
		// end fusing (shift following branch index by 2)
		let s2 = states.append_external_state(StateInput(2), &s0).unwrap();
		let s1b = states.append_external_state(StateInput(12), &s1).unwrap();
		let s1 = states.append_external_state(StateInput(13), &s1b).unwrap();
		let sx = states.append_external_state(StateInput(14), &s1).unwrap();
		let qp14 = states.get_db_state(&StateInput(14)).unwrap();
		assert_eq!(states.drop_state(&sx, true).unwrap().len(), 1);
		let s3 = states.append_external_state(StateInput(3), &s1).unwrap();
		let s4 = states.append_external_state(StateInput(4), &s1).unwrap();
		let s5 = states.append_external_state(StateInput(5), &s1b).unwrap();
		// 0> 1: _ _ X
		// |			 |> 3: 1
		// |			 |> 4: 1
		// |		 |> 5: 1
		// |> 2: _ _
		let mut item3: Tree<u32, u32, u16, D, BD> = Default::default();
		let mut item4: Tree<u32, u32, u16, D, BD> = Default::default();
		item1.set(15, &states.get_db_state_mut(&StateInput(5)).unwrap());
		item2.set(15, &states.get_db_state_mut(&StateInput(5)).unwrap());
		item1.set(12, &states.get_db_state_mut(&StateInput(2)).unwrap());

		let s3head = states.append_external_state(StateInput(32), &s3).unwrap();
		item1.set(13, &states.get_db_state_mut(&StateInput(32)).unwrap());
		item2.set(13, &states.get_db_state_mut(&StateInput(32)).unwrap());
		item3.set(13, &states.get_db_state_mut(&StateInput(32)).unwrap());
		item4.set(13, &states.get_db_state_mut(&StateInput(32)).unwrap());
		let s3tmp = states.append_external_state(StateInput(33), &s3head).unwrap();
		item1.set(14, &states.get_db_state_mut(&StateInput(33)).unwrap());
		item3.set(0, &states.get_db_state_mut(&StateInput(33)).unwrap());
		let s3head = states.append_external_state(StateInput(34), &s3tmp).unwrap();
		let s6 = states.append_external_state(StateInput(6), &s3tmp).unwrap();
		let s3head = states.append_external_state(StateInput(35), &s3head).unwrap();
		item1.set(15, &states.get_db_state_mut(&StateInput(35)).unwrap());
		item2.set(15, &states.get_db_state_mut(&StateInput(35)).unwrap());
		item4.set(0, &states.get_db_state_mut(&StateInput(35)).unwrap());
		item1.set(0, &states.get_db_state_mut(&StateInput(6)).unwrap());

		let old_state = states.clone();
		// Apply change of composite to 33
		let filter_out = [101, 104, 2, 4, 5];
		let mut filter_qp = vec![qp14];
		for i in filter_out.iter() {
			let qp = states.get_db_state(&StateInput(*i)).unwrap();
			filter_qp.push(qp);
		}

		let fp = states.get_db_state(&StateInput(35)).unwrap();
		states.canonicalize(fp, s3tmp, None);
		let filter_in = [1, 102, 103, 105, 12, 13, 32, 33, 34, 35, 6];
		let no_qp = [14];
		//panic!("{:?} \n {:?}", old_state, states);
		let mut gc_item1 = item1.clone();
		let mut gc_item2 = item2.clone();
		let mut gc_item3 = item3.clone();
		let mut gc_item4 = item4.clone();
		{
			let gc = states.get_gc().unwrap();
			gc_item1.gc(gc.as_ref());
			gc_item2.gc(gc.as_ref());
			gc_item3.gc(gc.as_ref());
			gc_item4.gc(gc.as_ref());
			//panic!("{:?}", (gc.as_ref(), item4, gc_item4));
		}
		assert_eq!(gc_item1.nb_internal_history(), 8);
		assert_eq!(gc_item2.nb_internal_history(), 4);
		assert_eq!(gc_item3.nb_internal_history(), 2); // actually untouched
		assert_eq!(gc_item4.nb_internal_history(), 2); // actually untouched
		assert_eq!(gc_item1.nb_internal_branch(), 5);
		assert_eq!(gc_item2.nb_internal_branch(), 3);
		assert_eq!(gc_item3.nb_internal_branch(), 1);
		assert_eq!(gc_item4.nb_internal_branch(), 1);
		/* TODO only with new_start where we actually prune stuff
		for fp in filter_qp.iter() {
			assert_ne!(
				gc_item1.get_ref(fp),
				item1.get_ref(&fp),
			);
		}
		*/

		for i in filter_in.iter() {
			let fp = states.get_db_state(&StateInput(*i)).unwrap();
			assert_eq!(gc_item1.get_ref(&fp), item1.get_ref(&fp));
			assert_eq!(gc_item2.get_ref(&fp), item2.get_ref(&fp));
			assert_eq!(gc_item3.get_ref(&fp), item3.get_ref(&fp));
			assert_eq!(gc_item4.get_ref(&fp), item4.get_ref(&fp));
		}
//		panic!("{:?}", (gc, item1, gc_item1));

		// migrate 
		let filter_in = [33, 34, 35, 6];
		let mut gc_item1 = item1.clone();
		let mut gc_item2 = item2.clone();
		let mut gc_item3 = item3.clone();
		let mut gc_item4 = item4.clone();
		let mut states = states;
		{
			let (mig, mut gc) = states.get_migrate();
			gc_item1.migrate(&mut gc);
			gc_item2.migrate(&mut gc);
			gc_item3.migrate(&mut gc);
			gc_item4.migrate(&mut gc);
			states = mig.applied_migrate();
		}
		for i in filter_in.iter() {
			let fp = states.get_db_state(&StateInput(*i)).unwrap();
			assert_eq!(gc_item1.get_ref(&fp), item1.get_ref(&fp));
			assert_eq!(gc_item2.get_ref(&fp), item2.get_ref(&fp));
			assert_eq!(gc_item3.get_ref(&fp), item3.get_ref(&fp));
			assert_eq!(gc_item4.get_ref(&fp), item4.get_ref(&fp));
		}
		assert_eq!(gc_item1.nb_internal_history(), 8);
		assert_eq!(gc_item2.nb_internal_history(), 4);
		assert_eq!(gc_item3.nb_internal_history(), 2);
		assert_eq!(gc_item4.nb_internal_history(), 2);
		assert_eq!(gc_item1.nb_internal_branch(), 2);
		assert_eq!(gc_item2.nb_internal_branch(), 1);
		assert_eq!(gc_item3.nb_internal_branch(), 1);
		assert_eq!(gc_item4.nb_internal_branch(), 1);

		// on previous state set migrate with composite_treshold_new_start 
		let filter_in = [33, 34, 35, 6];
		let mut gc_item1 = item1.clone();
		let mut gc_item2 = item2.clone();
		let mut gc_item3 = item3.clone();
		let mut gc_item4 = item4.clone();
		let mut states = old_state;
		let fp = states.get_db_state(&StateInput(35)).unwrap();
		states.canonicalize(fp, s3tmp, Some(s3tmp.1));
		{
			let (mig, mut gc) = states.get_migrate();
			gc_item1.migrate(&mut gc);
			gc_item2.migrate(&mut gc);
			gc_item3.migrate(&mut gc);
			gc_item4.migrate(&mut gc);
			states = mig.applied_migrate();
			//panic!("{:?}", (gc, item3, gc_item3));
		}
		for i in filter_in.iter() {
			let fp = states.get_db_state(&StateInput(*i)).unwrap();
			assert_eq!(gc_item1.get_ref(&fp), item1.get_ref(&fp));
			assert_eq!(gc_item2.get_ref(&fp), item2.get_ref(&fp));
			assert_eq!(gc_item3.get_ref(&fp), None); // neutral element
			assert_eq!(gc_item4.get_ref(&fp), item4.get_ref(&fp));
		}
		assert_eq!(gc_item1.nb_internal_history(), 4);
		assert_eq!(gc_item2.nb_internal_history(), 2);
		assert_eq!(gc_item3.nb_internal_history(), 0);
		assert_eq!(gc_item4.nb_internal_history(), 2);
		assert_eq!(gc_item1.nb_internal_branch(), 2);
		assert_eq!(gc_item2.nb_internal_branch(), 1);
		assert_eq!(gc_item3.nb_internal_branch(), 0);
		assert_eq!(gc_item4.nb_internal_branch(), 1);
	}
}
