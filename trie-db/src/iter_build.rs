// Copyright 2017, 2019 Parity Technologies
//
// Licensed under the Apache License, Version .0 (the "License");
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

//! Alternative tools for working with key value ordered iterator without recursion.
//! This is iterative implementation of `trie_root` algorithm, using `NodeCodec`
//! implementation.
//! See `trie_visit` function.

use hash_db::{Hasher, HashDB, Prefix};
use crate::rstd::{cmp, marker::PhantomData, vec::Vec};
use crate::triedbmut::{ChildReference};
use crate::nibble::NibbleSlice;
use crate::nibble::nibble_ops;
use crate::node_codec::NodeCodec;
use crate::{TrieLayout, TrieHash};
use crate::partial_db::{IndexBackend, IndexPosition, IndexOrValue, Index as PartialIndex};


macro_rules! exponential_out {
	(@3, [$($inpp:expr),*]) => { exponential_out!(@2, [$($inpp,)* $($inpp),*]) };
	(@2, [$($inpp:expr),*]) => { exponential_out!(@1, [$($inpp,)* $($inpp),*]) };
	(@1, [$($inpp:expr),*]) => { [$($inpp,)* $($inpp),*] };
}

type CacheNode<HO> = Option<ChildReference<HO>>;

#[inline(always)]
fn new_vec_slice_buffer<HO>() -> [CacheNode<HO>; 16] {
	exponential_out!(@3, [None, None])
}

type ArrayNode<T> = [CacheNode<TrieHash<T>>; 16];

/// Struct containing iteration cache, can be at most the length of the lowest nibble.
struct CacheAccum<T: TrieLayout, V> (
	Vec<CacheElt<T, V>>,
);

/// Struct containing iteration cache, can be at most the length of the lowest nibble.
/// This is to use with indexing, and can store inconsistent data, therefore we got
/// we store a first element in these nodes.
struct CacheAccumIndex<T: TrieLayout, V> (
	Vec<CacheEltIndex<T, V>>,
	Option<(Vec<u8>, CacheElt<T, V>, usize)>,
);

impl<T: TrieLayout, V> Default for CacheAccumIndex<T, V> {
	fn default() -> Self {
		let v = Vec::with_capacity(INITIAL_DEPTH);
		CacheAccumIndex(v, None)
	}
}

/// State for incomming indexes in stack
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum BuffedElt {
	// no need to bufff.
	Nothing,
	// Buff could happen, if return, buff should happen.
	Buff,
	// branch buffed TODO here nibble as vec with full key, maybe find
	// something faster later (like small vec...).
	// Last usize is the stack depth of the buffed element.
	Branch,
	// value buffed
	// (we use same repr as branch as same size but children are empty here).
	// Last usize is the stack depth of the buffed element.
	Value,
}

impl Default for BuffedElt {
	fn default() -> Self {
		BuffedElt::Buff
	}
}

#[derive(Default, Debug)]
struct CacheElt<T: TrieLayout, V> {
	children: ArrayNode<T>,
	value: Option<V>,
	depth: usize,
}

#[derive(Default, Debug)]
struct CacheEltIndex<T: TrieLayout, V> {
	children: ArrayNode<T>,
	nb_children: usize,
	value: Option<V>,
	depth: usize,
	buffed: BuffedElt,
	is_index: bool,
}

impl<T: TrieLayout, V> CacheEltIndex<T, V> { 
	// Return true when we need to buff.
	// Update status to post buff status.
	// Only when necessary (ordered change
	// so no value (this will not change), and
	// no prior cached child, and no already
	// buffed.
	fn buff_transition(
		&mut self,
		is_branch: bool,
		is_overwrite: bool,
	) -> bool {
		match self.buffed {
			BuffedElt::Branch => {
				false
			},
			BuffedElt::Value => {
				false
			},
			BuffedElt::Nothing => false,
			BuffedElt::Buff => {
				if self.value.is_some() {
					return false;
				}
				if !is_overwrite && self.nb_children > 0 {
					return false;
				}
				if is_overwrite && self.nb_children > 1 {
					return false;
				}
				if !is_overwrite {
					self.nb_children += 1;
				}
				if is_branch {
					self.buffed = BuffedElt::Branch;
				} else {
					self.buffed = BuffedElt::Value;
				}
				true
			},
		}
	}
}

/// Initially allocated cache depth.
const INITIAL_DEPTH: usize = 10;

impl<T, V> CacheAccum<T, V>
	where
		T: TrieLayout,
		V: AsRef<[u8]> + Clone,
{
	fn new() -> Self {
		let v = Vec::with_capacity(INITIAL_DEPTH);
		CacheAccum(v)
	}

	#[inline(always)]
	fn set_cache_value(&mut self, depth:usize, value: Option<V>) {
		if self.0.is_empty() || self.0[self.0.len() - 1].depth < depth {
			self.0.push(CacheElt {
				children: new_vec_slice_buffer(),
				value: None,
				depth
			});
		}
		let last = self.0.len() - 1;
		debug_assert!(self.0[last].depth <= depth);
		self.0[last].value = value;
	}

	#[inline(always)]
	fn set_node(&mut self, depth: usize, nibble_index: usize, node: CacheNode<TrieHash<T>>) {
		if self.0.is_empty() || self.0[self.0.len() - 1].depth < depth {
			self.0.push(CacheElt {
				children: new_vec_slice_buffer(),
				value: None, 
				depth
			});
		}

		let last = self.0.len() - 1;
		debug_assert!(self.0[last].depth == depth);

		self.0[last].children.as_mut()[nibble_index] = node;
	}

	#[inline(always)]
	fn last_depth(&self) -> usize {
		let ix = self.0.len();
		if ix > 0 {
			let last = ix - 1;
			self.0[last].depth
		} else {
			0
		}
	}

	#[inline(always)]
	fn last_last_depth(&self) -> usize {
		let ix = self.0.len();
		if ix > 1 {
			let last = ix - 2;
			self.0[last].depth
		} else {
			0
		}
	}

	#[inline(always)]
	fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	#[inline(always)]
	fn is_one(&self) -> bool {
		self.0.len() == 1
	}

	#[inline(always)]
	fn reset_depth(&mut self, depth: usize) {
		debug_assert!(self.0[self.0.len() - 1].depth == depth);
		self.0.pop();
	}

	fn flush_value(
		&mut self,
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		target_depth: usize,
		(k2, v2): &(impl AsRef<[u8]>, impl AsRef<[u8]>),
	) {
		let nibble_value = nibble_ops::left_nibble_at(&k2.as_ref()[..], target_depth) as usize;

		let nkey = NibbleSlice::new_offset(&k2.as_ref()[..], target_depth + 1);
		let encoded = T::Codec::leaf_node(nkey.right(), &v2.as_ref()[..]);
		let pr = NibbleSlice::new_offset(
			&k2.as_ref()[..],
			k2.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE - nkey.len(),
		);
		let hash = callback.process(pr.left(), encoded, false, (k2.as_ref(), k2.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE), true);

		// insert hash in branch (first level branch only at this point)
		self.set_node(target_depth, nibble_value, Some(hash));
	}

	fn flush_branch(
		&mut self,
		no_extension: bool,
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		ref_branch: impl AsRef<[u8]> + Ord,
		new_depth: usize,
		is_last: bool,
	) {
		while self.last_depth() > new_depth || is_last && !self.is_empty() {
			let lix = self.last_depth();
/*
			// after index, we can go upward
			let (offset, lix, llix) = if upward {
				let llix = new_depth;
				(lix, llix, lix)
			} else {*/
				let llix = cmp::max(self.last_last_depth(), new_depth);
				let offset = llix;
		//		(llix, lix, llix)
		//	};

			let (offset, slice_size, is_root) = if llix == 0 && is_last && self.is_one() {
				// branch root
				(offset, lix - llix, true)
			} else {
				(offset + 1, lix - llix - 1, false)
			};
			let nkey = if slice_size > 0 {
				Some((offset, slice_size))
			} else {
				None
			};

			let h = if no_extension {
				// encode branch
				self.no_extension(&ref_branch.as_ref()[..], callback, lix, is_root, nkey)
			} else {
				self.standard_extension(&ref_branch.as_ref()[..], callback, lix, is_root, nkey)
			};
			if !is_root {
				// put hash in parent
				let nibble: u8 = nibble_ops::left_nibble_at(&ref_branch.as_ref()[..], llix);
				self.set_node(llix, nibble as usize, Some(h));
			}
		}
	}

	#[inline(always)]
	fn standard_extension(
		&mut self,
		key_branch: &[u8],
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		branch_d: usize,
		is_root: bool,
		nkey: Option<(usize, usize)>,
	) -> ChildReference<TrieHash<T>> {
		let last = self.0.len() - 1;
		assert_eq!(self.0[last].depth, branch_d);

		// encode branch
		let v = self.0[last].value.take();
		let encoded = T::Codec::branch_node(
			self.0[last].children.as_ref().iter(),
			v.as_ref().map(|v| v.as_ref()),
		);
		self.reset_depth(branch_d);
		let pr = NibbleSlice::new_offset(&key_branch, branch_d);
		// index value is incorect here, in fact we shall never index for extension (we shall use the
		// following branch)
		let branch_hash = callback.process(pr.left(), encoded, is_root && nkey.is_none(), (key_branch, branch_d + 1), false);

		if let Some(nkeyix) = nkey {
			let pr = NibbleSlice::new_offset(&key_branch, nkeyix.0);
			let nib = pr.right_range_iter(nkeyix.1);
			let encoded = T::Codec::extension_node(nib, nkeyix.1, branch_hash);
			let h = callback.process(pr.left(), encoded, is_root, (key_branch, branch_d + 1), false);
			h
		} else {
			branch_hash
		}
	}

	#[inline(always)]
	fn no_extension(
		&mut self,
		key_branch: &[u8],
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		branch_d: usize,
		is_root: bool,
		nkey: Option<(usize, usize)>,
	) -> ChildReference<TrieHash<T>> {
		let last = self.0.len() - 1;
		debug_assert!(self.0[last].depth == branch_d);
		// encode branch
		let v = self.0[last].value.take();
		let nkeyix = nkey.unwrap_or((0, 0));
		let pr = NibbleSlice::new_offset(&key_branch, nkeyix.0);
		let encoded = T::Codec::branch_node_nibbled(
			pr.right_range_iter(nkeyix.1),
			nkeyix.1,
			self.0[last].children.as_ref().iter(), v.as_ref().map(|v| v.as_ref()));
		self.reset_depth(branch_d);
		let ext_length = nkey.as_ref().map(|nkeyix| nkeyix.1).unwrap_or(0);
		let pr = NibbleSlice::new_offset(
			&key_branch,
			branch_d - ext_length,
		);
		callback.process(pr.left(), encoded, is_root, (key_branch, branch_d), false)
	}
}

/// Function visiting trie from key value inputs with a `ProccessEncodedNode` callback.
/// This is the main entry point of this module.
/// Calls to each node occurs ordered by byte key value but with longest keys first (from node to
/// branch to root), this differs from standard byte array ordering a bit.
pub fn trie_visit<T, I, A, B, F>(input: I, callback: &mut F)
	where
		T: TrieLayout,
		I: IntoIterator<Item = (A, B)>,
		A: AsRef<[u8]> + Ord,
		B: AsRef<[u8]> + Clone,
		F: ProcessEncodedNode<TrieHash<T>>,
{
	let no_extension = !T::USE_EXTENSION;
	let mut depth_queue = CacheAccum::<T, B>::new();
	// compare iter ordering
	let mut iter_input = input.into_iter();
	if let Some(mut previous_value) = iter_input.next() {
		// depth of last item
		let mut last_depth = 0;

		let mut single = true;
		for (k, v) in iter_input {
			single = false;
			let common_depth = nibble_ops::biggest_depth(&previous_value.0.as_ref()[..], &k.as_ref()[..]);
			// 0 is a reserved value : could use option
			let depth_item = common_depth;
			if common_depth == previous_value.0.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE {
				// the new key include the previous one : branch value case
				// just stored value at branch depth
				depth_queue.set_cache_value(common_depth, Some(previous_value.1));
			} else if depth_item >= last_depth {
				// put previous with next (common branch previous value can be flush)
				depth_queue.flush_value(callback, depth_item, &previous_value);
			} else if depth_item < last_depth {
				// do not put with next, previous is last of a branch
				depth_queue.flush_value(callback, last_depth, &previous_value);
				let ref_branches = previous_value.0;
				depth_queue.flush_branch(no_extension, callback, ref_branches, depth_item, false);
			}

			previous_value = (k, v);
			last_depth = depth_item;
		}
		// last pendings
		if single {
			// one single element corner case
			let (k2, v2) = previous_value;
			let nkey = NibbleSlice::new_offset(&k2.as_ref()[..], last_depth);
			let encoded = T::Codec::leaf_node(nkey.right(), &v2.as_ref()[..]);
			let pr = NibbleSlice::new_offset(
				&k2.as_ref()[..],
				k2.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE - nkey.len(),
			);
			callback.process(pr.left(), encoded, true, (k2.as_ref(), k2.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE), true);
		} else {
			depth_queue.flush_value(callback, last_depth, &previous_value);
			let ref_branches = previous_value.0;
			depth_queue.flush_branch(no_extension, callback, ref_branches, 0, true);
		}
	} else {
		// nothing null root corner case
		callback.process(hash_db::EMPTY_PREFIX, T::Codec::empty_node().to_vec(), true, (&[], 0), true); // define empty node as leaf to avoid storing it in indexes
	}
}

/// Get iterator on value and index only, past a given index (used to recalculate
/// unchanged index child when we fuse the index).
/// This is needed when fusing an index with parent, and the only child
/// was not modified, then we need to use another
/// iteration that goes into this modified child entries (only index and value there).
/// Note that the subiteration will get an unordered key (but only unordered with a
/// delete that do not trigger a callback, this is only relevant when extracting 
/// proof).
pub trait SubIter<A, B> {
	/// Sub iterate on value and index. 
	fn sub_iterate(
		&mut self,
		key: &[u8],
		depth: usize,
		child_index: usize,
		buffed: (A, IndexOrValue<B>),
	);
}

/// Same as `trie_visit` but allows to use some indexes node to skip part of the processing.
/// The function assumes that index and value from the input iterator do not overlap.
pub fn trie_visit_with_indexes<T, I, A, F>(input: I, callback: &mut F)
	where
		T: TrieLayout,
		I: Iterator<Item = (A, IndexOrValue<Vec<u8>>)> + SubIter<A, Vec<u8>>,
		A: AsRef<[u8]> + Ord,
		F: ProcessEncodedNode<TrieHash<T>>,
{
	assert!(!T::USE_EXTENSION, "No extension not implemented");
	let mut depth_queue = CacheAccumIndex::<T, Vec<u8>>::default();
	// compare iter ordering
	let mut iter_input = input.into_iter();
	let mut first = iter_input.next();
	while let Some((_k, IndexOrValue::DroppedValue)) = &first {
		// drop value without a first stacked item do nothing (they are guarantied to be
		// above an index).
		first = iter_input.next();
	};
	// root index cannot be written in stack (we write the index in their direct parent branch).
	debug_assert!(!if let Some((_k, IndexOrValue::Index(PartialIndex { actual_depth, .. }))) = &first {
		if *actual_depth == 0 {
			true
		} else {
			false
		}
	} else {
		false
	});
/*	if index_root {
		// we simply ignore this index as it we will need to query bellow, so it is useless to index
		// at depth 0. TODO consider ignoring from conf index 0 instead??
		first = iter_input.next();
	}*/
	if let Some((k, v)) = first {

		// can taint if consecutive in depth with an index
		let index_depth = v.index_depth().unwrap_or(k.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE);
		depth_queue.stack_item2(k.as_ref(), v, callback);

		let mut previous_key = (k, index_depth);

		while let Some((k, v)) = iter_input.next() {

			let common_depth = nibble_ops::biggest_depth(&previous_key.0.as_ref()[..], &k.as_ref()[..]);
			let common_depth = cmp::min(previous_key.1, common_depth);
			let (common_depth, depth) = if let IndexOrValue::Index(index) = &v {
				(cmp::min(common_depth, index.actual_depth - 1), index.actual_depth - 1)
			} else {
				(common_depth, k.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE)
			};
			// TODO this depth could be return by the iterator since internally we already compare keys.
			let mut last_stack_depth = depth_queue.last_depth().unwrap_or(0);
			let parent_depth = common_depth;
			let mut subiterat = None;
			while parent_depth < last_stack_depth {
				match depth_queue.unstack_item(previous_key.0.as_ref(), callback, Some(parent_depth)) {
					Some((new_depth, None)) => {
						last_stack_depth = new_depth;
					},
					Some((new_depth, Some(sub))) => {
						last_stack_depth = new_depth;
						subiterat = Some(sub);
						break;
					},
					None => {
						last_stack_depth = 0;
					},
				}
			}
			if let Some((new_iter_depth, new_iter_index)) = subiterat {
				iter_input.sub_iterate(previous_key.0.as_ref(), new_iter_depth, new_iter_index, (k, v));
				continue;
			}

			if let IndexOrValue::DroppedValue = &v {
				if parent_depth == last_stack_depth {
					depth_queue.taint_child(k.as_ref());
				} else {
					unreachable!("we did insert empty in unstack");
				}
				previous_key = (k, depth);
				continue;
			}

			if parent_depth > last_stack_depth {
				// TODO seems unreachable since parent/common is from the last stacked item
				// which in turn is used to calculate common thuse best parent == last
				// Need to add an intermediatory branch as sibling and then simply stack.
				depth_queue.stack_empty_branch(parent_depth);
			}
			
			depth_queue.stack_item2(k.as_ref(), v, callback);

			previous_key = (k, depth);
		}
		loop {
			match depth_queue.unstack_item(previous_key.0.as_ref(), callback, None) {
				Some((new_depth, None)) => (),
				Some((new_depth, Some((new_iter_depth, new_iter_index)))) => {
					unimplemented!("switch iterator (also need to save current)");
				},
				None => break,
			}
		}
	} else {
		// nothing null root corner case
		callback.process(hash_db::EMPTY_PREFIX, T::Codec::empty_node().to_vec(), true, (&[], 0), false);
	}
}


impl<T> CacheAccumIndex<T, Vec<u8>>
	where
		T: TrieLayout,
		//K: AsRef<[u8]> + Ord,
		//F: ProcessEncodedNode<TrieHash<T>>,
{
	#[inline(always)]
	fn last_depth(&self) -> Option<usize> {
		let ix = self.0.len();
		if ix > 0 {
			let last = ix - 1;
			Some(self.0[last].depth)
		} else {
			None
		}
	}

	fn stack_item2(
		&mut self,
		key: &[u8],
		item: IndexOrValue<Vec<u8>>,
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>
	) {
		match item {
			IndexOrValue::StoredValue(v)
			| IndexOrValue::Value(v) => {
				// having a value we can unbuff first child (key ordering ensure that).
				self.unbuff_first_child(callback);

				let depth = key.len() * nibble_ops::NIBBLE_PER_BYTE;
				debug_assert!(self.last_depth().map(|last| last < depth).unwrap_or(true));
				self.0.push(CacheEltIndex {
					children: Default::default(),
					nb_children: 0,
					value: Some(v),
					depth,
					buffed: BuffedElt::Buff,
					is_index: false,
				})
			},
			IndexOrValue::Index(PartialIndex { hash, actual_depth }) => {
				debug_assert!(actual_depth > 0);
				let branch_ix = actual_depth - 1;
				let parent_ix = nibble_ops::left_nibble_at(key.as_ref(), branch_ix) as usize;

				let mut h_hash = TrieHash::<T>::default();  // TODO puth actual hash type in indexes (useless copy here)
				h_hash.as_mut()[..].copy_from_slice(hash.as_slice());
				let hash = ChildReference::Hash(h_hash);
				debug_assert!(self.last_depth().map(|last| last <= branch_ix).unwrap_or(true));
				let new_elt = if let Some(element) = self.0.last_mut() {
					if element.depth == branch_ix {
						element.children[parent_ix] = Some(hash);
						false
					} else {
						true
					}
				} else {
					true
				};
				if new_elt {
					let mut element = CacheEltIndex {
						children: Default::default(),
						nb_children: 1,
						value: None,
						depth: branch_ix,
						buffed: BuffedElt::Buff,
						is_index: true,
					};
					element.children[parent_ix] = Some(hash);
					self.0.push(element);
				}
			},
			IndexOrValue::DroppedValue => {
				unreachable!();
			},
		}
	}

	fn taint_child(&mut self, key_branch: &[u8]) {
		if let Some(CacheEltIndex { children, depth, is_index, nb_children, buffed, value, .. }) = self.0.last_mut() {
			if *is_index {
				let ix = NibbleSlice::new(key_branch.as_ref()).at(*depth) as usize;
				if children[ix].is_some() {
					*nb_children = *nb_children - 1;
				}
				children[ix] = None;
			}
		}
	}

	fn unstack_item(
		&mut self,
		current_key: &[u8],
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		target_parent_depth: Option<usize>,
	) -> Option<(usize, Option<(usize, usize)>)> {
		enum Action {
			UnstackValue,
			UnstackBranch,
			FuseParent,
			Ignore,
		};
		let (action, do_unbuff) = if let Some(
			CacheEltIndex { value, nb_children, .. }
		) = self.0.last() {
			if value.is_some() && nb_children == &0 {
				(Action::UnstackValue, true)
			} else if value.is_some() {
				(Action::UnstackBranch, true)
			} else if nb_children > &1 {
				(Action::UnstackBranch, true)
			} else if nb_children == &1 {
				(Action::FuseParent, false)
			} else {
				(Action::Ignore, false)
			}
		} else {
			(Action::Ignore, false)
		};
		if do_unbuff {
			self.unbuff_first_child(callback);
		}
		if let Some(
			CacheEltIndex { children, depth, is_index, value, buffed, nb_children: _ }
		) = self.0.pop() {
			if let Some(target) = target_parent_depth {
				let parent_depth = self.last_depth();
				if parent_depth.map(|parent| parent < target)
					.unwrap_or(true) {
					// push empty at parent
					self.stack_empty_branch(target);
				}
			}

			let key = current_key;
			let stack_len = self.0.len();
			// first child buffering
			if let Some(par) = self.0.last_mut() {
				let parent_ix = nibble_ops::left_nibble_at(key.as_ref(), par.depth) as usize;
				let is_overwrite = par.children[parent_ix].is_some();
				if let Some(do_stack) = match action {
					Action::UnstackBranch => Some(par.buff_transition(true, is_overwrite)),
					Action::UnstackValue => Some(par.buff_transition(false, is_overwrite)),
					_ => None,
				} {
					if do_stack {
						self.1 = Some((key.to_vec(), CacheElt {
							children,
							value,
							depth,
						}, stack_len - 1));

						return Some((par.depth, None));
					}
				}
			}
	
			if let Some(
				CacheEltIndex { children: parent_children, depth: parent_depth, nb_children: parent_nb_children, buffed: parent_buffed, .. }
			) = self.0.last_mut() {
				// not root
				let pr = NibbleSlice::new_offset(
					key.as_ref(),
					*parent_depth + 1,
				);

				let encoded = match action {
					Action::UnstackValue => {
						debug_assert!(value.is_some());
						let nkey = NibbleSlice::new_offset(&key[..depth / nibble_ops::NIBBLE_PER_BYTE], *parent_depth + 1);
						assert_eq!(*parent_depth, depth - nkey.len() - 1);
						T::Codec::leaf_node(nkey.right(), value.unwrap_or_default().as_ref())
					},
					Action::UnstackBranch => {
						let pr = NibbleSlice::new_offset(
							key.as_ref(),
							*parent_depth + 1,
						);
						let partial_length = depth - *parent_depth - 1;
						T::Codec::branch_node_nibbled(
							pr.right_range_iter(partial_length),
							partial_length,
							children.as_ref().iter(),
							value.as_ref().map(|v| v.as_ref()),
						)	
					},
					Action::FuseParent => {
						debug_assert!(is_index || children.iter().find(|c| c.is_some()).is_none());
						// first buff goes upward
						if let Some((key, _cache, buffed_stack_depth)) = self.1.as_mut() {
							//debug_assert!(*buffed_stack_depth == stack_len + 1); not true due to insert of empty br
							debug_assert!(buffed != BuffedElt::Buff);
							debug_assert!(buffed != BuffedElt::Nothing);
							let parent_ix = nibble_ops::left_nibble_at(key.as_ref(), *parent_depth) as usize;
							if parent_children[parent_ix].is_none() {
								*parent_nb_children = *parent_nb_children + 1;
							}
							*buffed_stack_depth = stack_len - 1;
							*parent_buffed = buffed;
							return Some((*parent_depth, None));
						} else {
							debug_assert!(is_index);
							let child_index = children.iter().position(|c| c.is_some())
								.expect("checked one index and no first buff before");
							// index is invalidated here (cannot use existing child hash since it will change post fuse)
							// the iterator must therefore goes bellow this index next child.
							// And this branch is already deleted.
							return self.last_depth().map(|d| (d, Some((depth, child_index))));
						}
					},
					Action::Ignore => return self.last_depth().map(|d| (d, None)),
				};
				let hash = callback.process(
					pr.left(),
					encoded,
					false,
					(key.as_ref(), depth),
					false,
				);
				let parent_ix = nibble_ops::left_nibble_at(key.as_ref(), *parent_depth) as usize;
				if parent_children[parent_ix].is_none() {
					*parent_nb_children = *parent_nb_children + 1;
				}
				parent_children[parent_ix] = Some(hash);
				return Some((*parent_depth, None));
			} else {
				// root
				match action {
					Action::UnstackValue => {
						let nkey = NibbleSlice::new_offset(&key[..depth / nibble_ops::NIBBLE_PER_BYTE], 0);
						let encoded = T::Codec::leaf_node(nkey.right(), value.unwrap_or_default().as_ref());
						callback.process(
							hash_db::EMPTY_PREFIX,
							encoded,
							true,
							(key.as_ref(), depth),
							false,
						);
						return None;
					},
					Action::UnstackBranch => {
						let pr = NibbleSlice::new_offset(key.as_ref(), 0);
						let partial_length = depth;
						let encoded = T::Codec::branch_node_nibbled(
							pr.right_range_iter(partial_length),
							partial_length,
							children.as_ref().iter(),
							value.as_ref().map(|v| v.as_ref()),
						);
						callback.process(
							pr.left(),
							encoded,
							true,
							(key.as_ref(), depth),
							false,
						);
						return None;
					},
					Action::FuseParent => {
						let stack_len = self.0.len();
						// buffed is root
						if let Some((key, CacheElt { children, value, depth }, stack_depth)) = self.1.take() {
							//debug_assert!(stack_depth == stack_len - 1);
							match buffed {
								BuffedElt::Value => {
									let nkey = NibbleSlice::new_offset(&key[..depth / nibble_ops::NIBBLE_PER_BYTE], 0);
									let encoded = T::Codec::leaf_node(nkey.right(), value.unwrap_or_default().as_ref());
									callback.process(
										hash_db::EMPTY_PREFIX,
										encoded,
										true,
										(key.as_ref(), depth),
										false,
									);
								},
								BuffedElt::Branch => {
									let pr = NibbleSlice::new_offset(key.as_ref(), 0);
									let encoded = T::Codec::branch_node_nibbled(
										pr.right_range_iter(depth),
										depth,
										children.as_ref().iter(),
										value.as_ref().map(|v| v.as_ref()),
									);
									callback.process(
										hash_db::EMPTY_PREFIX,
										encoded,
										true,
										(key.as_ref(), depth),
										false,
									);
								},
								_ => unreachable!(),
							}
						} else {
							unreachable!("fuse only with buffed child");
						}
						return None;
					},
					Action::Ignore => return self.last_depth().map(|d| (d, None)),
				}
			}
		}
		self.last_depth().map(|d| (d, None))
	}

	fn unbuff_first_child(&mut self, callback: &mut impl ProcessEncodedNode<TrieHash<T>>) {
		if let Some((key, CacheElt { children, value, depth }, stack_depth)) = self.1.take() {
			debug_assert!(self.0.len() > stack_depth);
			let parent_depth = self.0[stack_depth].depth;
			if let Some(parent) = self.0.get_mut(stack_depth) {
				let hash = match parent.buffed {
					BuffedElt::Branch => {
						let pr = NibbleSlice::new_offset(
							key.as_ref(),
							parent_depth + 1,
						);
						let partial_length = depth - parent_depth - 1;

						let encoded = T::Codec::branch_node_nibbled(
							pr.right_range_iter(partial_length),
							partial_length,
							children.as_ref().iter(),
							value.as_ref().map(|v| v.as_ref()),
						);
						callback.process(
							pr.left(),
							encoded,
							false,
							(key.as_ref(), depth),
							false,
						)
					},
					BuffedElt::Value => {
						debug_assert!(value.is_some());
						debug_assert!(depth == key.len() * nibble_ops::NIBBLE_PER_BYTE);
						let nkey = NibbleSlice::new_offset(key.as_ref(), parent_depth + 1);
						let encoded = T::Codec::leaf_node(nkey.right(), value.unwrap_or_default().as_ref());
						assert_eq!(parent_depth, depth - nkey.len() - 1);
						let pr = NibbleSlice::new_offset(
							key.as_ref(),
							parent_depth + 1,
						);
						callback.process(
							pr.left(),
							encoded,
							false,
							(key.as_ref(), depth),
							true,
						)
					},
					BuffedElt::Nothing
					| BuffedElt::Buff => return,
				};
				let nibble_index = nibble_ops::left_nibble_at(key.as_ref(), parent_depth);

				// No nb_children increase, this was already done when buffing the first child.
				parent.children.as_mut()[nibble_index as usize] = Some(hash);
				parent.buffed = BuffedElt::Nothing;
			}
		}
	}

	fn stack_empty_branch(&mut self, depth: usize) {
		debug_assert!(self.last_depth().map(|last| last < depth).unwrap_or(true));
		self.0.push(CacheEltIndex {
			children: Default::default(),
			nb_children: 0,
			value: None,
			depth,
			buffed: BuffedElt::Buff,
			is_index: false,
		});
	}
	
}

/// Visitor trait to implement when using `trie_visit`.
pub trait ProcessEncodedNode<HO> {
	/// Function call with prefix, encoded value and a boolean indicating if the
	/// node is the root for each node of the trie.
	/// Last parameter `node_key` is the path to the node as a slice of bytes and the depth in bit of this node.
	///
	/// Note that the returned value can change depending on implementation,
	/// but usually it should be the Hash of encoded node.
	/// This is not something direcly related to encoding but is here for
	/// optimisation purpose (builder hash_db does return this value).
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
		node_key: (&[u8], usize),
		is_leaf: bool,
	) -> ChildReference<HO>;
}

/// Get trie root and insert visited node in a hash_db.
/// As for all `ProcessEncodedNode` implementation, it
/// is only for full trie parsing (not existing trie).
pub struct TrieBuilder<'a, H, HO, V, DB> {
	db: &'a mut DB,
	pub root: Option<HO>,
	_ph: PhantomData<(H, V)>,
}

impl<'a, H, HO, V, DB> TrieBuilder<'a, H, HO, V, DB> {
	pub fn new(db: &'a mut DB) -> Self {
		TrieBuilder { db, root: None, _ph: PhantomData }
	}
}

impl<'a, H: Hasher, V, DB: HashDB<H, V>> ProcessEncodedNode<<H as Hasher>::Out>
	for TrieBuilder<'a, H, <H as Hasher>::Out, V, DB> {
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
		_node_key: (&[u8], usize),
		_is_leaf: bool,
	) -> ChildReference<<H as Hasher>::Out> {
		let len = encoded_node.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len);
		}
		let hash = self.db.insert(prefix, &encoded_node[..]);
		if is_root {
			self.root = Some(hash);
		};
		ChildReference::Hash(hash)
	}
}

/// Get trie root and insert index for trie.
pub struct TrieRootIndexes<'a, H, HO, DB> {
	db: &'a mut DB,
	pub root: Option<HO>,
	indexes: &'a crate::partial_db::DepthIndexes,
	_ph: PhantomData<H>,
}

impl<'a, H, HO, DB> TrieRootIndexes<'a, H, HO, DB> {
	pub fn new(db: &'a mut DB, indexes: &'a crate::partial_db::DepthIndexes) -> Self {
		TrieRootIndexes { db, indexes, root: None, _ph: PhantomData }
	}
}

impl<'a, H: Hasher, DB: IndexBackend> ProcessEncodedNode<<H as Hasher>::Out>
	for TrieRootIndexes<'a, H, <H as Hasher>::Out, DB> {
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
		node_key: (&[u8], usize),
		is_leaf: bool, // TODO remove this param
	) -> ChildReference<<H as Hasher>::Out> {
		let len = encoded_node.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len);
		}
		let hash = <H as Hasher>::hash(&encoded_node[..]);
		let prefix_start = prefix.0.len() * nibble_ops::NIBBLE_PER_BYTE + if prefix.1.is_some() { 1 } else { 0 };
		let index_position: IndexPosition = node_key.0.into();
		if let Some(next_save_index) = self.indexes.next_depth(prefix_start, &index_position) {
			if node_key.1 >= next_save_index {
				let partial_index = PartialIndex {
					hash: hash.as_ref().to_vec(),
					actual_depth: prefix_start,
				};
				// TODO consider changing write to reference input
				self.db.write(next_save_index, node_key.0.into(), partial_index);
			}
		}
		if is_root {
			self.root = Some(hash);
		};
		ChildReference::Hash(hash)
	}
}

/// Calculate the trie root of the trie.
pub struct TrieRoot<H, HO> {
	/// The resulting root.
	pub root: Option<HO>,
	_ph: PhantomData<H>,
}

impl<H, HO> Default for TrieRoot<H, HO> {
	fn default() -> Self {
		TrieRoot { root: None, _ph: PhantomData }
	}
}

impl<H: Hasher> ProcessEncodedNode<<H as Hasher>::Out> for TrieRoot<H, <H as Hasher>::Out> {
	fn process(
		&mut self,
		_: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
		_node_key: (&[u8], usize),
		_is_leaf: bool,
	) -> ChildReference<<H as Hasher>::Out> {
		let len = encoded_node.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len);
		}
		let hash = <H as Hasher>::hash(&encoded_node[..]);
		if is_root {
			self.root = Some(hash);
		};
		ChildReference::Hash(hash)
	}
}

/// Get the trie root node encoding.
pub struct TrieRootUnhashed<H> {
	/// The resulting encoded root.
	pub root: Option<Vec<u8>>,
	_ph: PhantomData<H>,
}

impl<H> Default for TrieRootUnhashed<H> {
	fn default() -> Self {
		TrieRootUnhashed { root: None, _ph: PhantomData }
	}
}

#[cfg(feature = "std")]
/// Calculate the trie root of the trie.
/// Print a debug trace.
pub struct TrieRootPrint<H, HO> {
	/// The resulting root.
	pub root: Option<HO>,
	_ph: PhantomData<H>,
}

#[cfg(feature = "std")]
impl<H, HO> Default for TrieRootPrint<H, HO> {
	fn default() -> Self {
		TrieRootPrint { root: None, _ph: PhantomData }
	}
}

#[cfg(feature = "std")]
impl<H: Hasher> ProcessEncodedNode<<H as Hasher>::Out> for TrieRootPrint<H, <H as Hasher>::Out> {
	fn process(
		&mut self,
		p: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
		_node_key: (&[u8], usize),
		_is_leaf: bool,
	) -> ChildReference<<H as Hasher>::Out> {
		println!("Encoded node: {:x?}", &encoded_node);
		println!("	with prefix: {:x?}", &p);
		let len = encoded_node.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			println!("	inline len {}", len);
			return ChildReference::Inline(h, len);
		}
		let hash = <H as Hasher>::hash(&encoded_node[..]);
		if is_root {
			self.root = Some(hash);
		};
		println!("	hashed to {:x?}", hash.as_ref());
		ChildReference::Hash(hash)
	}
}

impl<H: Hasher> ProcessEncodedNode<<H as Hasher>::Out> for TrieRootUnhashed<H> {
	fn process(
		&mut self,
		_: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
		_node_key: (&[u8], usize),
		_is_leaf: bool,
	) -> ChildReference<<H as Hasher>::Out> {
		let len = encoded_node.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len);
		}
		let hash = <H as Hasher>::hash(&encoded_node[..]);
		if is_root {
			self.root = Some(encoded_node);
		};
		ChildReference::Hash(hash)
	}
}

#[cfg(test)]
mod test {
	use crate::DBValue;
	use memory_db::{MemoryDB, HashKey, PrefixedKey};
	use keccak_hasher::KeccakHasher;

	#[test]
	fn trie_root_empty () {
		compare_implementations(vec![])
	}

	#[test]
	fn trie_one_node () {
		compare_implementations(vec![
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8]),
		]);
	}

	#[test]
	fn root_extension_one () {
		compare_implementations(vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
		]);
	}

	fn test_iter(data: Vec<(Vec<u8>, Vec<u8>)>) {
		use reference_trie::{RefTrieDBMut, TrieMut, RefTrieDB, Trie};

		let mut db = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut db, &mut root);
			for i in 0..data.len() {
				let key: &[u8]= &data[i].0;
				let value: &[u8] = &data[i].1;
				t.insert(key, value).unwrap();
			}
		}
		let t = RefTrieDB::new(&db, &root).unwrap();
		for (i, kv) in t.iter().unwrap().enumerate() {
			let (k, v) = kv.unwrap();
			let key: &[u8]= &data[i].0;
			let value: &[u8] = &data[i].1;
			assert_eq!(k, key);
			assert_eq!(v, value);
		}
		for (k, v) in data.into_iter() {
			assert_eq!(&t.get(&k[..]).unwrap().unwrap()[..], &v[..]);
		}
	}

	fn test_iter_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
		use reference_trie::{RefTrieDBMutNoExt, TrieMut, RefTrieDBNoExt, Trie};

		let mut db = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut db, &mut root);
			for i in 0..data.len() {
				let key: &[u8]= &data[i].0;
				let value: &[u8] = &data[i].1;
				t.insert(key, value).unwrap();
			}
		}
		let t = RefTrieDBNoExt::new(&db, &root).unwrap();
		for (i, kv) in t.iter().unwrap().enumerate() {
			let (k, v) = kv.unwrap();
			let key: &[u8]= &data[i].0;
			let value: &[u8] = &data[i].1;
			assert_eq!(k, key);
			assert_eq!(v, value);
		}
		for (k, v) in data.into_iter() {
			assert_eq!(&t.get(&k[..]).unwrap().unwrap()[..], &v[..]);
		}
	}

	fn compare_implementations(data: Vec<(Vec<u8>, Vec<u8>)>) {
		test_iter(data.clone());
		test_iter_no_extension(data.clone());
		compare_implementations_h(data.clone());
		compare_implementations_prefixed(data.clone());
		compare_implementations_no_extension(data.clone());
		compare_implementations_no_extension_prefixed(data.clone());
		compare_indexing(data.clone());
	}
	fn compare_indexing(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		let mut indexes = std::collections::BTreeMap::new();
		let indexes_conf = reference_trie::DepthIndexes::new(&[
			1, 2, 4, 6, 9,
		]);
		reference_trie::compare_indexing(data, memdb, &mut indexes, &indexes_conf);
	}
	fn compare_implementations_prefixed(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		let hashdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		reference_trie::compare_implementations(data, memdb, hashdb);
	}
	fn compare_implementations_h(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, HashKey<_>, _>::default();
		let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
		reference_trie::compare_implementations(data, memdb, hashdb);
	}
	fn compare_implementations_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, HashKey<_>, _>::default();
		let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
		reference_trie::compare_implementations_no_extension(data, memdb, hashdb);
	}
	fn compare_implementations_no_extension_prefixed(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		let hashdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		reference_trie::compare_implementations_no_extension(data, memdb, hashdb);
	}
	fn compare_implementations_no_extension_unordered(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, HashKey<_>, _>::default();
		let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
		reference_trie::compare_implementations_no_extension_unordered(data, memdb, hashdb);
	}
	fn compare_no_extension_insert_remove(data: Vec<(bool, Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		reference_trie::compare_no_extension_insert_remove(data, memdb);
	}
	fn compare_root(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, HashKey<_>, _>::default();
		reference_trie::compare_root(data, memdb);
	}
	fn compare_unhashed(data: Vec<(Vec<u8>, Vec<u8>)>) {
		reference_trie::compare_unhashed(data);
	}
	fn compare_unhashed_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
		reference_trie::compare_unhashed_no_extension(data);
	}

	// Following tests are a bunch of detected issue here for non regression.

	#[test]
	fn trie_middle_node1 () {
		compare_implementations(vec![
			(vec![1u8, 2u8], vec![8u8;32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
		]);
	}
	#[test]
	fn trie_middle_node2 () {
		compare_implementations(vec![
			(vec![0u8, 2u8, 3u8, 5u8, 3u8], vec![1u8;32]),
			(vec![1u8, 2u8], vec![8u8;32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
			(vec![1u8, 2u8, 3u8, 5u8], vec![7u8;32]),
			(vec![1u8, 2u8, 3u8, 5u8, 3u8], vec![7u8;32]),
		]);
	}
	#[test]
	fn root_extension_bis () {
		compare_root(vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
		]);
	}
	#[test]
	fn root_extension_tierce () {
		let d = vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;2]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;2]),
		];
		compare_unhashed(d.clone());
		compare_unhashed_no_extension(d);
	}
	#[test]
	fn root_extension_tierce_big () {
		// on more content unhashed would hash
		compare_unhashed(vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
			(vec![1u8, 6u8, 3u8, 3u8], vec![8u8;32]),
			(vec![6u8, 2u8, 3u8, 3u8], vec![8u8;32]),
			(vec![6u8, 2u8, 3u8, 13u8], vec![8u8;32]),
		]);
	}
	#[test]
	fn trie_middle_node2x () {
		compare_implementations(vec![
			(vec![0u8, 2u8, 3u8, 5u8, 3u8], vec![1u8;2]),
			(vec![1u8, 2u8], vec![8u8;2]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;2]),
			(vec![1u8, 2u8, 3u8, 5u8], vec![7u8;2]),
			(vec![1u8, 2u8, 3u8, 5u8, 3u8], vec![7u8;2]),
		]);
	}
	#[test]
	fn fuzz1 () {
		compare_implementations(vec![
			(vec![01u8], vec![42u8, 9]),
			(vec![01u8, 0u8], vec![0u8, 0]),
			(vec![255u8, 2u8], vec![1u8, 0]),
		]);
	}
	#[test]
	fn fuzz2 () {
		compare_implementations(vec![
			(vec![0, 01u8], vec![42u8, 9]),
			(vec![0, 01u8, 0u8], vec![0u8, 0]),
			(vec![0, 255u8, 2u8], vec![1u8, 0]),
		]);
	}
	#[test]
	fn fuzz3 () {
		compare_implementations(vec![
			(vec![0], vec![196, 255]),
			(vec![48], vec![138, 255]),
			(vec![67], vec![0, 0]),
			(vec![128], vec![255, 0]),
			(vec![247], vec![0, 196]),
			(vec![255], vec![0, 0]),
		]);
	}
	#[test]
	fn fuzz_no_extension1 () {
		compare_implementations(vec![
			(vec![0], vec![128, 0]),
			(vec![128], vec![0, 0]),
		]);
	}
	#[test]
	fn fuzz_no_extension2 () {
		compare_implementations(vec![
			(vec![0], vec![6, 255]),
			(vec![6], vec![255, 186]),
			(vec![255], vec![186, 255]),
		]);
	}
	#[test]
	fn fuzz_no_extension5 () {
		compare_implementations(vec![
			(vec![0xaa], vec![0xa0]),
			(vec![0xaa, 0xaa], vec![0xaa]),
			(vec![0xaa, 0xbb], vec![0xab]),
			(vec![0xbb], vec![0xb0]),
			(vec![0xbb, 0xbb], vec![0xbb]),
			(vec![0xbb, 0xcc], vec![0xbc]),
		]);
	}
	#[test]
	fn fuzz_no_extension3 () {
		compare_implementations(vec![
			(vec![0], vec![0, 0]),
			(vec![11, 0], vec![0, 0]),
			(vec![11, 252], vec![11, 0]),
		]);

		compare_implementations_no_extension_unordered(vec![
			(vec![11, 252], vec![11, 0]),
			(vec![11, 0], vec![0, 0]),
			(vec![0], vec![0, 0]),
		]);
	}
	#[test]
	fn fuzz_no_extension4 () {
		compare_implementations_no_extension(vec![
			(vec![0x01, 0x56], vec![0x1]),
			(vec![0x02, 0x42], vec![0x2]),
			(vec![0x02, 0x50], vec![0x3]),
		]);
	}
	#[test]
	fn fuzz_no_extension_insert_remove_1 () {
		let data = vec![
			(false, vec![0], vec![251, 255]),
			(false, vec![0, 1], vec![251, 255]),
			(false, vec![0, 1, 2], vec![255; 32]),
			(true, vec![0, 1], vec![0, 251]),
		];
		compare_no_extension_insert_remove(data);
	}
	#[test]
	fn fuzz_no_extension_insert_remove_2 () {
		let data = vec![
			(false, vec![0x00], vec![0xfd, 0xff]),
			(false, vec![0x10, 0x00], vec![1;32]),
			(false, vec![0x11, 0x10], vec![0;32]),
			(true, vec![0x10, 0x00], vec![])
		];
		compare_no_extension_insert_remove(data);
	}
	#[test]
	fn two_bytes_nibble_length () {
		let data = vec![
			(vec![00u8], vec![0]),
			(vec![01u8;64], vec![0;32]),
		];
		compare_implementations_no_extension(data.clone());
		compare_implementations_no_extension_prefixed(data.clone());
	}
	#[test]
	#[should_panic]
	fn too_big_nibble_length_old () {
		compare_implementations_h(vec![
			(vec![01u8;64], vec![0;32]),
		]);
	}
	#[test]
	fn too_big_nibble_length_new () {
		compare_implementations_no_extension(vec![
			(vec![01u8;((u16::max_value() as usize + 1) / 2) + 1], vec![0;32]),
		]);
	}
	#[test]
	fn polka_re_test () {
		compare_implementations(vec![
			(vec![77, 111, 111, 55, 111, 104, 121, 97], vec![68, 97, 105, 55, 105, 101, 116, 111]),
			(vec![101, 105, 67, 104, 111, 111, 66, 56], vec![97, 56, 97, 113, 117, 53, 97]),
			(vec![105, 97, 48, 77, 101, 105, 121, 101], vec![69, 109, 111, 111, 82, 49, 97, 105]),
		]);
	}

	fn compare_index_calc(
		data: Vec<(Vec<u8>, Vec<u8>)>,
		change: Vec<(Vec<u8>, Option<Vec<u8>>)>,
		depth_indexes: Vec<u32>,
		nb_node_fetch: Option<usize>,
	) {
		let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		let mut indexes = std::collections::BTreeMap::new();
		let indexes_conf = reference_trie::DepthIndexes::new(&depth_indexes[..]);
		reference_trie::compare_index_calc(data, change, memdb, &mut indexes, &indexes_conf, nb_node_fetch);
	}

	#[test]
	fn compare_index_calculations() {
		let empty = vec![];
		let one_level_branch = vec![
			(b"test".to_vec(), vec![2u8; 32]),
			(b"tett".to_vec(), vec![3u8; 32]),
			(b"teut".to_vec(), vec![4u8; 32]),
			(b"tevtc".to_vec(), vec![5u8; 32]),
			(b"tewtb".to_vec(), vec![6u8; 32]),
			(b"tezta".to_vec(), vec![6u8; 32]),
		];
		let two_level_branch = vec![
			(b"test".to_vec(), vec![2u8; 32]),
			(b"testi".to_vec(), vec![2u8; 32]),
			(b"tett".to_vec(), vec![3u8; 32]),
			(b"tetti".to_vec(), vec![3u8; 32]),
			(b"teut".to_vec(), vec![4u8; 32]),
			(b"teuti".to_vec(), vec![4u8; 32]),
			(b"tevtc".to_vec(), vec![5u8; 32]),
			(b"tevtci".to_vec(), vec![5u8; 32]),
			(b"tewtb".to_vec(), vec![6u8; 32]),
			(b"tewtbi".to_vec(), vec![6u8; 32]),
			(b"tezta".to_vec(), vec![6u8; 32]),
			(b"teztai".to_vec(), vec![6u8; 32]),
		];

		let inputs = vec![
//			(one_level_branch.clone(), vec![(b"testi".to_vec(), Some(vec![12; 32]))], vec![5], Some(1)),
			(empty.clone(), vec![], vec![], Some(0)),
			(empty.clone(), vec![], vec![2, 5], Some(0)),
			(empty.clone(), vec![(b"te".to_vec(), None)], vec![2, 5], Some(0)),
			(empty.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![], Some(0)),
			(empty.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![8, 20], Some(0)),

			(one_level_branch.clone(), vec![], vec![], Some(0)),
			(one_level_branch.clone(), vec![], vec![2, 5], Some(0)),
			(one_level_branch.clone(), vec![], vec![5], Some(0)),
			(one_level_branch.clone(), vec![], vec![6], Some(0)),
			(one_level_branch.clone(), vec![], vec![7], Some(0)),
			(one_level_branch.clone(), vec![], vec![6, 7], Some(0)),

			// insert before indexes
			// index one child
			(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![5], Some(1)),
			// index on children
			(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![7], Some(1)),
			(two_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![7], Some(1)),
			// index after children
			(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![10], Some(1)),
			(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![10], Some(1)),
			(two_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![10], Some(1)),

			// insert onto indexes
			// insert after indexes
		];
		for (data, change, depth_indexes, nb_fetch) in inputs.into_iter() {
			compare_index_calc(data, change, depth_indexes, nb_fetch);
		}
	}
}
