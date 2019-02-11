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

//! Alternative tools for working with key value iterator without recursion.

use elastic_array::ElasticArray36;
use hash_db::{Hasher, HashDB};
use std::marker::PhantomData;
use crate::triedbmut::{ChildReference};
use crate::nibbleslice::NibbleSlice;
use node_codec::NodeCodec;


fn biggest_depth(v1: &[u8], v2: &[u8]) -> usize {
	//for a in 0.. v1.len(), v2.len()) { sorted assertion preventing out of bound TODO fuzz that
	for a in 0.. v1.len() {
		if v1[a] == v2[a] {
		} else {
			if (v1[a] >> 4) ==	(v2[a] >> 4) {
				return a * 2 + 1;
			} else {
				return a * 2;
			}
		}
	}
	return v1.len() * 2;
}

// warn! start at 0 // TODO change biggest_depth??
// warn! slow don't loop on that when possible
fn nibble_at(v1: &[u8], ix: usize) -> u8 {
	if ix % 2 == 0 {
		v1[ix/2] >> 4
	} else {
		v1[ix/2] & 15
	}
}

// TODO remove for nibbleslice api TODO can be variable size
fn encoded_nibble(ori: &[u8], is_leaf: bool) -> ElasticArray36<u8> {
	let l = ori.len();
	let mut r = ElasticArray36::new();
	let mut i = l % 2;
	r.push(if i == 1 {0x10 + ori[0]} else {0} + if is_leaf {0x20} else {0});
	while i < l {
		r.push(ori[i] * 16 + ori[i+1]);
		i += 2;
	}
	r
}


// (64 * 16) aka 2*byte size of key * nb nibble value, 2 being byte/nible (8/4)
// TODO test others layout
// first usize to get nb of added value, second usize last added index
// second str is in branch value and can be remove for fix key scenario
struct CacheAccum<H: Hasher,C,V> (Vec<(Vec<CacheNode<<H as Hasher>::Out>>, usize, usize)>,Vec<Option<V>>,PhantomData<(H,C)>);

/// initially allocated cache
const INITIAL_DEPTH: usize = 10;
const NIBBLE_SIZE: usize = 16;
impl<H,C,V> CacheAccum<H,C,V>
where
	H: Hasher,
	C: NodeCodec<H>,
	V: AsRef<[u8]>,
	{
	// TODO switch to static and bench
	fn new() -> Self {
		CacheAccum(vec![(vec![CacheNode::None; NIBBLE_SIZE],0,0); INITIAL_DEPTH],
		std::iter::repeat_with(|| None).take(INITIAL_DEPTH).collect() // vec![None; DEPTH] for non clone
		, PhantomData)
	}
	fn set_node(&mut self, depth:usize, nibble_ix:usize, node: CacheNode<H::Out>) {
    if depth >= self.0.len() {
		  for _i in self.0.len()..depth+1 { 
        self.0.push((vec![CacheNode::None; NIBBLE_SIZE],0,0));
        self.1.push(None);
      }
    }
		self.0[depth].0[nibble_ix] = node;
		// strong heuristic from the fact that we do not delete depth except globally
		// and that we only check relevant size for 0 and 1 TODO replace counter by enum
		// -> so we do not manage replace case
		self.0[depth].1 += 1;
		self.0[depth].2 = nibble_ix; // TODO bench a set if self.0[depth].1 is 0 (probably slower)
	}
	fn depth_added(&self, depth:usize) -> usize {
		self.0[depth].1
	}
	fn depth_last_added(&self, depth:usize) -> usize {
		self.0[depth].2
	}

	fn rem_node(&mut self, depth:usize, nibble:usize) -> CacheNode<H::Out> {
		self.0[depth].1 -= 1;
		self.0[depth].2 = NIBBLE_SIZE; // out of ix -> need to check all value in this case TODO optim it ??
		std::mem::replace(&mut self.0[depth].0[nibble], CacheNode::None)
	}
	fn reset_depth(&mut self, depth:usize) {
		self.0[depth] = (vec![CacheNode::None; NIBBLE_SIZE], 0, 0);
	}

	fn encode_branch(&mut self, depth:usize, has_val: bool, cb_ext: &mut impl ProcessEncodedNode<<H as Hasher>::Out>) -> Vec<u8>	{
    let v = if has_val {
      std::mem::replace(&mut self.1[depth], None)
    } else { None };
		C::branch_node(
			self.0[depth].0.iter().map(|v| 
				match v {
					CacheNode::Hash(h) => Some(clone_child_ref(h)), // TODO try to avoid that clone
					CacheNode::Ext(n, h) => {
						let mut n = n.to_vec();
						n.reverse();// TODO use proper encoded_nibble algo.
						let enc_nibble = encoded_nibble(&n[..], false); // not leaf!!
						let encoded = C::ext_node(&enc_nibble[..], clone_child_ref(h));
						let h = cb_ext.process(encoded, false);
						Some(h)
					},
					CacheNode::None => None,
				}
			), v.as_ref().map(|v|v.as_ref()))
	}

	fn flush_val (
		&mut self, //(64 * 16 size) 
		cb_ext: &mut impl ProcessEncodedNode<<H as Hasher>::Out>,
		target_depth: usize, 
		(k2, v2): &(impl AsRef<[u8]>,impl AsRef<[u8]>), 
	) {
		let nibble_value = nibble_at(&k2.as_ref()[..], target_depth-1);
		// is it a branch value (two candidate same ix)
		let nkey = NibbleSlice::new_offset(&k2.as_ref()[..],target_depth).encoded(true);
		// Note: fwiu, having fixed key size, all values are in leaf (no value in
		// branch). TODO run metrics on a node to count branch with values
		let encoded = C::leaf_node(&nkey.as_ref()[..], &v2.as_ref()[..]);
		let hash = cb_ext.process(encoded, false);

		// insert hash in branch (first level branch only at this point)
		self.set_node(target_depth - 1, nibble_value as usize, CacheNode::Hash(hash));
	}

	fn flush_branch(
		&mut self,
		cb_ext: &mut impl ProcessEncodedNode<<H as Hasher>::Out>,
		ref_branch: impl AsRef<[u8]> + Ord,
		new_depth: usize, 
		old_depth: usize, 
	) {
		for d in (new_depth..old_depth).rev() {
	 
			// check if branch empty TODO switch to optional storage
			let has_val = self.1[d].is_some();
			let depth_size = self.depth_added(d);
			assert!(depth_size != 0);
			if !has_val && depth_size == 1 {
				// extension case
				let unit = self.depth_last_added(d);

				let node = self.rem_node(d, unit);
				// already extension
				if let CacheNode::Ext(mut n,v_hash) = node {
					if d > 0 {
						let nibble: u8 = nibble_at(&ref_branch.as_ref()[..],d-1);
						n.push(unit as u8);
						self.set_node(d-1, nibble as usize, CacheNode::Ext(n, v_hash));
					} else {
						n.push(unit as u8);
						n.reverse(); // TODO use proper encoded_nibble algo.
						let enc_nibble = encoded_nibble(&n[..], false);
						let encoded = C::ext_node(&enc_nibble[..], v_hash);
						cb_ext.process(encoded, true);
					}
				} else {
					let v_hash = node.hash(); // TODO proper match!!
					if d > 0 {
						let nibble: u8 = nibble_at(&ref_branch.as_ref()[..],d-1);
						// TODO capacity vec of 64?
						self.set_node(d-1, nibble as usize, CacheNode::Ext(vec![unit as u8], v_hash));
					} else {
						let enc_nibble = encoded_nibble(&[unit as u8], false);
						let encoded = C::ext_node(&enc_nibble[..], v_hash);
						cb_ext.process(encoded, true);
					}
				}
			} else {
				let encoded = self.encode_branch(d, has_val, cb_ext);
				self.reset_depth(d);
				let hash = cb_ext.process(encoded, d == 0);
				// clear tmp val
				// put hash in parent
				if d > 0 {
					let nibble: u8 = nibble_at(&ref_branch.as_ref()[..],d-1);
					self.set_node(d-1, nibble as usize, CacheNode::Hash(hash));
				} else {
					// reachable !!
				}
			}
		}
	}
}

enum CacheNode<HO> {
	None,
	Hash(ChildReference<HO>),
	Ext(Vec<u8>,ChildReference<HO>),// vec<u8> for nibble slice is not super good looking): TODO bench diff if explicitely boxed
}

impl<HO: Clone> Clone for CacheNode<HO> {
	fn clone(&self) -> Self {
		match self {
			CacheNode::None => CacheNode::None,
			CacheNode::Hash(ChildReference::Hash(h)) => CacheNode::Hash(ChildReference::Hash(h.clone())),
			CacheNode::Hash(ChildReference::Inline(h, s)) => CacheNode::Hash(ChildReference::Inline(h.clone(), *s)),
			CacheNode::Ext(v, ChildReference::Hash(h)) => CacheNode::Ext(v.clone(), ChildReference::Hash(h.clone())),
			CacheNode::Ext(v, ChildReference::Inline(h, s)) => CacheNode::Ext(v.clone(), ChildReference::Inline(h.clone(), *s)),
		}
	}
}

fn clone_child_ref<HO: Clone>(r: &ChildReference<HO>) -> ChildReference<HO> {
	match r {
		ChildReference::Hash(h) => ChildReference::Hash(h.clone()),
		ChildReference::Inline(h, s) => ChildReference::Inline(h.clone(), *s),
	}
}

impl<HO> CacheNode<HO> {
	// unsafe accessors TODO bench diff with safe one
	fn hash(self) -> ChildReference<HO> {
		if let CacheNode::Hash(h) = self {
			return h
		}
		unreachable!()
	}
}


pub fn trie_visit<H, C, I, A, B, F>(input: I, cb_ext: &mut F) 
	where
		I: IntoIterator<Item = (A, B)>,
		A: AsRef<[u8]> + Ord,
		B: AsRef<[u8]>,
		H: Hasher,
		C: NodeCodec<H>,
		F: ProcessEncodedNode<<H as Hasher>::Out>,
	{
	let mut depth_queue = CacheAccum::<H,C,B>::new();
	// compare iter ordering
	let mut iter_input = input.into_iter();
	if let Some(mut prev_val) = iter_input.next() {
		// depth of last item TODO rename to last_depth
		let mut prev_depth = 0;

		for (k, v) in iter_input {
			let common_depth = biggest_depth(&prev_val.0.as_ref()[..], &k.as_ref()[..]);
			// 0 is a reserved value : could use option
			let depth_item = common_depth + 1;
			if common_depth == prev_val.0.as_ref().len() * 2 {
				// the new key include the previous one : branch value case
				depth_queue.1[common_depth] = Some(prev_val.1);
			} else if depth_item >= prev_depth {
				// put prev with next
				depth_queue.flush_val(cb_ext, depth_item, &prev_val);
			} else if depth_item < prev_depth {
				// do not put with next
				depth_queue.flush_val(cb_ext, prev_depth, &prev_val);
				let ref_branches = prev_val.0;
				depth_queue.flush_branch(cb_ext, ref_branches, depth_item, prev_depth);
			}

			prev_val = (k, v);
			prev_depth = depth_item;
		}
		// last pendings
		if prev_depth == 0 {
			// one element
			let (k2, v2) = prev_val; 
			let nkey = NibbleSlice::new_offset(&k2.as_ref()[..],prev_depth).encoded(true);
			let encoded = C::leaf_node(&nkey.as_ref()[..], &v2.as_ref()[..]);
			cb_ext.process(encoded, true);
		} else {
			depth_queue.flush_val(cb_ext, prev_depth, &prev_val);
			let ref_branches = prev_val.0;

			depth_queue.flush_branch(cb_ext, ref_branches, 0, prev_depth);
		}
	} else {
		// nothing null root case
		cb_ext.process(C::empty_node(), true);
	}
}

pub trait ProcessEncodedNode<HO> {
  fn process(&mut self, Vec<u8>, bool) -> ChildReference<HO>;
}

pub struct TrieBuilder<'a, H, HO, V, DB> {
  pub db: &'a mut DB,
  pub root: Option<HO>,
  _ph: PhantomData<(H,V)>,
}

impl<'a, H, HO, V, DB> TrieBuilder<'a, H, HO, V, DB> {
  pub fn new(db: &'a mut DB) -> Self {
    TrieBuilder { db, root: None, _ph: PhantomData } 
  }
}

impl<'a, H: Hasher, V, DB: HashDB<H,V>> ProcessEncodedNode<<H as Hasher>::Out> for TrieBuilder<'a, H, <H as Hasher>::Out, V, DB> {
  fn process(&mut self, enc_ext: Vec<u8>, is_root: bool) -> ChildReference<<H as Hasher>::Out> {
		let len = enc_ext.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&enc_ext[..len]);

			return ChildReference::Inline(h, len);
		}
		let hash = self.db.insert(&enc_ext[..]);
		if is_root {
			self.root = Some(hash.clone());
		};
		ChildReference::Hash(hash)
  }
}

pub struct TrieRoot<H, HO> {
  pub root: Option<HO>,
  _ph: PhantomData<(H)>,
}

impl<H, HO> Default for TrieRoot<H, HO> {
  fn default() -> Self {
    TrieRoot { root: None, _ph: PhantomData } 
  }
}

impl<H: Hasher> ProcessEncodedNode<<H as Hasher>::Out> for TrieRoot<H, <H as Hasher>::Out> {
  fn process(&mut self, enc_ext: Vec<u8>, is_root: bool) -> ChildReference<<H as Hasher>::Out> {
		let len = enc_ext.len();
		if !is_root && len < <H as Hasher>::LENGTH {
			let mut h = <<H as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&enc_ext[..len]);

			return ChildReference::Inline(h, len);
		}
		let hash = <H as Hasher>::hash(&enc_ext[..]);
		if is_root {
			self.root = Some(hash.clone());
		};
		ChildReference::Hash(hash)
  }
}


#[cfg(test)]
mod test {
	use super::*;
	use env_logger;
	use standardmap::*;
	use DBValue;
	use memory_db::MemoryDB;
	use hash_db::{Hasher, HashDB};
	use keccak_hasher::KeccakHasher;
	use reference_trie::{RefTrieDBMut, RefTrieDB, Trie, TrieMut,
	ReferenceNodeCodec, ref_trie_root};

	#[test]
	fn trie_root_empty () {
		compare_impl(vec![])
	}

	#[test]
	fn trie_one_node () {
		compare_impl(vec![
			(vec![1u8,2u8,3u8,4u8],vec![7u8]),
		]);
	}

	#[test]
	fn root_extension () {
		compare_impl(vec![
			(vec![1u8,2u8,3u8,3u8],vec![8u8;32]),
			(vec![1u8,2u8,3u8,4u8],vec![7u8;32]),
		]);
	}

	fn compare_impl(data: Vec<(Vec<u8>,Vec<u8>)>) {
		let memdb = MemoryDB::default();
		let hashdb = MemoryDB::<KeccakHasher, DBValue>::default();
		reference_trie::compare_impl(data, memdb, hashdb);
	}
	fn compare_root(data: Vec<(Vec<u8>,Vec<u8>)>) {
		let memdb = MemoryDB::default();
		reference_trie::compare_root(data, memdb);
	}


	#[test]
	fn trie_middle_node () {
		compare_impl(vec![
			(vec![1u8,2u8],vec![8u8;32]),
			(vec![1u8,2u8,3u8,4u8],vec![7u8;32]),
		]);
	}

	#[test]
	fn trie_middle_node2 () {
		compare_impl(vec![
			(vec![0u8,2u8,3u8,5u8,3u8],vec![1u8;32]),
			(vec![1u8,2u8],vec![8u8;32]),
			(vec![1u8,2u8,3u8,4u8],vec![7u8;32]),
			(vec![1u8,2u8,3u8,5u8],vec![7u8;32]),
			(vec![1u8,2u8,3u8,5u8,3u8],vec![7u8;32]),
		]);
	}
	#[test]
	fn root_extension_bis () {
		compare_root(vec![
			(vec![1u8,2u8,3u8,3u8],vec![8u8;32]),
			(vec![1u8,2u8,3u8,4u8],vec![7u8;32]),
		]);
	}

	#[test]
	fn trie_middle_node2x () {
		compare_impl(vec![
			(vec![0u8,2u8,3u8,5u8,3u8],vec![1u8;2]),
			(vec![1u8,2u8],vec![8u8;2]),
			(vec![1u8,2u8,3u8,4u8],vec![7u8;2]),
			(vec![1u8,2u8,3u8,5u8],vec![7u8;2]),
			(vec![1u8,2u8,3u8,5u8,3u8],vec![7u8;2]),
		]);
 }
}
