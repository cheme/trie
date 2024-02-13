// Copyright 2017, 2020 Parity Technologies
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

use crate::{
	memory_db::{KeyFunction, MemoryDB},
	nibble::{nibble_ops, NibbleSlice},
	node::Value,
	node_codec::NodeCodec,
	node_db::{Hasher, Prefix, EMPTY_PREFIX},
	range_proof::{Bitmap1, ProofOp, RangeProofCodec, RangeProofError},
	rstd::{cmp::max, marker::PhantomData, vec::Vec},
	triedbmut::ChildReference,
	DBValue, NibbleVec, TrieHash, TrieLayout,
};

macro_rules! exponential_out {
	(@3, [$($inpp:expr),*]) => { exponential_out!(@2, [$($inpp,)* $($inpp),*]) };
	(@2, [$($inpp:expr),*]) => { exponential_out!(@1, [$($inpp,)* $($inpp),*]) };
	(@1, [$($inpp:expr),*]) => { [$($inpp,)* $($inpp),*] };
}

type CacheNode<HO> = Option<ChildReference<HO, ()>>;

#[inline(always)]
fn new_vec_slice_buffer<HO>() -> [CacheNode<HO>; 16] {
	exponential_out!(@3, [None, None])
}

type ArrayNode<T> = [CacheNode<TrieHash<T>>; 16];

enum CacheValue<V, H> {
	Value(V),
	Hash(H),
	None,
}

/// Struct containing iteration cache, can be at most the length of the lowest nibble.
///
/// Note that it is not memory optimal (all depth are allocated even if some are empty due
/// to node partial).
/// Three field are used, a cache over the children, an optional associated value and the depth.
struct CacheAccum<T: TrieLayout, V>(Vec<(ArrayNode<T>, CacheValue<V, TrieHash<T>>, usize)>);

/// Initially allocated cache depth.
const INITIAL_DEPTH: usize = 10;

impl<T, V> CacheAccum<T, V>
where
	T: TrieLayout,
	V: AsRef<[u8]>,
{
	fn new() -> Self {
		let v = Vec::with_capacity(INITIAL_DEPTH);
		CacheAccum(v)
	}

	#[inline(always)]
	fn set_cache_value(&mut self, depth: usize, value: CacheValue<V, TrieHash<T>>) {
		if self.0.is_empty() || self.0[self.0.len() - 1].2 < depth {
			self.0.push((new_vec_slice_buffer(), CacheValue::None, depth));
		}
		let last = self.0.len() - 1;
		debug_assert!(self.0[last].2 <= depth);
		self.0[last].1 = value;
	}

	#[inline(always)]
	fn set_node(&mut self, depth: usize, nibble_index: usize, node: CacheNode<TrieHash<T>>) {
		if self.0.is_empty() || self.0[self.0.len() - 1].2 < depth {
			self.0.push((new_vec_slice_buffer(), CacheValue::None, depth));
		}

		let last = self.0.len() - 1;
		debug_assert!(self.0[last].2 == depth);

		self.0[last].0.as_mut()[nibble_index] = node;
	}

	#[inline(always)]
	fn last_depth(&self) -> usize {
		let ix = self.0.len();
		if ix > 0 {
			let last = ix - 1;
			self.0[last].2
		} else {
			0
		}
	}

	#[inline(always)]
	fn last_last_depth(&self) -> usize {
		let ix = self.0.len();
		if ix > 1 {
			let last = ix - 2;
			self.0[last].2
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

	// return true on success
	fn drop_to(
		&mut self,
		key: &mut NibbleVec,
		to: Option<usize>,
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
	) -> bool {
		loop {
			debug_assert!(!T::USE_EXTENSION);
			let is_root = to.is_none() && self.0.len() == 1;
			let mut target_depth = to.unwrap_or(0);
			let hashed;
			let (index, nkey, nsize) = if is_root {
				let nsize = key.len() - (target_depth);
				(0, NibbleSlice::new_offset(&key.inner()[..], 0), nsize)
			} else {
				let parent_parent_depth = self.last_last_depth();
				if target_depth < parent_parent_depth {
					target_depth = parent_parent_depth;
				}
				let nsize = key.len() - (target_depth + 1);
				let index = nibble_ops::left_nibble_at(&key.inner()[..], target_depth) as usize;
				let nkey = NibbleSlice::new_offset(&key.inner()[..], target_depth + 1);
				(index, nkey, nsize)
			};
			let prefix = NibbleSlice::new_offset(
				&key.inner()[..],
				key.inner().len() * nibble_ops::NIBBLE_PER_BYTE - nkey.len(),
			); // TODO is prefix ever different from nkey?

			let Some((children, value, _depth)) = self.0.pop() else {
				return false;
			};
			debug_assert!(_depth == key.len());

			let value = match &value {
				CacheValue::Value(value) => Some(
					if let Some(value) = Value::new_inline(value.as_ref(), T::MAX_INLINE_VALUE) {
						value
					} else {
						hashed = callback
							.process_inner_hashed_value((key.inner(), None), value.as_ref());
						Value::Node(hashed.as_ref(), ())
					},
				),
				CacheValue::Hash(hash) => Some(Value::Node(hash.as_ref(), ())),
				CacheValue::None => None,
			};

			let is_branch = children.iter().any(|c| c.is_some());
			let hash = if is_branch {
				let encoded = T::Codec::branch_node_nibbled(
					nkey.right_range_iter(nsize),
					nsize,
					children.iter(),
					value,
				);
				callback.process(prefix.left(), encoded, is_root)
			} else {
				let encoded = T::Codec::leaf_node(nkey.right_iter(), nkey.len(), value.unwrap());
				callback.process(prefix.left(), encoded, is_root)
			};

			key.drop_lasts(key.len() - target_depth);

			if is_root {
				return true;
			}
			let insert = if let Some(parent_depth) = self.0.last().map(|i| i.2) {
				//debug_assert!(target_depth >= parent_depth);
				parent_depth != target_depth
			} else {
				true
			};
			if insert {
				self.0.push((new_vec_slice_buffer(), CacheValue::None, target_depth));
			}
			let (parent_children, _, _) = self.0.last_mut().unwrap();
			debug_assert!(parent_children[index].is_none());
			parent_children[index] = Some(hash);
			if to.map(|t| t == target_depth).unwrap_or(false) {
				return true;
			}
		}
	}

	fn flush_value(
		&mut self,
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		target_depth: usize,
		(k2, v2): &(impl AsRef<[u8]>, impl AsRef<[u8]>),
	) {
		let nibble_value = nibble_ops::left_nibble_at(&k2.as_ref()[..], target_depth);
		// is it a branch value (two candidate same ix)
		let nkey = NibbleSlice::new_offset(&k2.as_ref()[..], target_depth + 1);
		let pr = NibbleSlice::new_offset(
			&k2.as_ref()[..],
			k2.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE - nkey.len(),
		);

		let hashed;
		let value = if let Some(value) = Value::new_inline(v2.as_ref(), T::MAX_INLINE_VALUE) {
			value
		} else {
			hashed = callback.process_inner_hashed_value((k2.as_ref(), None), v2.as_ref());
			Value::Node(hashed.as_ref(), ())
		};
		let encoded = T::Codec::leaf_node(nkey.right_iter(), nkey.len(), value);
		let hash = callback.process(pr.left(), encoded, false);

		// insert hash in branch (first level branch only at this point)
		self.set_node(target_depth, nibble_value as usize, Some(hash));
	}

	fn flush_branch(
		&mut self,
		callback: &mut impl ProcessEncodedNode<TrieHash<T>>,
		ref_branch: impl AsRef<[u8]> + Ord,
		new_depth: usize,
		is_last: bool,
	) {
		while self.last_depth() > new_depth || is_last && !self.is_empty() {
			let lix = self.last_depth();
			let llix = max(self.last_last_depth(), new_depth);

			let (offset, slice_size, is_root) = if llix == 0 && is_last && self.is_one() {
				// branch root
				(llix, lix - llix, true)
			} else {
				(llix + 1, lix - llix - 1, false)
			};
			let nkey = if slice_size > 0 { Some((offset, slice_size)) } else { None };

			let h = if T::USE_EXTENSION {
				self.standard_extension(&ref_branch.as_ref()[..], callback, lix, is_root, nkey)
			} else {
				// encode branch
				self.no_extension(&ref_branch.as_ref()[..], callback, lix, is_root, nkey)
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
	) -> ChildReference<TrieHash<T>, ()> {
		let last = self.0.len() - 1;
		assert_eq!(self.0[last].2, branch_d);

		let (children, v, depth) = self.0.pop().expect("checked");

		debug_assert!(branch_d == depth);
		let pr = NibbleSlice::new_offset(&key_branch, branch_d);

		let hashed;

		let value = match &v {
			CacheValue::Value(v) =>
				Some(if let Some(value) = Value::new_inline(v.as_ref(), T::MAX_INLINE_VALUE) {
					value
				} else {
					let mut prefix = NibbleSlice::new_offset(&key_branch, 0);
					prefix.advance(branch_d);
					hashed = callback.process_inner_hashed_value(prefix.left(), v.as_ref());
					Value::Node(hashed.as_ref(), ())
				}),
			CacheValue::Hash(hash) => Some(Value::Node(hash.as_ref(), ())),
			CacheValue::None => None,
		};

		// encode branch
		let encoded = T::Codec::branch_node(children.iter(), value);
		let branch_hash = callback.process(pr.left(), encoded, is_root && nkey.is_none());

		if let Some(nkeyix) = nkey {
			let pr = NibbleSlice::new_offset(&key_branch, nkeyix.0);
			let nib = pr.right_range_iter(nkeyix.1);
			let encoded = T::Codec::extension_node(nib, nkeyix.1, branch_hash);
			callback.process(pr.left(), encoded, is_root)
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
	) -> ChildReference<TrieHash<T>, ()> {
		let (children, v, depth) = self.0.pop().expect("checked");

		debug_assert!(branch_d == depth);
		// encode branch
		let nkeyix = nkey.unwrap_or((branch_d, 0));
		let pr = NibbleSlice::new_offset(&key_branch, nkeyix.0);
		let hashed;
		let value = match &v {
			CacheValue::Value(v) =>
				Some(if let Some(value) = Value::new_inline(v.as_ref(), T::MAX_INLINE_VALUE) {
					value
				} else {
					let mut prefix = NibbleSlice::new_offset(&key_branch, 0);
					prefix.advance(branch_d);
					hashed = callback.process_inner_hashed_value(prefix.left(), v.as_ref());
					Value::Node(hashed.as_ref(), ())
				}),
			CacheValue::Hash(hash) => Some(Value::Node(hash.as_ref(), ())),
			CacheValue::None => None,
		};

		let encoded = T::Codec::branch_node_nibbled(
			pr.right_range_iter(nkeyix.1),
			nkeyix.1,
			children.iter(),
			value,
		);
		callback.process(pr.left(), encoded, is_root)
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
	B: AsRef<[u8]>,
	F: ProcessEncodedNode<TrieHash<T>>,
{
	let mut depth_queue = CacheAccum::<T, B>::new();
	// compare iter ordering
	let mut iter_input = input.into_iter();
	if let Some(mut previous_value) = iter_input.next() {
		// depth of last item
		let mut last_depth = 0;

		let mut single = true;
		for (k, v) in iter_input {
			single = false;
			let common_depth =
				nibble_ops::biggest_depth(&previous_value.0.as_ref()[..], &k.as_ref()[..]);
			// 0 is a reserved value : could use option
			let depth_item = common_depth;
			if common_depth == previous_value.0.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE {
				// the new key include the previous one : branch value case
				// just stored value at branch depth
				depth_queue.set_cache_value(common_depth, CacheValue::Value(previous_value.1));
			} else if depth_item >= last_depth {
				// put previous with next (common branch previous value can be flush)
				depth_queue.flush_value(callback, depth_item, &previous_value);
			} else if depth_item < last_depth {
				// do not put with next, previous is last of a branch
				depth_queue.flush_value(callback, last_depth, &previous_value);
				let ref_branches = previous_value.0;
				depth_queue.flush_branch(callback, ref_branches, depth_item, false);
			}

			previous_value = (k, v);
			last_depth = depth_item;
		}
		// last pendings
		if single {
			// one single element corner case
			let (k2, v2) = previous_value;
			let nkey = NibbleSlice::new_offset(&k2.as_ref()[..], last_depth);
			let pr = NibbleSlice::new_offset(
				&k2.as_ref()[..],
				k2.as_ref().len() * nibble_ops::NIBBLE_PER_BYTE - nkey.len(),
			);

			let hashed;
			let value = if let Some(value) = Value::new_inline(v2.as_ref(), T::MAX_INLINE_VALUE) {
				value
			} else {
				hashed = callback.process_inner_hashed_value((k2.as_ref(), None), v2.as_ref());
				Value::Node(hashed.as_ref(), ())
			};

			let encoded = T::Codec::leaf_node(nkey.right_iter(), nkey.len(), value);
			callback.process(pr.left(), encoded, true);
		} else {
			depth_queue.flush_value(callback, last_depth, &previous_value);
			let ref_branches = previous_value.0;
			depth_queue.flush_branch(callback, ref_branches, 0, true);
		}
	} else {
		// nothing null root corner case
		callback.process(EMPTY_PREFIX, T::Codec::empty_node().to_vec(), true);
	}
}

/// Visitor trait to implement when using `trie_visit`.
pub trait ProcessEncodedNode<HO> {
	/// Function call with prefix, encoded value and a boolean indicating if the
	/// node is the root for each node of the trie.
	///
	/// Note that the returned value can change depending on implementation,
	/// but usually it should be the Hash of encoded node.
	/// This is not something direcly related to encoding but is here for
	/// optimisation purpose (builder node_db does return this value).
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<HO, ()>;

	/// Callback for hashed value in encoded node.
	fn process_inner_hashed_value(&mut self, prefix: Prefix, value: &[u8]) -> HO;
}

/// Get trie root and insert visited node in a node_db.
/// As for all `ProcessEncodedNode` implementation, it
/// is only for full trie parsing (not existing trie).
pub struct TrieBuilder<'a, T: TrieLayout, K: KeyFunction<T::Hash> + Send + Sync> {
	db: &'a mut MemoryDB<T::Hash, K, DBValue>,
	pub root: Option<TrieHash<T>>,
}

impl<'a, T: TrieLayout, K: KeyFunction<T::Hash> + Send + Sync> TrieBuilder<'a, T, K> {
	pub fn new(db: &'a mut MemoryDB<T::Hash, K, DBValue>) -> Self {
		TrieBuilder { db, root: None }
	}
}

impl<'a, T, K: KeyFunction<T::Hash> + Send + Sync> ProcessEncodedNode<TrieHash<T>>
	for TrieBuilder<'a, T, K>
where
	T: TrieLayout,
{
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<TrieHash<T>, ()> {
		let len = encoded_node.len();
		if !is_root && len < <T::Hash as Hasher>::LENGTH {
			let mut h = <<T::Hash as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len)
		}
		let hash = self.db.insert(prefix, &encoded_node[..]);
		if is_root {
			self.root = Some(hash);
		};
		ChildReference::Hash(hash, ())
	}

	fn process_inner_hashed_value(&mut self, prefix: Prefix, value: &[u8]) -> TrieHash<T> {
		self.db.insert(prefix, value)
	}
}

/// Calculate the trie root of the trie.
pub struct TrieRoot<T: TrieLayout> {
	/// The resulting root.
	pub root: Option<TrieHash<T>>,
}

impl<T: TrieLayout> Default for TrieRoot<T> {
	fn default() -> Self {
		TrieRoot { root: None }
	}
}

impl<T: TrieLayout> ProcessEncodedNode<TrieHash<T>> for TrieRoot<T> {
	fn process(
		&mut self,
		_: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<TrieHash<T>, ()> {
		let len = encoded_node.len();
		if !is_root && len < <T::Hash as Hasher>::LENGTH {
			let mut h = <<T::Hash as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len)
		}
		let hash = <T::Hash as Hasher>::hash(encoded_node.as_slice());
		if is_root {
			self.root = Some(hash);
		};
		ChildReference::Hash(hash, ())
	}

	fn process_inner_hashed_value(&mut self, _prefix: Prefix, value: &[u8]) -> TrieHash<T> {
		<T::Hash as Hasher>::hash(value)
	}
}

/// Get the trie root node encoding.
pub struct TrieRootUnhashed<T: TrieLayout> {
	/// The resulting encoded root.
	pub root: Option<Vec<u8>>,
	_ph: PhantomData<T>,
}

impl<T: TrieLayout> Default for TrieRootUnhashed<T> {
	fn default() -> Self {
		TrieRootUnhashed { root: None, _ph: PhantomData }
	}
}

#[cfg(feature = "std")]
/// Calculate the trie root of the trie.
/// Print a debug trace.
pub struct TrieRootPrint<T: TrieLayout> {
	/// The resulting root.
	pub root: Option<TrieHash<T>>,
	_ph: PhantomData<T>,
}

#[cfg(feature = "std")]
impl<T: TrieLayout> Default for TrieRootPrint<T> {
	fn default() -> Self {
		TrieRootPrint { root: None, _ph: PhantomData }
	}
}

#[cfg(feature = "std")]
impl<T: TrieLayout> ProcessEncodedNode<TrieHash<T>> for TrieRootPrint<T> {
	fn process(
		&mut self,
		p: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<TrieHash<T>, ()> {
		println!("Encoded node: {:x?}", &encoded_node);
		println!("	with prefix: {:x?}", &p);
		let len = encoded_node.len();
		if !is_root && len < <T::Hash as Hasher>::LENGTH {
			let mut h = <<T::Hash as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			println!("	inline len {}", len);
			return ChildReference::Inline(h, len)
		}
		let hash = <T::Hash as Hasher>::hash(encoded_node.as_slice());
		if is_root {
			self.root = Some(hash);
		};
		println!("	hashed to {:x?}", hash.as_ref());
		ChildReference::Hash(hash, ())
	}

	fn process_inner_hashed_value(&mut self, _prefix: Prefix, value: &[u8]) -> TrieHash<T> {
		println!("Hashed node: {:x?}", &value);
		<T::Hash as Hasher>::hash(value)
	}
}

impl<T: TrieLayout> ProcessEncodedNode<TrieHash<T>> for TrieRootUnhashed<T> {
	fn process(
		&mut self,
		_: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<<T::Hash as Hasher>::Out, ()> {
		let len = encoded_node.len();
		if !is_root && len < <T::Hash as Hasher>::LENGTH {
			let mut h = <<T::Hash as Hasher>::Out as Default>::default();
			h.as_mut()[..len].copy_from_slice(&encoded_node[..len]);

			return ChildReference::Inline(h, len)
		}
		let hash = <T::Hash as Hasher>::hash(encoded_node.as_slice());

		if is_root {
			self.root = Some(encoded_node);
		};
		ChildReference::Hash(hash, ())
	}

	fn process_inner_hashed_value(&mut self, _prefix: Prefix, value: &[u8]) -> TrieHash<T> {
		<T::Hash as Hasher>::hash(value)
	}
}

pub fn visit_range_proof<
	'a,
	'cache,
	L: TrieLayout,
	F: ProcessEncodedNode<TrieHash<L>>,
	C: RangeProofCodec,
>(
	input: &mut impl crate::range_proof::Read,
	callback: &mut F,
	start_key: Option<&[u8]>,
) -> Result<Option<Vec<u8>>, RangeProofError> {
	let mut key = NibbleVec::new();
	let mut depth_queue = crate::iter_build::CacheAccum::<L, DBValue>::new();

	const BUFF_LEN: usize = 32;
	let mut buff = [0u8; BUFF_LEN];
	let mut can_seek = start_key.is_some();
	let mut exiting: Option<Vec<u8>> = None;
	let mut last_drop: Option<u8> = None;
	let mut prev_op: Option<ProofOp> = None;
	let mut last_key: Option<Vec<u8>> = None;
	loop {
		if let Err(e) = input.read_exact(&mut buff[..1]) {
			match &e {
				RangeProofError::EndOfStream =>
					if depth_queue.drop_to(&mut key, None, callback) {
						return Ok(exiting);
					} else {
						return Err(RangeProofError::UnexpectedBehavior);
					}, // aka ccannot read
				_ => (),
			}
			return Err(e);
		};
		let (proof_op, attached) =
			C::decode_op(buff[0]).ok_or(RangeProofError::MalformedSequence)?;
		match proof_op {
			ProofOp::Partial => {
				// TODO check the partial is post prev partial!!
				match prev_op {
					Some(ProofOp::Partial) => {
						// no two consecutive partial
						return Err(RangeProofError::MalformedSequence)
					},
					_ => (),
				}
				if exiting.is_some() {
					return Err(RangeProofError::MalformedSequence);
				}
				last_drop = None;
				let size = C::decode_size(proof_op, attached, input)?;
				if size == 0 {
					return Err(RangeProofError::MalformedProofOp);
				}
				let (mut nb_byte, last_byte) = if size % nibble_ops::NIBBLE_PER_BYTE == 0 {
					(size / nibble_ops::NIBBLE_PER_BYTE, false)
				} else {
					(size / nibble_ops::NIBBLE_PER_BYTE, true)
				};

				while nb_byte > 0 {
					let bound = core::cmp::min(nb_byte, BUFF_LEN);
					input.read_exact(&mut buff[..bound])?;
					key.append_partial(((0, 0), &buff[..bound]));
					nb_byte -= bound;
				}
				if last_byte {
					input.read_exact(&mut buff[..1])?;
					key.push(NibbleSlice::new(&buff[..1]).at(0));
				}
				if can_seek {
					if let Some(start_key) = start_key {
						let common = nibble_ops::biggest_depth(start_key, key.inner());
						let common = core::cmp::min(common, key.len());
						if common < key.len() {
							// TODO here if valid seek should only be equal
							let start_key_len = start_key.len() * nibble_ops::NIBBLE_PER_BYTE;
							if common == start_key_len {
								can_seek = false;
							} else {
								// not into seek
								return Err(RangeProofError::MalformedSequence);
							}
						}
					}
				}
			},
			ProofOp::Value => {
				match prev_op {
					Some(ProofOp::Value) => {
						// no two value at same heigth
						return Err(RangeProofError::MalformedSequence);
					},
					Some(ProofOp::Hashes) => {
						// value is before hashes
						return Err(RangeProofError::MalformedSequence);
					},
					Some(ProofOp::DropPartial) => {
						// value already sent after a drop
						return Err(RangeProofError::MalformedSequence);
					},
					_ => (),
				}
				if exiting.is_some() {
					return Err(RangeProofError::MalformedSequence);
				}
				last_drop = None;
				can_seek = false;

				let value = read_value::<BUFF_LEN, C>(input, attached, &mut buff)?;
				// not the most efficient as this is guaranted to be a push
				depth_queue.set_cache_value(key.len(), CacheValue::Value(value));
				last_key = Some(key.inner().to_vec());
			},
			ProofOp::DropPartial => {
				match prev_op {
					Some(ProofOp::DropPartial) => {
						// consecutive drop
						return Err(RangeProofError::MalformedSequence);
					},
					Some(ProofOp::Partial) => {
						// drop after a push, we should at least have hash or value
						return Err(RangeProofError::MalformedSequence);
					},
					None => {
						// drop from start doesn't work
						return Err(RangeProofError::MalformedSequence);
					},
					_ => (),
				}
				can_seek = false;

				let to_drop = C::decode_size(ProofOp::DropPartial, attached, input)?;
				if to_drop == 0 {
					return Err(RangeProofError::MalformedProofOp);
				}
				if to_drop > key.len() {
					return Err(RangeProofError::MalformedSequence);
				}
				let to = key.len() - to_drop;
				last_drop = Some(key.at(to));
				depth_queue.drop_to(&mut key, Some(to), callback);
			},
			ProofOp::Hashes => {
				// hash are either nodes from seeking: value and children
				// up to seek key.
				// Or nodes after suspending, starting after last accessed child.
				match prev_op {
					Some(ProofOp::Value) | Some(ProofOp::DropPartial) => {
						if last_key.is_none() {
							return Err(RangeProofError::MalformedSequence);
						}
						exiting = last_key.clone();
					},
					Some(ProofOp::Hashes) => {
						// consecutive hashes
						return Err(RangeProofError::MalformedSequence);
					},
					Some(ProofOp::Partial) =>
						if !can_seek {
							return Err(RangeProofError::MalformedSequence);
						},
					None =>
						if !can_seek {
							return Err(RangeProofError::MalformedSequence);
						},
				}
				let mut i = 8; // trigger a header read.
				let mut bitmap = Bitmap1(0);
				if let Some(att) = attached {
					let nb_bitmap_hash =
						C::op_attached_range(ProofOp::Hashes).expect("has attached");
					i = 8 - (nb_bitmap_hash as usize);
					bitmap = Bitmap1(att << i);
				}
				let mut expect_value = false;
				let mut range_bef = 0;
				let mut range_aft = nibble_ops::NIBBLE_LENGTH as u8;
				if exiting.is_none() && can_seek {
					expect_value = true;
					if let Some(k) = start_key {
						let start_nibble = NibbleSlice::new(k);

						if start_nibble.len() > key.len() {
							range_bef = start_nibble.at(key.len());
						} else if start_nibble.len() == key.len() {
							// only value hash
						} else {
							// even eq should contain no hash
							return Err(RangeProofError::UnexpectedBehavior);
						}
					}
				} else if exiting.is_some() {
					range_aft = 0;
					if let Some(at) = last_drop.take() {
						range_aft = at + 1;
					}
				} else {
					// should be unreachable, but error just in case.
					return Err(RangeProofError::UnexpectedBehavior);
				}

				if expect_value {
					let has_value = read_bitmap(&mut i, &mut bitmap, input)?;

					if has_value {
						if read_bitmap(&mut i, &mut bitmap, input)? {
							// inline value
							let value = read_value::<BUFF_LEN, C>(input, None, &mut buff)?;
							depth_queue.set_cache_value(key.len(), CacheValue::Value(value));
						} else {
							// hash value
							let hash = read_hash::<L>(input)?;
							depth_queue.set_cache_value(key.len(), CacheValue::Hash(hash));
						}
					}
				}

				let mut child_ix = 0;
				let mut know_has_first_child = false;
				loop {
					let mut first_read = true;
					while child_ix < nibble_ops::NIBBLE_LENGTH {
						if child_ix == range_bef as usize {
							child_ix = range_aft as usize;
							if child_ix == nibble_ops::NIBBLE_LENGTH {
								break;
							}
						}
						let has_child = if know_has_first_child {
							know_has_first_child = false;
							true
						} else if first_read {
							first_read = false;
							read_bitmap(&mut i, &mut bitmap, input)?
						} else {
							if let Some(has_child) = read_bitmap_no_fetch(&mut i, &bitmap) {
								has_child
							} else {
								break;
							}
						};
						if has_child {
							if let Some(is_inline) = read_bitmap_no_fetch(&mut i, &bitmap) {
								if is_inline {
									// child inline
									let (value, len) = read_value_hash::<L, C>(input)?;
									depth_queue.set_node(
										key.len(),
										child_ix as usize,
										Some(ChildReference::Inline(value, len)),
									);
								} else {
									// child hash
									let hash = read_hash::<L>(input)?;
									depth_queue.set_node(
										key.len(),
										child_ix as usize,
										Some(ChildReference::Hash(hash, ())),
									);
								}
								child_ix += 1;
							} else {
								know_has_first_child = true;
								break;
							}
						} else {
							child_ix += 1;
						}
					}

					if child_ix == nibble_ops::NIBBLE_LENGTH {
						break; // TODO always break and use label on inner break
					}
				}
			},
		}
		prev_op = Some(proof_op);
	}
}

fn read_bitmap(
	i: &mut usize,
	bitmap: &mut Bitmap1,
	input: &mut impl crate::range_proof::Read,
) -> Result<bool, RangeProofError> {
	let mut buff = [0u8; 1];
	if *i == 8 {
		*i = 0;
		input.read_exact(&mut buff[0..1])?;
		*bitmap = Bitmap1(buff[0]);
	}
	let r = Ok(bitmap.get(*i));
	*i += 1;
	r
}

fn read_bitmap_no_fetch(i: &mut usize, bitmap: &Bitmap1) -> Option<bool> {
	if *i == 8 {
		return None;
	}
	let r = Some(bitmap.get(*i));
	*i += 1;
	r
}

fn read_value<const BUFF_LEN: usize, C: RangeProofCodec>(
	input: &mut impl crate::range_proof::Read,
	attached: Option<u8>,
	buff: &mut [u8; BUFF_LEN],
) -> Result<DBValue, RangeProofError> {
	let mut nb_byte = C::decode_size(ProofOp::Value, attached, input)?;
	let mut value = DBValue::with_capacity(nb_byte);
	while nb_byte > 0 {
		let bound = core::cmp::min(nb_byte, BUFF_LEN);
		input.read_exact(&mut buff[..bound])?; // TODO we got our own bufs: use read on value
		value.extend_from_slice(&buff[..bound]);
		nb_byte -= bound;
	}

	Ok(value)
}

fn read_hash<L: TrieLayout>(
	input: &mut impl crate::range_proof::Read,
) -> Result<TrieHash<L>, RangeProofError> {
	let mut hash = TrieHash::<L>::default();
	input.read_exact(&mut hash.as_mut())?;

	Ok(hash)
}

fn read_value_hash<L: TrieLayout, C: RangeProofCodec>(
	input: &mut impl crate::range_proof::Read,
) -> Result<(TrieHash<L>, usize), RangeProofError> {
	let nb_byte = C::varint_decode_from(input)? as usize;

	let mut hash = TrieHash::<L>::default();
	if nb_byte > hash.as_ref().len() {
		return Err(RangeProofError::MalformedInlineValue);
	}

	input.read_exact(&mut hash.as_mut()[..nb_byte])?;

	Ok((hash, nb_byte))
}
