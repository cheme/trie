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
	nibble::{NibbleOps, NibbleSlice},
	node::Value,
	node_codec::NodeCodec,
	rstd::{cmp::max, marker::PhantomData, vec::Vec},
	triedbmut::ChildReference,
	DBValue, TrieHash, TrieLayout,
};
use hash_db::{HashDB, Hasher, Prefix};

type CacheNode<HO> = Option<ChildReference<HO>>;

type ArrayNode<T, const N: usize> = [CacheNode<TrieHash<T, N>>; N];

/// Struct containing iteration cache, can be at most the length of the lowest nibble.
///
/// Note that it is not memory optimal (all depth are allocated even if some are empty due
/// to node partial).
/// Three field are used, a cache over the children, an optional associated value and the depth.
struct CacheAccum<T: TrieLayout<N>, V, const N: usize>(Vec<(ArrayNode<T, N>, Option<V>, usize)>);

/// Initially allocated cache depth.
const INITIAL_DEPTH: usize = 10;

impl<T, V, const N: usize> CacheAccum<T, V, N>
where
	T: TrieLayout<N>,
	V: AsRef<[u8]>,
{
	fn new() -> Self {
		let v = Vec::with_capacity(INITIAL_DEPTH);
		CacheAccum(v)
	}

	#[inline(always)]
	fn set_cache_value(&mut self, depth: usize, value: Option<V>) {
		if self.0.is_empty() || self.0[self.0.len() - 1].2 < depth {
			self.0.push(([None; N], None, depth));
		}
		let last = self.0.len() - 1;
		debug_assert!(self.0[last].2 <= depth);
		self.0[last].1 = value;
	}

	#[inline(always)]
	fn set_node(&mut self, depth: usize, nibble_index: usize, node: CacheNode<TrieHash<T, N>>) {
		if self.0.is_empty() || self.0[self.0.len() - 1].2 < depth {
			self.0.push(([None; N], None, depth));
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

	fn flush_value(
		&mut self,
		callback: &mut impl ProcessEncodedNode<TrieHash<T, N>>,
		target_depth: usize,
		(k2, v2): &(impl AsRef<[u8]>, impl AsRef<[u8]>),
	) {
		let nibble_value = NibbleOps::<N>::left_nibble_at(&k2.as_ref()[..], target_depth);
		// is it a branch value (two candidate same ix)
		let nkey = NibbleSlice::<N>::new_offset(&k2.as_ref()[..], target_depth + 1);
		let pr = NibbleSlice::<N>::new_offset(&k2.as_ref()[..], k2.as_ref().len() * N - nkey.len());

		let hashed;
		let value = if let Some(value) = Value::new_inline(v2.as_ref(), T::MAX_INLINE_VALUE) {
			value
		} else {
			hashed = callback.process_inner_hashed_value((k2.as_ref(), (0, 0)), v2.as_ref());
			Value::Node(hashed.as_ref())
		};
		let encoded = T::Codec::leaf_node(nkey.right_iter(), nkey.len(), value);
		let hash = callback.process(pr.left(), encoded, false);

		// insert hash in branch (first level branch only at this point)
		self.set_node(target_depth, nibble_value as usize, Some(hash));
	}

	fn flush_branch(
		&mut self,
		callback: &mut impl ProcessEncodedNode<TrieHash<T, N>>,
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
				let nibble: u8 = NibbleOps::<N>::left_nibble_at(&ref_branch.as_ref()[..], llix);
				self.set_node(llix, nibble as usize, Some(h));
			}
		}
	}

	#[inline(always)]
	fn standard_extension(
		&mut self,
		key_branch: &[u8],
		callback: &mut impl ProcessEncodedNode<TrieHash<T, N>>,
		branch_d: usize,
		is_root: bool,
		nkey: Option<(usize, usize)>,
	) -> ChildReference<TrieHash<T, N>> {
		let last = self.0.len() - 1;
		assert_eq!(self.0[last].2, branch_d);

		let (children, v, depth) = self.0.pop().expect("checked");

		debug_assert!(branch_d == depth);
		let pr = NibbleSlice::<N>::new_offset(&key_branch, branch_d);

		let hashed;
		let value = if let Some(v) = v.as_ref() {
			Some(if let Some(value) = Value::new_inline(v.as_ref(), T::MAX_INLINE_VALUE) {
				value
			} else {
				let mut prefix = NibbleSlice::<N>::new_offset(&key_branch, 0);
				prefix.advance(branch_d);
				hashed = callback.process_inner_hashed_value(prefix.left(), v.as_ref());
				Value::Node(hashed.as_ref())
			})
		} else {
			None
		};

		// encode branch
		let encoded = T::Codec::branch_node(children.as_ref().iter(), value);
		let branch_hash = callback.process(pr.left(), encoded, is_root && nkey.is_none());

		if let Some(nkeyix) = nkey {
			let pr = NibbleSlice::<N>::new_offset(&key_branch, nkeyix.0);
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
		callback: &mut impl ProcessEncodedNode<TrieHash<T, N>>,
		branch_d: usize,
		is_root: bool,
		nkey: Option<(usize, usize)>,
	) -> ChildReference<TrieHash<T, N>> {
		let (children, v, depth) = self.0.pop().expect("checked");

		debug_assert!(branch_d == depth);
		// encode branch
		let nkeyix = nkey.unwrap_or((branch_d, 0));
		let pr = NibbleSlice::<N>::new_offset(&key_branch, nkeyix.0);
		let hashed;
		let value = if let Some(v) = v.as_ref() {
			Some(if let Some(value) = Value::new_inline(v.as_ref(), T::MAX_INLINE_VALUE) {
				value
			} else {
				let mut prefix = NibbleSlice::<N>::new_offset(&key_branch, 0);
				prefix.advance(branch_d);
				hashed = callback.process_inner_hashed_value(prefix.left(), v.as_ref());
				Value::Node(hashed.as_ref())
			})
		} else {
			None
		};

		let encoded = T::Codec::branch_node_nibbled(
			pr.right_range_iter(nkeyix.1),
			nkeyix.1,
			children.as_ref().iter(),
			value,
		);
		callback.process(pr.left(), encoded, is_root)
	}
}

/// Function visiting trie from key value inputs with a `ProccessEncodedNode` callback.
/// This is the main entry point of this module.
/// Calls to each node occurs ordered by byte key value but with longest keys first (from node to
/// branch to root), this differs from standard byte array ordering a bit.
pub fn trie_visit<T, I, A, B, F, const N: usize>(input: I, callback: &mut F)
where
	T: TrieLayout<N>,
	I: IntoIterator<Item = (A, B)>,
	A: AsRef<[u8]> + Ord,
	B: AsRef<[u8]>,
	F: ProcessEncodedNode<TrieHash<T, N>>,
{
	let mut depth_queue = CacheAccum::<T, B, N>::new();
	// compare iter ordering
	let mut iter_input = input.into_iter();
	if let Some(mut previous_value) = iter_input.next() {
		// depth of last item
		let mut last_depth = 0;

		let mut single = true;
		for (k, v) in iter_input {
			single = false;
			let common_depth =
				NibbleOps::<N>::biggest_depth(&previous_value.0.as_ref()[..], &k.as_ref()[..]);
			// 0 is a reserved value : could use option
			let depth_item = common_depth;
			if common_depth == previous_value.0.as_ref().len() * N {
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
				depth_queue.flush_branch(callback, ref_branches, depth_item, false);
			}

			previous_value = (k, v);
			last_depth = depth_item;
		}
		// last pendings
		if single {
			// one single element corner case
			let (k2, v2) = previous_value;
			let nkey = NibbleSlice::<N>::new_offset(&k2.as_ref()[..], last_depth);
			let pr =
				NibbleSlice::<N>::new_offset(&k2.as_ref()[..], k2.as_ref().len() * N - nkey.len());

			let hashed;
			let value = if let Some(value) = Value::new_inline(v2.as_ref(), T::MAX_INLINE_VALUE) {
				value
			} else {
				hashed = callback.process_inner_hashed_value((k2.as_ref(), (0, 0)), v2.as_ref());
				Value::Node(hashed.as_ref())
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
		callback.process(hash_db::EMPTY_PREFIX, T::Codec::empty_node().to_vec(), true);
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
	/// optimisation purpose (builder hash_db does return this value).
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<HO>;

	/// Callback for hashed value in encoded node.
	fn process_inner_hashed_value(&mut self, prefix: Prefix, value: &[u8]) -> HO;
}

/// Get trie root and insert visited node in a hash_db.
/// As for all `ProcessEncodedNode` implementation, it
/// is only for full trie parsing (not existing trie).
pub struct TrieBuilder<'a, T: TrieLayout<N>, DB, const N: usize> {
	db: &'a mut DB,
	pub root: Option<TrieHash<T, N>>,
}

impl<'a, T: TrieLayout<N>, DB, const N: usize> TrieBuilder<'a, T, DB, N> {
	pub fn new(db: &'a mut DB) -> Self {
		TrieBuilder { db, root: None }
	}
}

impl<'a, T, DB, const N: usize> ProcessEncodedNode<TrieHash<T, N>> for TrieBuilder<'a, T, DB, N>
where
	T: TrieLayout<N>,
	DB: HashDB<T::Hash, DBValue>,
{
	fn process(
		&mut self,
		prefix: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<TrieHash<T, N>> {
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
		ChildReference::Hash(hash)
	}

	fn process_inner_hashed_value(&mut self, prefix: Prefix, value: &[u8]) -> TrieHash<T, N> {
		self.db.insert(prefix, value)
	}
}

/// Calculate the trie root of the trie.
pub struct TrieRoot<T: TrieLayout<N>, const N: usize> {
	/// The resulting root.
	pub root: Option<TrieHash<T, N>>,
}

impl<T: TrieLayout<N>, const N: usize> Default for TrieRoot<T, N> {
	fn default() -> Self {
		TrieRoot { root: None }
	}
}

impl<T: TrieLayout<N>, const N: usize> ProcessEncodedNode<TrieHash<T, N>> for TrieRoot<T, N> {
	fn process(
		&mut self,
		_: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<TrieHash<T, N>> {
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
		ChildReference::Hash(hash)
	}

	fn process_inner_hashed_value(&mut self, _prefix: Prefix, value: &[u8]) -> TrieHash<T, N> {
		<T::Hash as Hasher>::hash(value)
	}
}

/// Get the trie root node encoding.
pub struct TrieRootUnhashed<T: TrieLayout<N>, const N: usize> {
	/// The resulting encoded root.
	pub root: Option<Vec<u8>>,
	_ph: PhantomData<T>,
}

impl<T: TrieLayout<N>, const N: usize> Default for TrieRootUnhashed<T, N> {
	fn default() -> Self {
		TrieRootUnhashed { root: None, _ph: PhantomData }
	}
}

#[cfg(feature = "std")]
/// Calculate the trie root of the trie.
/// Print a debug trace.
pub struct TrieRootPrint<T: TrieLayout<N>, const N: usize> {
	/// The resulting root.
	pub root: Option<TrieHash<T, N>>,
	_ph: PhantomData<T>,
}

#[cfg(feature = "std")]
impl<T: TrieLayout<N>, const N: usize> Default for TrieRootPrint<T, N> {
	fn default() -> Self {
		TrieRootPrint { root: None, _ph: PhantomData }
	}
}

#[cfg(feature = "std")]
impl<T: TrieLayout<N>, const N: usize> ProcessEncodedNode<TrieHash<T, N>> for TrieRootPrint<T, N> {
	fn process(
		&mut self,
		p: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<TrieHash<T, N>> {
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
		ChildReference::Hash(hash)
	}

	fn process_inner_hashed_value(&mut self, _prefix: Prefix, value: &[u8]) -> TrieHash<T, N> {
		println!("Hashed node: {:x?}", &value);
		<T::Hash as Hasher>::hash(value)
	}
}

impl<T: TrieLayout<N>, const N: usize> ProcessEncodedNode<TrieHash<T, N>>
	for TrieRootUnhashed<T, N>
{
	fn process(
		&mut self,
		_: Prefix,
		encoded_node: Vec<u8>,
		is_root: bool,
	) -> ChildReference<<T::Hash as Hasher>::Out> {
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
		ChildReference::Hash(hash)
	}

	fn process_inner_hashed_value(&mut self, _prefix: Prefix, value: &[u8]) -> TrieHash<T, N> {
		<T::Hash as Hasher>::hash(value)
	}
}

#[cfg(test)]
mod test {
	use crate::DBValue;
	use keccak_hasher::KeccakHasher;
	use memory_db::{HashKey, MemoryDB, PrefixedKey};

	#[test]
	fn trie_root_empty() {
		compare_implementations(vec![])
	}

	#[test]
	fn trie_one_node() {
		compare_implementations(vec![(vec![1u8, 2u8, 3u8, 4u8], vec![7u8])]);
	}

	#[test]
	fn root_extension_one() {
		compare_implementations(vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8; 32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 32]),
		]);
	}

	fn test_iter(data: Vec<(Vec<u8>, Vec<u8>)>) {
		use reference_trie::{RefTrieDB, RefTrieDBMut, Trie, TrieMut};

		let mut db = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut db, &mut root);
			for i in 0..data.len() {
				let key: &[u8] = &data[i].0;
				let value: &[u8] = &data[i].1;
				t.insert(key, value).unwrap();
			}
		}
		let t = RefTrieDB::new(&db, &root).unwrap();
		for (i, kv) in t.iter().unwrap().enumerate() {
			let (k, v) = kv.unwrap();
			let key: &[u8] = &data[i].0;
			let value: &[u8] = &data[i].1;
			assert_eq!(k, key);
			assert_eq!(v, value);
		}
		for (k, v) in data.into_iter() {
			assert_eq!(&t.get(&k[..]).unwrap().unwrap()[..], &v[..]);
		}
	}

	fn test_iter_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
		use reference_trie::{RefTrieDBMutNoExt, RefTrieDBNoExt, Trie, TrieMut};

		let mut db = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut db, &mut root);
			for i in 0..data.len() {
				let key: &[u8] = &data[i].0;
				let value: &[u8] = &data[i].1;
				t.insert(key, value).unwrap();
			}
		}
		let t = RefTrieDBNoExt::new(&db, &root).unwrap();
		for (i, kv) in t.iter().unwrap().enumerate() {
			let (k, v) = kv.unwrap();
			let key: &[u8] = &data[i].0;
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
		compare_implementations_no_extension_q(data.clone());
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
	fn compare_implementations_no_extension_q(data: Vec<(Vec<u8>, Vec<u8>)>) {
		let memdb = MemoryDB::<_, HashKey<_>, _>::default();
		let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
		reference_trie::compare_implementations_no_extension_q(data, memdb, hashdb);
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
	fn trie_middle_node1() {
		compare_implementations(vec![
			(vec![1u8, 2u8], vec![8u8; 32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 32]),
		]);
	}
	#[test]
	fn trie_middle_node2() {
		compare_implementations(vec![
			(vec![0u8, 2u8, 3u8, 5u8, 3u8], vec![1u8; 32]),
			(vec![1u8, 2u8], vec![8u8; 32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 32]),
			(vec![1u8, 2u8, 3u8, 5u8], vec![7u8; 32]),
			(vec![1u8, 2u8, 3u8, 5u8, 3u8], vec![7u8; 32]),
		]);
	}
	#[test]
	fn root_extension_bis() {
		compare_root(vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8; 32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 32]),
		]);
	}
	#[test]
	fn root_extension_tierce() {
		let d = vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8; 2]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 2]),
		];
		compare_unhashed(d.clone());
		compare_unhashed_no_extension(d);
	}
	#[test]
	fn root_extension_tierce_big() {
		// on more content unhashed would hash
		compare_unhashed(vec![
			(vec![1u8, 2u8, 3u8, 3u8], vec![8u8; 32]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 32]),
			(vec![1u8, 6u8, 3u8, 3u8], vec![8u8; 32]),
			(vec![6u8, 2u8, 3u8, 3u8], vec![8u8; 32]),
			(vec![6u8, 2u8, 3u8, 13u8], vec![8u8; 32]),
		]);
	}
	#[test]
	fn trie_middle_node2x() {
		compare_implementations(vec![
			(vec![0u8, 2u8, 3u8, 5u8, 3u8], vec![1u8; 2]),
			(vec![1u8, 2u8], vec![8u8; 2]),
			(vec![1u8, 2u8, 3u8, 4u8], vec![7u8; 2]),
			(vec![1u8, 2u8, 3u8, 5u8], vec![7u8; 2]),
			(vec![1u8, 2u8, 3u8, 5u8, 3u8], vec![7u8; 2]),
		]);
	}
	#[test]
	fn fuzz1() {
		compare_implementations(vec![
			(vec![01u8], vec![42u8, 9]),
			(vec![01u8, 0u8], vec![0u8, 0]),
			(vec![255u8, 2u8], vec![1u8, 0]),
		]);
	}
	#[test]
	fn fuzz2() {
		compare_implementations(vec![
			(vec![0, 01u8], vec![42u8, 9]),
			(vec![0, 01u8, 0u8], vec![0u8, 0]),
			(vec![0, 255u8, 2u8], vec![1u8, 0]),
		]);
	}
	#[test]
	fn fuzz3() {
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
	fn fuzz_no_extension1() {
		compare_implementations(vec![(vec![0], vec![128, 0]), (vec![128], vec![0, 0])]);
	}
	#[test]
	fn fuzz_no_extension2() {
		compare_implementations(vec![
			(vec![0], vec![6, 255]),
			(vec![6], vec![255, 186]),
			(vec![255], vec![186, 255]),
		]);
	}
	#[test]
	fn fuzz_no_extension5() {
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
	fn fuzz_no_extension3() {
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
	fn fuzz_no_extension4() {
		compare_implementations_no_extension(vec![
			(vec![0x01, 0x56], vec![0x1]),
			(vec![0x02, 0x42], vec![0x2]),
			(vec![0x02, 0x50], vec![0x3]),
		]);
	}
	#[test]
	fn fuzz_no_extension_insert_remove_1() {
		let data = vec![
			(false, vec![0], vec![251, 255]),
			(false, vec![0, 1], vec![251, 255]),
			(false, vec![0, 1, 2], vec![255; 32]),
			(true, vec![0, 1], vec![0, 251]),
		];
		compare_no_extension_insert_remove(data);
	}
	#[test]
	fn fuzz_no_extension_insert_remove_2() {
		let data = vec![
			(false, vec![0x00], vec![0xfd, 0xff]),
			(false, vec![0x10, 0x00], vec![1; 32]),
			(false, vec![0x11, 0x10], vec![0; 32]),
			(true, vec![0x10, 0x00], vec![]),
		];
		compare_no_extension_insert_remove(data);
	}
	#[test]
	fn two_bytes_nibble_length() {
		let data = vec![(vec![00u8], vec![0]), (vec![01u8; 64], vec![0; 32])];
		compare_implementations_no_extension(data.clone());
		compare_implementations_no_extension_prefixed(data.clone());
	}
	#[test]
	#[should_panic]
	fn too_big_nibble_length_old() {
		compare_implementations_h(vec![(vec![01u8; 64], vec![0; 32])]);
	}
	#[test]
	fn too_big_nibble_length_new() {
		compare_implementations_no_extension(vec![(
			vec![01u8; ((u16::max_value() as usize + 1) / 2) + 1],
			vec![0; 32],
		)]);
	}
	#[test]
	fn polka_re_test() {
		compare_implementations(vec![
			(vec![77, 111, 111, 55, 111, 104, 121, 97], vec![68, 97, 105, 55, 105, 101, 116, 111]),
			(vec![101, 105, 67, 104, 111, 111, 66, 56], vec![97, 56, 97, 113, 117, 53, 97]),
			(vec![105, 97, 48, 77, 101, 105, 121, 101], vec![69, 109, 111, 111, 82, 49, 97, 105]),
		]);
	}
}
