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

use hash_db::{HashDBRef, Prefix, EMPTY_PREFIX};
use nibble::{NibbleSlice, NibbleOps, ChildSliceIndex};
use super::node::{Node, OwnedNode};
use node_codec::NodeCodec;
use super::lookup::Lookup;
use super::{Result, DBValue, Trie, TrieItem, TrieError, TrieIterator, Query,
	TrieLayout, CError, TrieHash};
use super::nibble::NibbleVec;
#[cfg(feature = "std")]
use ::std::fmt;
#[cfg(feature = "std")]
use ::std::borrow::Cow;
#[cfg(not(feature = "std"))]
use ::alloc::borrow::Cow;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A `Trie` implementation using a generic `HashDB` backing database, a `Hasher`
/// implementation to generate keys and a `NodeCodec` implementation to encode/decode
/// the nodes.
///
/// Use it as a `Trie` trait object. You can use `db()` to get the backing database object.
/// Use `get` and `contains` to query values associated with keys in the trie.
///
/// # Example
/// ```
/// extern crate trie_db;
/// extern crate reference_trie;
/// extern crate hash_db;
/// extern crate keccak_hasher;
/// extern crate memory_db;
///
/// use hash_db::Hasher;
/// use reference_trie::{RefTrieDBMut, RefTrieDB, Trie, TrieMut};
/// use trie_db::DBValue;
/// use keccak_hasher::KeccakHasher;
/// use memory_db::*;
///
/// fn main() {
///   let mut memdb = MemoryDB::<KeccakHasher, HashKey<_>, _>::default();
///   let mut root = Default::default();
///   RefTrieDBMut::new(&mut memdb, &mut root).insert(b"foo", b"bar").unwrap();
///   let t = RefTrieDB::new(&memdb, &root).unwrap();
///   assert!(t.contains(b"foo").unwrap());
///   assert_eq!(t.get(b"foo").unwrap().unwrap(), DBValue::from_slice(b"bar"));
/// }
/// ```
pub struct TrieDB<'db, L>
where
	L: TrieLayout,
{
	db: &'db dyn HashDBRef<L::Hash, DBValue>,
	root: &'db TrieHash<L>,
	/// The number of hashes performed so far in operations on this trie.
	hash_count: usize,
}

impl<'db, L> TrieDB<'db, L>
where
	L: TrieLayout,
{
	/// Create a new trie with the backing database `db` and `root`
	/// Returns an error if `root` does not exist
	pub fn new(
		db: &'db dyn HashDBRef<L::Hash, DBValue>,
		root: &'db TrieHash<L>
	) -> Result<Self, TrieHash<L>, CError<L>> {
		if !db.contains(root, EMPTY_PREFIX) {
			Err(Box::new(TrieError::InvalidStateRoot(*root)))
		} else {
			Ok(TrieDB {db, root, hash_count: 0})
		}
	}

	/// Get the backing database.
	pub fn db(&'db self) -> &'db dyn HashDBRef<L::Hash, DBValue> { self.db }

	/// Get the data of the root node.
	pub fn root_data(&self) -> Result<DBValue, TrieHash<L>, CError<L>> {
		self.db
			.get(self.root, EMPTY_PREFIX)
			.ok_or_else(|| Box::new(TrieError::InvalidStateRoot(*self.root)))
	}

	/// Given some node-describing data `node`, and node key return the actual node RLP.
	/// This could be a simple identity operation in the case that the node is sufficiently small,
	/// but may require a database lookup. If `is_root_data` then this is root-data and
	/// is known to be literal.
	/// `partial_key` is encoded nibble slice that addresses the node.
	fn get_raw_or_lookup(
		&'db self, node: &[u8],
		partial_key: Prefix,
	) -> Result<Cow<'db, DBValue>, TrieHash<L>, CError<L>> {
		match (partial_key.0.is_empty() && (partial_key.1).0 == 0, L::Codec::try_decode_hash(node)) {
			(false, Some(key)) => {
				self.db
					.get(&key, partial_key)
					.map(|v| Cow::Owned(v))
					.ok_or_else(|| Box::new(TrieError::IncompleteDatabase(key)))
			}
			_ => Ok(Cow::Owned(DBValue::from_slice(node)))
		}
	}
}

impl<'db, L> Trie<L> for TrieDB<'db, L>
where
	L: TrieLayout,
{
	fn root(&self) -> &TrieHash<L> { self.root }

	fn get_with<'a, 'key, Q: Query<L::Hash>>(
		&'a self,
		key: &'key [u8],
		query: Q,
	) -> Result<Option<Q::Item>, TrieHash<L>, CError<L>>
		where 'a: 'key,
	{
		Lookup::<L, Q> {
			db: self.db,
			query: query,
			hash: self.root.clone(),
		}.look_up(NibbleSlice::new(key))
	}

	fn iter<'a>(&'a self)-> Result<
		Box<dyn TrieIterator<L, Item=TrieItem<TrieHash<L>, CError<L>>> + 'a>,
		TrieHash<L>,
		CError<L>,
	> {
		TrieDBIterator::new(self).map(|iter| Box::new(iter) as Box<_>)
	}
}


#[cfg(feature="std")]
// This is for pretty debug output only
struct TrieAwareDebugNode<'db, 'a, L>
where
	L: TrieLayout,
{
	trie: &'db TrieDB<'db, L>,
	node_key: &'a[u8],
	partial_key: NibbleVec<L::Nibble>,
	index: Option<u8>,
}

#[cfg(feature="std")]
impl<'db, 'a, L> fmt::Debug for TrieAwareDebugNode<'db, 'a, L>
where
	L: TrieLayout,
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		if let Ok(node) = self.trie.get_raw_or_lookup(self.node_key, self.partial_key.as_prefix()) {
			match L::Codec::decode(&node) {
				Ok(Node::Leaf(slice, value)) =>
					match (f.debug_struct("Node::Leaf"), self.index) {
						(ref mut d, Some(i)) => d.field("index", &i),
						(ref mut d, _) => d,
					}
						.field("slice", &slice)
						.field("value", &value)
						.finish(),
				Ok(Node::Extension(slice, item)) =>
					match (f.debug_struct("Node::Extension"), self.index) {
						(ref mut d, Some(i)) => d.field("index", &i),
						(ref mut d, _) => d,
					}
						.field("slice", &slice)
						.field("item", &TrieAwareDebugNode{
							trie: self.trie,
							node_key: item,
							partial_key: self.partial_key
								.clone_append_optional_slice_and_nibble(Some(&slice), None),
							index: None,
						})
						.finish(),
				Ok(Node::Branch(ref nodes, ref value)) => {
					let nodes: Vec<TrieAwareDebugNode<L>> = nodes.0.iter(nodes.1)
						.enumerate()
						.filter_map(|(i, n)| n.map(|n| (i, n)))
						.map(|(i, n)| TrieAwareDebugNode {
							trie: self.trie,
							index: Some(i as u8),
							node_key: n,
							partial_key: self.partial_key
								.clone_append_optional_slice_and_nibble(None, Some(i as u8)),
						})
						.collect();
					match (f.debug_struct("Node::Branch"), self.index) {
						(ref mut d, Some(ref i)) => d.field("index", i),
						(ref mut d, _) => d,
					}
						.field("nodes", &nodes)
						.field("value", &value)
						.finish()
				},
				Ok(Node::NibbledBranch(slice, nodes, value)) => {
					let nodes: Vec<TrieAwareDebugNode<L>> = nodes.0.iter(nodes.1)
						.enumerate()
						.filter_map(|(i, n)| n.map(|n| (i, n)))
						.map(|(i, n)| TrieAwareDebugNode {
							trie: self.trie,
							index: Some(i as u8),
							node_key: n,
							partial_key: self.partial_key
								.clone_append_optional_slice_and_nibble(Some(&slice), Some(i as u8)),
						}).collect();
					match (f.debug_struct("Node::NibbledBranch"), self.index) {
						(ref mut d, Some(ref i)) => d.field("index", i),
						(ref mut d, _) => d,
					}
						.field("slice", &slice)
						.field("nodes", &nodes)
						.field("value", &value)
						.finish()
				},
				Ok(Node::Empty) => f.debug_struct("Node::Empty").finish(),

				Err(e) => f.debug_struct("BROKEN_NODE")
					.field("index", &self.index)
					.field("key", &self.node_key)
					.field("error", &format!("ERROR decoding node branch Rlp: {}", e))
					.finish()
			}
		} else {
			f.debug_struct("BROKEN_NODE")
				.field("index", &self.index)
				.field("key", &self.node_key)
				.field("error", &"Not found")
				.finish()
		}
	}
}

#[cfg(feature="std")]
impl<'db, L> fmt::Debug for TrieDB<'db, L>
where
	L: TrieLayout,
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let root_rlp = self.root_data().unwrap();
		f.debug_struct("TrieDB")
			.field("hash_count", &self.hash_count)
			.field("root", &TrieAwareDebugNode {
				trie: self,
				node_key: &root_rlp[..],
				partial_key: NibbleVec::new(),
				index: None,
			})
			.finish()
	}
}


#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, Eq, PartialEq)]
enum Status {
	Entering,
	At,
	AtChild(usize),
	Exiting,
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Eq, PartialEq)]
struct Crumb<N> {
	node: OwnedNode<N>,
	status: Status,
}

impl<N: NibbleOps> Crumb<N> {
	/// Move on to next status in the node's sequence.
	fn increment(&mut self) {
		self.status = match (&self.status, &self.node) {
			(_, &OwnedNode::Empty) => Status::Exiting,
			(&Status::Entering, _) => Status::At,
			(&Status::At, &OwnedNode::Branch(_))
				| (&Status::At, &OwnedNode::NibbledBranch(..)) => Status::AtChild(0),
			(&Status::AtChild(x), &OwnedNode::Branch(_))
				| (&Status::AtChild(x), &OwnedNode::NibbledBranch(..))
				if x < (N::NIBBLE_LENGTH - 1) => Status::AtChild(x + 1),
			_ => Status::Exiting,
		}
	}
}

/// Iterator for going through all values in the trie.
pub struct TrieDBIterator<'a, L: TrieLayout> {
	db: &'a TrieDB<'a, L>,
	trail: Vec<Crumb<L::Nibble>>,
	key_nibbles: NibbleVec<L::Nibble>,
}

impl<'a, L: TrieLayout> TrieDBIterator<'a, L> {
	/// Create a new iterator.
	pub fn new(db: &'a TrieDB<L>) -> Result<TrieDBIterator<'a, L>, TrieHash<L>, CError<L>> {
		let mut r = TrieDBIterator {
			db,
			trail: Vec::with_capacity(8),
			key_nibbles: NibbleVec::new(),
		};
		db.root_data().and_then(|root_data| r.descend(&root_data))?;
		Ok(r)
	}

	fn seek<'key>(
		&mut self,
		node_data: &DBValue,
		key: NibbleSlice<'key, L::Nibble>,
	) -> Result<(), TrieHash<L>, CError<L>> {
		let mut node_data = Cow::Borrowed(node_data);
		let mut partial = key;
		let mut full_key_nibbles = 0;
		loop {
			let data = {
				let node = L::Codec::decode(&node_data)
					.map_err(|e| Box::new(TrieError::DecoderError(<TrieHash<L>>::default(), e)))?;
				match node {
					Node::Leaf(slice, _) => {
						if slice >= partial {
							self.trail.push(Crumb {
								status: Status::Entering,
								node: node.clone().into(),
							});
						} else {
							self.trail.push(Crumb {
								status: Status::Exiting,
								node: node.clone().into(),
							});
						}

						self.key_nibbles.append_partial(slice.right());
						return Ok(())
					},
					Node::Extension(ref slice, ref item) => {
						if partial.starts_with(slice) {
							self.trail.push(Crumb {
								status: Status::At,
								node: node.clone().into(),
							});
							self.key_nibbles.append_partial(slice.right());
							full_key_nibbles += slice.len();
							partial = partial.mid(slice.len());
							let data = self.db
								.get_raw_or_lookup(&*item, key.back(full_key_nibbles)
								.left())?;
							data
						} else {
							self.descend(&node_data)?;
							return Ok(())
						}
					},
					Node::Branch(ref nodes, _) => match partial.is_empty() {
						true => {
							self.trail.push(Crumb {
								status: Status::Entering,
								node: node.clone().into(),
							});
							return Ok(())
						},
						false => {
							let i = partial.at(0);
							self.trail.push(Crumb {
								status: Status::AtChild(i as usize),
								node: node.clone().into(),
							});
							self.key_nibbles.push(i);
							if let Some(ref child) = nodes.0.slice_at(i as usize, nodes.1) {
								full_key_nibbles += 1;
								partial = partial.mid(1);
								let child = self.db
									.get_raw_or_lookup(&*child, key.back(full_key_nibbles)
									.left())?;
								child
							} else {
								return Ok(())
							}
						}
					},
					Node::NibbledBranch(ref slice, ref nodes, _) => {
						if !partial.starts_with(slice) {
							self.descend(&node_data)?;
							return Ok(())
						}
						if partial.len() == slice.len() {
							self.trail.push(Crumb {
								status: Status::Entering,
								node: node.clone().into(),
							});
							return Ok(())
						} else {

							let i = partial.at(slice.len());
							self.trail.push(Crumb {
								status: Status::AtChild(i as usize),
								node: node.clone().into(),
							});
							self.key_nibbles.append_partial(slice.right());
							self.key_nibbles.push(i);
							if let Some(ref child) = nodes.0.slice_at(i as usize, nodes.1) {
								full_key_nibbles += slice.len() + 1;
								partial = partial.mid(slice.len() + 1);
								let child = self.db
									.get_raw_or_lookup(&*child, key.back(full_key_nibbles)
									.left())?;
								child
							} else {
								return Ok(())
							}
						}
					},
					Node::Empty => return Ok(()),
				}
			};

			node_data = data;
		}
	}

	/// Descend into a payload.
	fn descend(&mut self, d: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		let node_data = &self.db.get_raw_or_lookup(d, self.key_nibbles.as_prefix())?;
		let node = L::Codec::decode(&node_data)
			.map_err(|e| Box::new(TrieError::DecoderError(<TrieHash<L>>::default(), e)))?;
		Ok(self.descend_into_node(node.into()))
	}

	/// Descend into a payload.
	fn descend_into_node(&mut self, node: OwnedNode<L::Nibble>) {
		let trail = &mut self.trail;
		let key_nibbles = &mut self.key_nibbles;
		trail.push(Crumb { status: Status::Entering, node });
		match &trail.last().expect("just pushed item; qed").node {
			&OwnedNode::Leaf(ref n, _)
				| &OwnedNode::Extension(ref n, _)
				| &OwnedNode::NibbledBranch(ref n, _)
				=> key_nibbles.append(n),
			_ => {}
		}
	}

	/// The present key. This can only be called on valued node (key is therefore
	/// aligned to byte).
	fn key(&self) -> NibbleSlice<L::Nibble> {
		self.key_nibbles.as_nibbleslice().expect("a key is aligned to byte;qed")
	}

}

impl<'a, L: TrieLayout> TrieIterator<L> for TrieDBIterator<'a, L> {
	/// Position the iterator on the first element with key >= `key`
	fn seek(&mut self, key: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		self.trail.clear();
		self.key_nibbles.clear();
		let root_node = self.db.root_data()?;
		self.seek(&root_node, NibbleSlice::new(key.as_ref()))
	}
}

impl<'a, L: TrieLayout> Iterator for TrieDBIterator<'a, L> {
	type Item = TrieItem<'a, TrieHash<L>, CError<L>>;

	fn next(&mut self) -> Option<Self::Item> {
		enum IterStep<'b, O, E> {
			Continue,
			PopTrail,
			Descend(Result<Cow<'b, DBValue>, O, E>),
		}
		loop {
			let iter_step = {
				self.trail.last_mut()?.increment();
				let b = self.trail.last().expect("trail.last_mut().is_some(); qed");

				match (b.status.clone(), &b.node) {
					(Status::Exiting, n) => {
						match *n {
							OwnedNode::Leaf(ref n, _) | OwnedNode::Extension(ref n, _) => {
								self.key_nibbles.drop_lasts(n.len());
							},
							OwnedNode::Branch(_) => { self.key_nibbles.pop(); },
							OwnedNode::NibbledBranch(ref n, _) => {
								self.key_nibbles.drop_lasts(n.len() + 1);
							},
							OwnedNode::Empty => {},
						}
						IterStep::PopTrail
					},
					(Status::At, &OwnedNode::NibbledBranch(_, ref branch))
						| (Status::At, &OwnedNode::Branch(ref branch)) if branch.has_value() => {
						let value = branch.get_value().expect("already checked `has_value`");
						return Some(Ok((self.key().right().1.into(), DBValue::from_slice(value))));
					},
					(Status::At, &OwnedNode::Leaf(_, ref v)) => {
						return Some(Ok((self.key().right().1.into(), v.clone())));
					},
					(Status::At, &OwnedNode::Extension(_, ref d)) => {
						IterStep::Descend::<TrieHash<L>, CError<L>>(
							self.db.get_raw_or_lookup(&*d, self.key_nibbles.as_prefix())
						)
					},
					(Status::At, &OwnedNode::Branch(_))
						| (Status::At, &OwnedNode::NibbledBranch(_, _)) => IterStep::Continue,
					(Status::AtChild(i), &OwnedNode::Branch(ref branch))
						| (Status::AtChild(i), &OwnedNode::NibbledBranch(_, ref branch)) 
						if branch.index(i).is_some() => {
						match i {
							0 => self.key_nibbles.push(0),
							i => {
								self.key_nibbles.pop();
								self.key_nibbles.push(i as u8);
							},
						}
						IterStep::Descend::<TrieHash<L>, CError<L>>(self.db.get_raw_or_lookup(
							&branch.index(i).expect("this arm guarded by branch[i].is_some(); qed"),
							self.key_nibbles.as_prefix()))
					},
					(Status::AtChild(i), &OwnedNode::Branch(_))
						| (Status::AtChild(i), &OwnedNode::NibbledBranch(_, _)) => {
						if i == 0 {
							self.key_nibbles.push(0);
						}
						IterStep::Continue
					},
					_ => panic!() // Should never see Entering or AtChild without a Branch here.
				}
			};

			match iter_step {
				IterStep::PopTrail => {
					self.trail.pop();
				},
				IterStep::Descend::<TrieHash<L>, CError<L>>(Ok(d)) => {
					let node = L::Codec::decode(&d).ok()?;
					self.descend_into_node(node.into())
				},
				IterStep::Descend::<TrieHash<L>, CError<L>>(Err(e)) => {
					return Some(Err(e))
				}
				IterStep::Continue => {},
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use memory_db::{MemoryDB, PrefixedKey};
	use keccak_hasher::KeccakHasher;
	use DBValue;
	use reference_trie::{RefTrieDB, RefTrieDBMut, RefLookup, Trie, TrieMut, NibbleSlice};
	use reference_trie::{RefTrieDBNoExt, RefTrieDBMutNoExt};

	#[test]
	fn iterator_works() {
		let pairs = vec![
			(hex!("0103000000000000000464").to_vec(), hex!("fffffffffe").to_vec()),
			(hex!("0103000000000000000469").to_vec(), hex!("ffffffffff").to_vec()),
		];

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
			for (x, y) in &pairs {
				t.insert(x, y).unwrap();
			}
		}

		let trie = RefTrieDB::new(&memdb, &root).unwrap();

		let iter = trie.iter().unwrap();
		let mut iter_pairs = Vec::new();
		for pair in iter {
			let (key, value) = pair.unwrap();
			iter_pairs.push((key, value.to_vec()));
		}

		assert_eq!(pairs, iter_pairs);
	}

	#[test]
	fn iterator_works_without_extension() {
		let pairs = vec![
			(hex!("0103000000000000000464").to_vec(), hex!("fffffffffe").to_vec()),
			(hex!("0103000000000000000469").to_vec(), hex!("ffffffffff").to_vec()),
		];

		let mut memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
			for (x, y) in &pairs {
				t.insert(x, y).unwrap();
			}
		}

		let trie = RefTrieDBNoExt::new(&memdb, &root).unwrap();

		let iter = trie.iter().unwrap();
		let mut iter_pairs = Vec::new();
		for pair in iter {
			let (key, value) = pair.unwrap();
			iter_pairs.push((key, value.to_vec()));
		}

		assert_eq!(pairs, iter_pairs);
	}

	#[test]
	fn iterator_seek_works() {
		let pairs = vec![
			(hex!("0103000000000000000464").to_vec(), hex!("fffffffffe").to_vec()),
			(hex!("0103000000000000000469").to_vec(), hex!("ffffffffff").to_vec()),
		];

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
			for (x, y) in &pairs {
				t.insert(x, y).unwrap();
			}
		}

		let t = RefTrieDB::new(&memdb, &root).unwrap();

		let mut iter = t.iter().unwrap();
		assert_eq!(
			iter.next().unwrap().unwrap(),
			(
				hex!("0103000000000000000464").to_vec(),
				DBValue::from_slice(&hex!("fffffffffe")[..]),
			)
		);
		iter.seek(&hex!("00")[..]).unwrap();
		assert_eq!(
			pairs,
			iter.map(|x| x.unwrap())
				.map(|(k, v)| (k, v[..].to_vec()))
				.collect::<Vec<_>>()
		);
		let mut iter = t.iter().unwrap();
		iter.seek(&hex!("0103000000000000000465")[..]).unwrap();
		assert_eq!(
			&pairs[1..],
			&iter.map(|x| x.unwrap())
				.map(|(k, v)| (k, v[..].to_vec()))
				.collect::<Vec<_>>()[..]
		);
	}

	#[test]
	fn iterator_seek_works_without_extension() {
		let pairs = vec![
			(hex!("0103000000000000000464").to_vec(), hex!("fffffffffe").to_vec()),
			(hex!("0103000000000000000469").to_vec(), hex!("ffffffffff").to_vec()),
		];

		let mut memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
			for (x, y) in &pairs {
				t.insert(x, y).unwrap();
			}
		}

		let t = RefTrieDBNoExt::new(&memdb, &root).unwrap();

		let mut iter = t.iter().unwrap();
		assert_eq!(
			iter.next().unwrap().unwrap(),
			(hex!("0103000000000000000464").to_vec(), DBValue::from_slice(&hex!("fffffffffe")[..]))
		);
		iter.seek(&hex!("00")[..]).unwrap();
		assert_eq!(
			pairs,
			iter.map(|x| x.unwrap()).map(|(k, v)| (k, v[..].to_vec())).collect::<Vec<_>>(),
		);
		let mut iter = t.iter().unwrap();
		iter.seek(&hex!("0103000000000000000465")[..]).unwrap();
		assert_eq!(
			&pairs[1..],
			&iter.map(|x| x.unwrap()).map(|(k, v)| (k, v[..].to_vec())).collect::<Vec<_>>()[..],
		);
	}

	#[test]
	fn iterator() {
		let d = vec![
			DBValue::from_slice(b"A"),
			DBValue::from_slice(b"AA"),
			DBValue::from_slice(b"AB"),
			DBValue::from_slice(b"B"),
		];

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
			for x in &d {
				t.insert(x, x).unwrap();
			}
		}

		let t = RefTrieDB::new(&memdb, &root).unwrap();
		assert_eq!(
			d.iter()
				.map(|i| i.clone().into_vec())
				.collect::<Vec<_>>(),
			t.iter()
				.unwrap()
				.map(|x| x.unwrap().0)
				.collect::<Vec<_>>()
		);
		assert_eq!(d, t.iter().unwrap().map(|x| x.unwrap().1).collect::<Vec<_>>());
	}

	#[test]
	fn iterator_without_extension() {
		let d = vec![
			DBValue::from_slice(b"A"),
			DBValue::from_slice(b"AA"),
			DBValue::from_slice(b"AB"),
			DBValue::from_slice(b"B"),
		];

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
			for x in &d {
				t.insert(x, x).unwrap();
			}
		}

		let t = RefTrieDBNoExt::new(&memdb, &root).unwrap();
		assert_eq!(
			d.iter().map(|i| i.clone().into_vec()).collect::<Vec<_>>(),
			t.iter().unwrap().map(|x| x.unwrap().0).collect::<Vec<_>>(),
		);
		assert_eq!(d, t.iter().unwrap().map(|x| x.unwrap().1).collect::<Vec<_>>());
	}

	#[test]
	fn iterator_seek() {
		let d = vec![
			DBValue::from_slice(b"A"),
			DBValue::from_slice(b"AA"),
			DBValue::from_slice(b"AB"),
			DBValue::from_slice(b"B"),
		];

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
			for x in &d {
				t.insert(x, x).unwrap();
			}
		}

		let t = RefTrieDBNoExt::new(&memdb, &root).unwrap();
		let mut iter = t.iter().unwrap();
		assert_eq!(iter.next().unwrap().unwrap(), (b"A".to_vec(), DBValue::from_slice(b"A")));
		iter.seek(b"!").unwrap();
		assert_eq!(d, iter.map(|x| x.unwrap().1).collect::<Vec<_>>());
		let mut iter = t.iter().unwrap();
		iter.seek(b"A").unwrap();
		assert_eq!(d, &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
		let mut iter = t.iter().unwrap();
		iter.seek(b"AA").unwrap();
		assert_eq!(&d[1..], &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
		let mut iter = t.iter().unwrap();
		iter.seek(b"A!").unwrap();
		assert_eq!(&d[1..], &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
		let mut iter = t.iter().unwrap();
		iter.seek(b"AB").unwrap();
		assert_eq!(&d[2..], &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
		let mut iter = t.iter().unwrap();
		iter.seek(b"AB!").unwrap();
		assert_eq!(&d[3..], &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
		let mut iter = t.iter().unwrap();
		iter.seek(b"B").unwrap();
		assert_eq!(&d[3..], &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
		let mut iter = t.iter().unwrap();
		iter.seek(b"C").unwrap();
		assert_eq!(&d[4..], &iter.map(|x| x.unwrap().1).collect::<Vec<_>>()[..]);
	}

	#[test]
	fn get_length_with_extension() {
		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
			t.insert(b"A", b"ABC").unwrap();
			t.insert(b"B", b"ABCBAAAAAAAAAAAAAAAAAAAAAAAAAAAA").unwrap();
		}

		let t = RefTrieDB::new(&memdb, &root).unwrap();
		assert_eq!(t.get_with(b"A", |x: &[u8]| x.len()).unwrap(), Some(3));
		assert_eq!(t.get_with(b"B", |x: &[u8]| x.len()).unwrap(), Some(32));
		assert_eq!(t.get_with(b"C", |x: &[u8]| x.len()).unwrap(), None);
	}

	#[test]
	fn get_length_without_extension() {
		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
			t.insert(b"A", b"ABC").unwrap();
			t.insert(b"B", b"ABCBA").unwrap();
		}

		let t = RefTrieDBNoExt::new(&memdb, &root).unwrap();
		assert_eq!(t.get_with(b"A", |x: &[u8]| x.len()).unwrap(), Some(3));
		assert_eq!(t.get_with(b"B", |x: &[u8]| x.len()).unwrap(), Some(5));
		assert_eq!(t.get_with(b"C", |x: &[u8]| x.len()).unwrap(), None);
	}

	#[test]
	fn debug_output_supports_pretty_print() {
		let d = vec![
			DBValue::from_slice(b"A"),
			DBValue::from_slice(b"AA"),
			DBValue::from_slice(b"AB"),
			DBValue::from_slice(b"B"),
		];

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		let root = {
			let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
			for x in &d {
				t.insert(x, x).unwrap();
			}
			t.root().clone()
		};
		let t = RefTrieDB::new(&memdb, &root).unwrap();

		assert_eq!(format!("{:#?}", t),
"TrieDB {
    hash_count: 0,
    root: Node::Extension {
        slice: 4,
        item: Node::Branch {
            nodes: [
                Node::Branch {
                    index: 1,
                    nodes: [
                        Node::Branch {
                            index: 4,
                            nodes: [
                                Node::Leaf {
                                    index: 1,
                                    slice: ,
                                    value: [
                                        65,
                                        65,
                                    ],
                                },
                                Node::Leaf {
                                    index: 2,
                                    slice: ,
                                    value: [
                                        65,
                                        66,
                                    ],
                                },
                            ],
                            value: None,
                        },
                    ],
                    value: Some(
                        [
                            65,
                        ],
                    ),
                },
                Node::Leaf {
                    index: 2,
                    slice: ,
                    value: [
                        66,
                    ],
                },
            ],
            value: None,
        },
    },
}");
	
	}

	#[test]
	fn test_lookup_with_corrupt_data_returns_decoder_error() {

		let mut memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
		let mut root = Default::default();
		{
			let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
			t.insert(b"A", b"ABC").unwrap();
			t.insert(b"B", b"ABCBA").unwrap();
		}

		let t = RefTrieDB::new(&memdb, &root).unwrap();

		// query for an invalid data type to trigger an error
		let q = |x: &[u8]| x.len() < 64;
		let lookup = RefLookup { db: t.db(), query: q, hash: root };
		let query_result = lookup.look_up(NibbleSlice::new(b"A"));
		assert_eq!(query_result.unwrap().unwrap(), true);
	}
}
