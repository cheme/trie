// Copyright 2017, 2021 Parity Technologies
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

use super::{CError, DBValue, Result, Trie, TrieHash, TrieIterator, TrieLayout};
use crate::{
	nibble::{NibbleOps, NibbleSlice, NibbleVec},
	node::{Node, NodeHandle, NodePlan, OwnedNode, Value},
	triedb::TrieDB,
	TrieError, TrieItem, TrieKeyItem,
};
use hash_db::{Hasher, Prefix, EMPTY_PREFIX};

use crate::rstd::{boxed::Box, sync::Arc, vec::Vec};

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, Copy, Eq, PartialEq)]
enum Status {
	Entering,
	At,
	AtChild(usize),
	Exiting,
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Eq, PartialEq)]
struct Crumb<L: TrieLayout<N>, const N: usize> {
	hash: Option<TrieHash<L, N>>,
	node: Arc<OwnedNode<DBValue, N>>,
	status: Status,
}

impl<L: TrieLayout<N>, const N: usize> Crumb<L, N> {
	/// Move on to next status in the node's sequence.
	fn increment(&mut self) {
		self.status = match (self.status, self.node.node_plan()) {
			(Status::Entering, NodePlan::Extension { .. }) => Status::At,
			(Status::Entering, NodePlan::Branch { .. }) |
			(Status::Entering, NodePlan::NibbledBranch { .. }) => Status::At,
			(Status::At, NodePlan::Branch { .. }) |
			(Status::At, NodePlan::NibbledBranch { .. }) => Status::AtChild(0),
			(Status::AtChild(x), NodePlan::Branch { .. }) |
			(Status::AtChild(x), NodePlan::NibbledBranch { .. })
				if x < (N - 1) =>
				Status::AtChild(x + 1),
			_ => Status::Exiting,
		}
	}
}

/// Iterator for going through all nodes in the trie in pre-order traversal order.
pub struct TrieDBRawIterator<L: TrieLayout<N>, const N: usize> {
	trail: Vec<Crumb<L, N>>,
	key_nibbles: NibbleVec<N>,
}

impl<L: TrieLayout<N>, const N: usize> TrieDBRawIterator<L, N> {
	/// Create a new empty iterator.
	pub fn empty() -> Self {
		Self { trail: Vec::new(), key_nibbles: NibbleVec::new() }
	}

	/// Create a new iterator.
	pub fn new(db: &TrieDB<L, N>) -> Result<Self, TrieHash<L, N>, CError<L, N>> {
		let mut r =
			TrieDBRawIterator { trail: Vec::with_capacity(8), key_nibbles: NibbleVec::new() };
		let (root_node, root_hash) = db.get_raw_or_lookup(
			*db.root(),
			NodeHandle::Hash(db.root().as_ref()),
			EMPTY_PREFIX,
			true,
		)?;
		r.descend(root_node, root_hash);
		Ok(r)
	}

	/// Create a new iterator, but limited to a given prefix.
	pub fn new_prefixed(
		db: &TrieDB<L, N>,
		prefix: &[u8],
	) -> Result<Self, TrieHash<L, N>, CError<L, N>> {
		let mut iter = TrieDBRawIterator::new(db)?;
		iter.prefix(db, prefix)?;

		Ok(iter)
	}

	/// Create a new iterator, but limited to a given prefix.
	/// It then do a seek operation from prefixed context (using `seek` lose
	/// prefix context by default).
	pub fn new_prefixed_then_seek(
		db: &TrieDB<L, N>,
		prefix: &[u8],
		start_at: &[u8],
	) -> Result<Self, TrieHash<L, N>, CError<L, N>> {
		let mut iter = TrieDBRawIterator::new(db)?;
		iter.prefix_then_seek(db, prefix, start_at)?;
		Ok(iter)
	}

	/// Descend into a payload.
	fn descend(&mut self, node: OwnedNode<DBValue, N>, node_hash: Option<TrieHash<L, N>>) {
		self.trail
			.push(Crumb { hash: node_hash, status: Status::Entering, node: Arc::new(node) });
	}

	/// Fetch value by hash at a current node height
	pub(crate) fn fetch_value(
		db: &TrieDB<L, N>,
		key: &[u8],
		prefix: Prefix,
	) -> Result<DBValue, TrieHash<L, N>, CError<L, N>> {
		let mut res = TrieHash::<L, N>::default();
		res.as_mut().copy_from_slice(key);
		db.fetch_value(res, prefix)
	}

	/// Seek a node position at 'key' for iterator.
	/// Returns true if the cursor is at or after the key, but still shares
	/// a common prefix with the key, return false if the key do not
	/// share its prefix with the node.
	/// This indicates if there is still nodes to iterate over in the case
	/// where we limit iteration to 'key' as a prefix.
	pub(crate) fn seek(
		&mut self,
		db: &TrieDB<L, N>,
		key: &[u8],
	) -> Result<bool, TrieHash<L, N>, CError<L, N>> {
		self.trail.clear();
		self.key_nibbles.clear();
		let key = NibbleSlice::new(key);

		let (mut node, mut node_hash) = db.get_raw_or_lookup(
			<TrieHash<L, N>>::default(),
			NodeHandle::Hash(db.root().as_ref()),
			EMPTY_PREFIX,
			true,
		)?;
		let mut partial = key;
		let mut full_key_nibbles = 0;
		loop {
			let (next_node, next_node_hash) = {
				self.descend(node, node_hash);
				let crumb = self.trail.last_mut().expect(
					"descend_into_node pushes a crumb onto the trial; \
						thus the trail is non-empty; qed",
				);
				let node_data = crumb.node.data();

				match crumb.node.node_plan() {
					NodePlan::Leaf { partial: partial_plan, .. } => {
						let slice = partial_plan.build(node_data);
						if slice < partial {
							crumb.status = Status::Exiting;
							return Ok(false)
						}
						return Ok(slice.starts_with(&partial))
					},
					NodePlan::Extension { partial: partial_plan, child } => {
						let slice = partial_plan.build(node_data);
						if !partial.starts_with(&slice) {
							if slice < partial {
								crumb.status = Status::Exiting;
								self.key_nibbles.append_partial(slice.right());
								return Ok(false)
							}
							return Ok(slice.starts_with(&partial))
						}

						full_key_nibbles += slice.len();
						partial = partial.mid(slice.len());
						crumb.status = Status::At;
						self.key_nibbles.append_partial(slice.right());

						let prefix = key.back(full_key_nibbles);
						db.get_raw_or_lookup(
							node_hash.unwrap_or_default(),
							child.build(node_data),
							prefix.left(),
							true,
						)?
					},
					NodePlan::Branch { value: _, children } => {
						if partial.is_empty() {
							return Ok(true)
						}

						let i = partial.at(0);
						crumb.status = Status::AtChild(i as usize);
						self.key_nibbles.push(i);

						if let Some(child) = &children[i as usize] {
							full_key_nibbles += 1;
							partial = partial.mid(1);

							let prefix = key.back(full_key_nibbles);
							db.get_raw_or_lookup(
								node_hash.unwrap_or_default(),
								child.build(node_data),
								prefix.left(),
								true,
							)?
						} else {
							return Ok(false)
						}
					},
					NodePlan::NibbledBranch { partial: partial_plan, value: _, children } => {
						let slice = partial_plan.build(node_data);
						if !partial.starts_with(&slice) {
							if slice < partial {
								crumb.status = Status::Exiting;
								self.key_nibbles.append_partial(slice.right());
								self.key_nibbles.push((N - 1) as u8);
								return Ok(false)
							}
							return Ok(slice.starts_with(&partial))
						}

						full_key_nibbles += slice.len();
						partial = partial.mid(slice.len());

						if partial.is_empty() {
							return Ok(true)
						}

						let i = partial.at(0);
						crumb.status = Status::AtChild(i as usize);
						self.key_nibbles.append_partial(slice.right());
						self.key_nibbles.push(i);

						if let Some(child) = &children[i as usize] {
							full_key_nibbles += 1;
							partial = partial.mid(1);

							let prefix = key.back(full_key_nibbles);
							db.get_raw_or_lookup(
								node_hash.unwrap_or_default(),
								child.build(node_data),
								prefix.left(),
								true,
							)?
						} else {
							return Ok(false)
						}
					},
					NodePlan::Empty => {
						if !partial.is_empty() {
							crumb.status = Status::Exiting;
							return Ok(false)
						}
						return Ok(true)
					},
				}
			};

			node = next_node;
			node_hash = next_node_hash;
		}
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	fn prefix(
		&mut self,
		db: &TrieDB<L, N>,
		prefix: &[u8],
	) -> Result<(), TrieHash<L, N>, CError<L, N>> {
		if self.seek(db, prefix)? {
			if let Some(v) = self.trail.pop() {
				self.trail.clear();
				self.trail.push(v);
			}
		} else {
			self.trail.clear();
		}

		Ok(())
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	fn prefix_then_seek(
		&mut self,
		db: &TrieDB<L, N>,
		prefix: &[u8],
		seek: &[u8],
	) -> Result<(), TrieHash<L, N>, CError<L, N>> {
		if prefix.is_empty() {
			// There's no prefix, so just seek.
			return self.seek(db, seek).map(|_| ())
		}

		if seek.is_empty() || seek <= prefix {
			// Either we're not supposed to seek anywhere,
			// or we're supposed to seek *before* the prefix,
			// so just directly go to the prefix.
			return self.prefix(db, prefix)
		}

		if !seek.starts_with(prefix) {
			// We're supposed to seek *after* the prefix,
			// so just return an empty iterator.
			self.trail.clear();
			return Ok(())
		}

		if !self.seek(db, prefix)? {
			// The database doesn't have a key with such a prefix.
			self.trail.clear();
			return Ok(())
		}

		// Now seek forward again.
		self.seek(db, seek)?;

		let prefix_len = prefix.len() * N;
		let mut len = 0;
		// look first prefix in trail
		for i in 0..self.trail.len() {
			match self.trail[i].node.node_plan() {
				NodePlan::Empty => {},
				NodePlan::Branch { .. } => {
					len += 1;
				},
				NodePlan::Leaf { partial, .. } => {
					len += partial.len();
				},
				NodePlan::Extension { partial, .. } => {
					len += partial.len();
				},
				NodePlan::NibbledBranch { partial, .. } => {
					len += 1;
					len += partial.len();
				},
			}
			if len > prefix_len {
				self.trail = self.trail.split_off(i);
				return Ok(())
			}
		}

		self.trail.clear();
		Ok(())
	}

	/// Fetches the next raw item.
	//
	/// Must be called with the same `db` as when the iterator was created.
	pub(crate) fn next_raw_item(
		&mut self,
		db: &TrieDB<L, N>,
	) -> Option<
		Result<
			(&NibbleVec<N>, Option<&TrieHash<L, N>>, &Arc<OwnedNode<DBValue, N>>),
			TrieHash<L, N>,
			CError<L, N>,
		>,
	> {
		loop {
			let crumb = self.trail.last_mut()?;
			let node_data = crumb.node.data();

			match (crumb.status, crumb.node.node_plan()) {
				(Status::Entering, _) => {
					// This is only necessary due to current borrow checker's limitation.
					let crumb = self.trail.last_mut().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
					crumb.increment();
					return Some(Ok((&self.key_nibbles, crumb.hash.as_ref(), &crumb.node)))
				},
				(Status::Exiting, node) => {
					match node {
						NodePlan::Empty | NodePlan::Leaf { .. } => {},
						NodePlan::Extension { partial, .. } => {
							self.key_nibbles.drop_lasts(partial.len());
						},
						NodePlan::Branch { .. } => {
							self.key_nibbles.pop();
						},
						NodePlan::NibbledBranch { partial, .. } => {
							self.key_nibbles.drop_lasts(partial.len() + 1);
						},
					}
					self.trail.pop().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
					self.trail.last_mut()?.increment();
				},
				(Status::At, NodePlan::Extension { partial: partial_plan, child }) => {
					let partial = partial_plan.build(node_data);
					self.key_nibbles.append_partial(partial.right());

					match db.get_raw_or_lookup(
						crumb.hash.unwrap_or_default(),
						child.build(node_data),
						self.key_nibbles.as_prefix(),
						true,
					) {
						Ok((node, node_hash)) => {
							self.descend(node, node_hash);
						},
						Err(err) => {
							crumb.increment();
							return Some(Err(err))
						},
					}
				},
				(Status::At, NodePlan::Branch { .. }) => {
					self.key_nibbles.push(0);
					crumb.increment();
				},
				(Status::At, NodePlan::NibbledBranch { partial: partial_plan, .. }) => {
					let partial = partial_plan.build(node_data);
					self.key_nibbles.append_partial(partial.right());
					self.key_nibbles.push(0);
					crumb.increment();
				},
				(Status::AtChild(i), NodePlan::Branch { children, .. }) |
				(Status::AtChild(i), NodePlan::NibbledBranch { children, .. }) => {
					if let Some(child) = &children[i] {
						self.key_nibbles.pop();
						self.key_nibbles.push(i as u8);

						match db.get_raw_or_lookup(
							crumb.hash.unwrap_or_default(),
							child.build(node_data),
							self.key_nibbles.as_prefix(),
							true,
						) {
							Ok((node, node_hash)) => {
								self.descend(node, node_hash);
							},
							Err(err) => {
								crumb.increment();
								return Some(Err(err))
							},
						}
					} else {
						crumb.increment();
					}
				},
				_ => panic!(
					"Crumb::increment and TrieDBNodeIterator are implemented so that \
						the above arms are the only possible states"
				),
			}
		}
	}

	/// Fetches the next trie item.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn next_item(
		&mut self,
		db: &TrieDB<L, N>,
	) -> Option<TrieItem<TrieHash<L, N>, CError<L, N>>> {
		while let Some(raw_item) = self.next_raw_item(db) {
			let (prefix, _, node) = match raw_item {
				Ok(raw_item) => raw_item,
				Err(err) => return Some(Err(err)),
			};

			let mut prefix = prefix.clone();
			let value = match node.node() {
				Node::Leaf(partial, value) => {
					prefix.append_partial(partial.right());
					value
				},
				Node::Branch(_, value) => match value {
					Some(value) => value,
					None => continue,
				},
				Node::NibbledBranch(partial, _, value) => {
					prefix.append_partial(partial.right());
					match value {
						Some(value) => value,
						None => continue,
					}
				},
				_ => continue,
			};

			let prefix = prefix.as_prefix();
			let key = prefix.slice.to_vec();
			if prefix.align > 0 {
				return Some(Err(Box::new(TrieError::ValueAtIncompleteKey(
					key,
					(prefix.last, prefix.align),
				))))
			}

			let value = match value {
				Value::Node(hash) => match Self::fetch_value(db, &hash, prefix) {
					Ok(value) => value,
					Err(err) => return Some(Err(err)),
				},
				Value::Inline(value) => value.to_vec(),
			};

			return Some(Ok((key, value)))
		}
		None
	}

	/// Fetches the next key.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn next_key(
		&mut self,
		db: &TrieDB<L, N>,
	) -> Option<TrieKeyItem<TrieHash<L, N>, CError<L, N>>> {
		while let Some(raw_item) = self.next_raw_item(db) {
			let (prefix, _, node) = match raw_item {
				Ok(raw_item) => raw_item,
				Err(err) => return Some(Err(err)),
			};

			let mut prefix = prefix.clone();
			match node.node() {
				Node::Leaf(partial, _) => {
					prefix.append_partial(partial.right());
				},
				Node::Branch(_, value) =>
					if value.is_none() {
						continue
					},
				Node::NibbledBranch(partial, _, value) => {
					prefix.append_partial(partial.right());
					if value.is_none() {
						continue
					}
				},
				_ => continue,
			};

			let prefix = prefix.as_prefix();
			let key = prefix.slice.to_vec();
			if prefix.align > 0 {
				return Some(Err(Box::new(TrieError::ValueAtIncompleteKey(
					key,
					(prefix.last, prefix.align),
				))))
			}

			return Some(Ok(key))
		}
		None
	}
}

/// Iterator for going through all nodes in the trie in pre-order traversal order.
pub struct TrieDBNodeIterator<'a, 'cache, L: TrieLayout<N>, const N: usize> {
	db: &'a TrieDB<'a, 'cache, L, N>,
	raw_iter: TrieDBRawIterator<L, N>,
}

impl<'a, 'cache, L: TrieLayout<N>, const N: usize> TrieDBNodeIterator<'a, 'cache, L, N> {
	/// Create a new iterator.
	pub fn new(db: &'a TrieDB<'a, 'cache, L, N>) -> Result<Self, TrieHash<L, N>, CError<L, N>> {
		Ok(Self { raw_iter: TrieDBRawIterator::new(db)?, db })
	}

	/// Restore an iterator from a raw iterator.
	pub fn from_raw(db: &'a TrieDB<'a, 'cache, L, N>, raw_iter: TrieDBRawIterator<L, N>) -> Self {
		Self { db, raw_iter }
	}

	/// Convert the iterator to a raw iterator.
	pub fn into_raw(self) -> TrieDBRawIterator<L, N> {
		self.raw_iter
	}

	/// Fetch value by hash at a current node height
	pub fn fetch_value(
		&self,
		key: &[u8],
		prefix: Prefix,
	) -> Result<DBValue, TrieHash<L, N>, CError<L, N>> {
		TrieDBRawIterator::fetch_value(self.db, key, prefix)
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	pub fn prefix(&mut self, prefix: &[u8]) -> Result<(), TrieHash<L, N>, CError<L, N>> {
		self.raw_iter.prefix(self.db, prefix)
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	pub fn prefix_then_seek(
		&mut self,
		prefix: &[u8],
		seek: &[u8],
	) -> Result<(), TrieHash<L, N>, CError<L, N>> {
		self.raw_iter.prefix_then_seek(self.db, prefix, seek)
	}

	/// Access inner hash db.
	pub fn db(&self) -> &dyn hash_db::HashDBRef<L::Hash, DBValue> {
		self.db.db()
	}
}

impl<'a, 'cache, L: TrieLayout<N>, const N: usize> TrieIterator<L, N>
	for TrieDBNodeIterator<'a, 'cache, L, N>
{
	fn seek(&mut self, key: &[u8]) -> Result<(), TrieHash<L, N>, CError<L, N>> {
		self.raw_iter.seek(self.db, key).map(|_| ())
	}
}

impl<'a, 'cache, L: TrieLayout<N>, const N: usize> Iterator
	for TrieDBNodeIterator<'a, 'cache, L, N>
{
	type Item = Result<
		(NibbleVec<N>, Option<TrieHash<L, N>>, Arc<OwnedNode<DBValue, N>>),
		TrieHash<L, N>,
		CError<L, N>,
	>;

	fn next(&mut self) -> Option<Self::Item> {
		self.raw_iter.next_raw_item(self.db).map(|result| {
			result.map(|(nibble, hash, node)| (nibble.clone(), hash.cloned(), node.clone()))
		})
	}
}
