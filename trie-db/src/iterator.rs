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

use crate::range_proof::{CountedWrite, ProofOp, RangeProofCodec, RangeProofError};

use super::{CError, DBValue, Result, Trie, TrieHash, TrieIterator, TrieLayout};
use crate::{
	nibble::{nibble_ops, NibbleSlice, NibbleVec},
	node::{Node, NodeHandle, NodeHandlePlan, NodePlan, OwnedNode, Value, ValuePlan},
	node_db::{self, Prefix, EMPTY_PREFIX},
	triedb::TrieDB,
	TrieDoubleEndedIterator, TrieError, TrieItem, TrieKeyItem,
};

use crate::rstd::{boxed::Box, sync::Arc, vec::Vec};

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, Copy, Eq, PartialEq)]
enum Status {
	Entering,
	At,
	AtChild(usize),
	Exiting,
	AftExiting,
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Eq, PartialEq)]
struct Crumb<L: TrieLayout> {
	hash: Option<TrieHash<L>>,
	node: Arc<OwnedNode<DBValue, L::Location>>,
	status: Status,
}

impl<L: TrieLayout> Crumb<L> {
	/// Move on to the next status in the node's sequence in a direction.
	fn step(&mut self, fwd: bool) {
		self.status = match (self.status, self.node.node_plan()) {
			(Status::Entering, NodePlan::Extension { .. }) => Status::At,
			(Status::Entering, NodePlan::Branch { .. }) |
			(Status::Entering, NodePlan::NibbledBranch { .. }) => Status::At,
			(Status::At, NodePlan::Branch { .. }) |
			(Status::At, NodePlan::NibbledBranch { .. }) =>
				if fwd {
					Status::AtChild(0)
				} else {
					Status::AtChild(nibble_ops::NIBBLE_LENGTH - 1)
				},
			(Status::AtChild(x), NodePlan::Branch { .. }) |
			(Status::AtChild(x), NodePlan::NibbledBranch { .. })
				if fwd && x < (nibble_ops::NIBBLE_LENGTH - 1) =>
				Status::AtChild(x + 1),
			(Status::AtChild(x), NodePlan::Branch { .. }) |
			(Status::AtChild(x), NodePlan::NibbledBranch { .. })
				if !fwd && x > 0 =>
				Status::AtChild(x - 1),
			(Status::Exiting, _) => Status::AftExiting,
			_ => Status::Exiting,
		}
	}
}

/// Iterator for going through all nodes in the trie in pre-order traversal order.
pub struct TrieDBRawIterator<L: TrieLayout> {
	/// Forward trail of nodes to visit.
	trail: Vec<Crumb<L>>,
	/// Forward iteration key nibbles of the current node.
	key_nibbles: NibbleVec,
}

impl<L: TrieLayout> TrieDBRawIterator<L> {
	/// Create a new empty iterator.
	pub fn empty() -> Self {
		Self { trail: Vec::new(), key_nibbles: NibbleVec::new() }
	}

	/// Create a new iterator.
	pub fn new(db: &TrieDB<L>) -> Result<Self, TrieHash<L>, CError<L>> {
		let mut r =
			TrieDBRawIterator { trail: Vec::with_capacity(8), key_nibbles: NibbleVec::new() };
		let (root_node, root_hash) = db.get_raw_or_lookup(
			*db.root(),
			NodeHandle::Hash(db.root().as_ref(), db.root_location()),
			EMPTY_PREFIX,
			true,
		)?;

		r.descend(root_node, root_hash);
		Ok(r)
	}

	/// Create a new iterator, but limited to a given prefix.
	pub fn new_prefixed(db: &TrieDB<L>, prefix: &[u8]) -> Result<Self, TrieHash<L>, CError<L>> {
		let mut iter = TrieDBRawIterator::new(db)?;
		iter.prefix(db, prefix, true)?;

		Ok(iter)
	}

	/// Create a new iterator, but limited to a given prefix.
	/// It then do a seek operation from prefixed context (using `seek` lose
	/// prefix context by default).
	pub fn new_prefixed_then_seek(
		db: &TrieDB<L>,
		prefix: &[u8],
		start_at: &[u8],
	) -> Result<Self, TrieHash<L>, CError<L>> {
		let mut iter = TrieDBRawIterator::new(db)?;
		iter.prefix_then_seek(db, prefix, start_at)?;
		Ok(iter)
	}

	/// Descend into a node.
	fn descend(&mut self, node: OwnedNode<DBValue, L::Location>, node_hash: Option<TrieHash<L>>) {
		self.trail
			.push(Crumb { hash: node_hash, status: Status::Entering, node: Arc::new(node) });
	}

	/// Fetch value by hash at a current node height
	pub(crate) fn fetch_value(
		db: &TrieDB<L>,
		key: &[u8],
		prefix: Prefix,
		location: L::Location,
	) -> Result<DBValue, TrieHash<L>, CError<L>> {
		let mut res = TrieHash::<L>::default();
		res.as_mut().copy_from_slice(key);
		db.fetch_value(res, prefix, location)
	}

	/// Seek a node position at 'key' for iterator.
	/// Returns true if the cursor is at or after the key, but still shares
	/// a common prefix with the key, return false if the key do not
	/// share its prefix with the node.
	/// This indicates if there is still nodes to iterate over in the case
	/// where we limit iteration to 'key' as a prefix.
	pub(crate) fn seek(
		&mut self,
		db: &TrieDB<L>,
		key: &[u8],
		fwd: bool,
	) -> Result<bool, TrieHash<L>, CError<L>> {
		self.trail.clear();
		self.key_nibbles.clear();
		let key = NibbleSlice::new(key);

		let (mut node, mut node_hash) = db.get_raw_or_lookup(
			<TrieHash<L>>::default(),
			NodeHandle::Hash(db.root().as_ref(), Default::default()),
			EMPTY_PREFIX,
			true,
		)?;
		let mut partial = key;
		let mut full_key_nibbles = 0;
		loop {
			let (next_node, next_node_hash) = {
				self.descend(node, node_hash);
				let crumb = self.trail.last_mut().expect(
					"descend pushes a crumb onto the trail; \
						thus the trail is non-empty; qed",
				);
				let node_data = crumb.node.data();
				let locations = crumb.node.locations();

				match crumb.node.node_plan() {
					NodePlan::Leaf { partial: partial_plan, .. } => {
						let slice = partial_plan.build(node_data);
						if (fwd && slice < partial) || (!fwd && slice > partial) {
							crumb.status = Status::Exiting;
							return Ok(false);
						}
						return Ok(slice.starts_with(&partial));
					},
					NodePlan::Extension { partial: partial_plan, child } => {
						let slice = partial_plan.build(node_data);
						if !partial.starts_with(&slice) {
							if (fwd && slice < partial) || (!fwd && slice > partial) {
								crumb.status = Status::Exiting;
								self.key_nibbles.append_partial(slice.right());
								return Ok(false);
							}
							return Ok(slice.starts_with(&partial));
						}

						full_key_nibbles += slice.len();
						partial = partial.mid(slice.len());
						crumb.status = Status::At;
						self.key_nibbles.append_partial(slice.right());

						let prefix = key.back(full_key_nibbles);
						db.get_raw_or_lookup(
							node_hash.unwrap_or_default(),
							child.build(node_data, locations.first().copied().unwrap_or_default()),
							prefix.left(),
							true,
						)?
					},
					NodePlan::Branch { value, children } => {
						if partial.is_empty() {
							return Ok(true);
						}

						let i = partial.at(0);
						crumb.status = Status::AtChild(i as usize);
						self.key_nibbles.push(i);

						if children[i as usize].is_some() {
							// TODO would make sense to put location in NodePlan: this is rather
							// costy
							let (_, children) = NodePlan::build_value_and_children(
								value.as_ref(),
								children,
								node_data,
								locations,
							);

							full_key_nibbles += 1;
							partial = partial.mid(1);

							let prefix = key.back(full_key_nibbles);
							db.get_raw_or_lookup(
								node_hash.unwrap_or_default(),
								children[i as usize].unwrap(),
								prefix.left(),
								true,
							)?
						} else {
							return Ok(false);
						}
					},
					NodePlan::NibbledBranch { partial: partial_plan, value, children } => {
						let slice = partial_plan.build(node_data);
						if !partial.starts_with(&slice) {
							if (fwd && slice < partial) || (!fwd && slice > partial) {
								crumb.status = Status::Exiting;
								self.key_nibbles.append_partial(slice.right());
								self.key_nibbles.push((nibble_ops::NIBBLE_LENGTH - 1) as u8);
								return Ok(false);
							}
							return Ok(slice.starts_with(&partial));
						}

						full_key_nibbles += slice.len();
						partial = partial.mid(slice.len());

						if partial.is_empty() {
							return Ok(true);
						}

						let i = partial.at(0);
						crumb.status = Status::AtChild(i as usize);
						self.key_nibbles.append_partial(slice.right());
						self.key_nibbles.push(i);

						if children[i as usize].is_some() {
							// TODO would make sense to put location in NodePlan: this is rather
							// costy
							let (_, children) = NodePlan::build_value_and_children(
								value.as_ref(),
								children,
								node_data,
								locations,
							);

							full_key_nibbles += 1;
							partial = partial.mid(1);

							let prefix = key.back(full_key_nibbles);
							db.get_raw_or_lookup(
								node_hash.unwrap_or_default(),
								children[i as usize].unwrap(),
								prefix.left(),
								true,
							)?
						} else {
							return Ok(false);
						}
					},
					NodePlan::Empty => {
						if !partial.is_empty() {
							crumb.status = Status::Exiting;
							return Ok(false);
						}
						return Ok(true);
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
		db: &TrieDB<L>,
		prefix: &[u8],
		fwd: bool,
	) -> Result<(), TrieHash<L>, CError<L>> {
		if self.seek(db, prefix, fwd)? {
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
		db: &TrieDB<L>,
		prefix: &[u8],
		seek: &[u8],
	) -> Result<(), TrieHash<L>, CError<L>> {
		if prefix.is_empty() {
			// There's no prefix, so just seek.
			return self.seek(db, seek, true).map(|_| ());
		}

		if seek.is_empty() || seek <= prefix {
			// Either we're not supposed to seek anywhere,
			// or we're supposed to seek *before* the prefix,
			// so just directly go to the prefix.
			return self.prefix(db, prefix, true);
		}

		if !seek.starts_with(prefix) {
			// We're supposed to seek *after* the prefix,
			// so just return an empty iterator.
			self.trail.clear();
			return Ok(());
		}

		if !self.seek(db, prefix, true)? {
			// The database doesn't have a key with such a prefix.
			self.trail.clear();
			return Ok(());
		}

		// Now seek forward again.
		self.seek(db, seek, true)?;

		let prefix_len = prefix.len() * crate::nibble::nibble_ops::NIBBLE_PER_BYTE;
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
				return Ok(());
			}
		}

		self.trail.clear();
		Ok(())
	}

	/// Fetches the next raw item.
	//
	/// Must be called with the same `db` as when the iterator was created.
	///
	/// Specify `fwd` to indicate the direction of the iteration (`true` for forward).
	pub(crate) fn next_raw_item(
		&mut self,
		db: &TrieDB<L>,
		fwd: bool,
	) -> Option<
		Result<
			(&NibbleVec, Option<&TrieHash<L>>, &Arc<OwnedNode<DBValue, L::Location>>),
			TrieHash<L>,
			CError<L>,
		>,
	> {
		loop {
			let crumb = self.trail.last_mut()?;
			let node_data = crumb.node.data();
			let locations = crumb.node.locations();

			match (crumb.status, crumb.node.node_plan()) {
				(Status::Entering, _) =>
					if fwd {
						let crumb = self.trail.last_mut().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
						crumb.step(fwd);
						return Some(Ok((&self.key_nibbles, crumb.hash.as_ref(), &crumb.node)));
					} else {
						crumb.step(fwd);
					},
				(Status::AftExiting, _) => {
					self.trail.pop().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
					self.trail.last_mut()?.step(fwd);
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
					self.trail.last_mut()?.step(fwd);
					if !fwd {
						let crumb = self.trail.last_mut().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
						return Some(Ok((&self.key_nibbles, crumb.hash.as_ref(), &crumb.node)));
					}
				},
				(Status::At, NodePlan::Extension { partial: partial_plan, child }) => {
					let partial = partial_plan.build(node_data);
					self.key_nibbles.append_partial(partial.right());

					match db.get_raw_or_lookup(
						crumb.hash.unwrap_or_default(),
						child.build(node_data, locations.first().copied().unwrap_or_default()),
						self.key_nibbles.as_prefix(),
						true,
					) {
						Ok((node, node_hash)) => {
							self.descend(node, node_hash);
						},
						Err(err) => {
							crumb.step(fwd);
							return Some(Err(err));
						},
					}
				},
				(Status::At, NodePlan::Branch { .. }) => {
					self.key_nibbles.push(if fwd {
						0
					} else {
						(nibble_ops::NIBBLE_LENGTH - 1) as u8
					});
					crumb.step(fwd);
				},
				(Status::At, NodePlan::NibbledBranch { partial: partial_plan, .. }) => {
					let partial = partial_plan.build(node_data);
					self.key_nibbles.append_partial(partial.right());
					self.key_nibbles.push(if fwd {
						0
					} else {
						(nibble_ops::NIBBLE_LENGTH - 1) as u8
					});
					crumb.step(fwd);
				},
				(Status::AtChild(i), NodePlan::Branch { value, children, .. }) |
				(Status::AtChild(i), NodePlan::NibbledBranch { value, children, .. }) => {
					if children[i].is_some() {
						// TODO would make sense to put location in NodePlan: this is rather costy
						let (_, children) = NodePlan::build_value_and_children(
							value.as_ref(),
							children,
							node_data,
							locations,
						);

						self.key_nibbles.pop();
						self.key_nibbles.push(i as u8);

						match db.get_raw_or_lookup(
							crumb.hash.unwrap_or_default(),
							children[i].unwrap(),
							self.key_nibbles.as_prefix(),
							true,
						) {
							Ok((node, node_hash)) => {
								self.descend(node, node_hash);
							},
							Err(err) => {
								crumb.step(fwd);
								return Some(Err(err));
							},
						}
					} else {
						crumb.step(fwd);
					}
				},
				_ => panic!(
					"Crumb::step and TrieDBNodeIterator are implemented so that \
						the above arms are the only possible states"
				),
			}
		}
	}

	/// Fetches the next trie item.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn next_item(&mut self, db: &TrieDB<L>) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		while let Some(raw_item) = self.next_raw_item(db, true) {
			let (prefix, _, node) = match raw_item {
				Ok(raw_item) => raw_item,
				Err(err) => return Some(Err(err)),
			};

			match Self::value_from_raw(prefix, node, db) {
				Some(r) => return Some(r),
				None => continue,
			}
		}
		None
	}

	/// Fetches the previous trie item.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn prev_item(&mut self, db: &TrieDB<L>) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		while let Some(raw_item) = self.next_raw_item(db, false) {
			let (prefix, _, node) = match raw_item {
				Ok(raw_item) => raw_item,
				Err(err) => return Some(Err(err)),
			};

			match Self::value_from_raw(prefix, node, db) {
				Some(r) => return Some(r),
				None => continue,
			}
		}
		None
	}

	pub(crate) fn value_from_raw(
		prefix: &NibbleVec,
		node: &Arc<OwnedNode<DBValue, L::Location>>,
		db: &TrieDB<L>,
	) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		let mut prefix = prefix.clone();
		let value = match node.node() {
			Node::Leaf(partial, value) => {
				prefix.append_partial(partial.right());
				value
			},
			Node::Branch(_, value) => match value {
				Some(value) => value,
				None => return None,
			},
			Node::NibbledBranch(partial, _, value) => {
				prefix.append_partial(partial.right());
				match value {
					Some(value) => value,
					None => return None,
				}
			},
			_ => return None,
		};

		let (key_slice, maybe_extra_nibble) = prefix.as_prefix();
		let key = key_slice.to_vec();
		if let Some(extra_nibble) = maybe_extra_nibble {
			return Some(Err(Box::new(TrieError::ValueAtIncompleteKey(key, extra_nibble))))
		}

		let value = match value {
			Value::Node(hash, location) =>
				match Self::fetch_value(db, &hash, (key_slice, None), location) {
					Ok(value) => value,
					Err(err) => return Some(Err(err)),
				},
			Value::Inline(value) => value.to_vec(),
		};

		return Some(Ok((key, value)))
	}

	/// Fetches the next key.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn next_key(&mut self, db: &TrieDB<L>) -> Option<TrieKeyItem<TrieHash<L>, CError<L>>> {
		while let Some(raw_item) = self.next_raw_item(db, true) {
			let (key, maybe_extra_nibble, _) = match Self::extract_key_from_raw_item(raw_item) {
				Some(Ok(k)) => k,
				Some(Err(err)) => return Some(Err(err)),
				None => continue,
			};

			if let Some(extra_nibble) = maybe_extra_nibble {
				return Some(Err(Box::new(TrieError::ValueAtIncompleteKey(key, extra_nibble))));
			}

			return Some(Ok(key));
		}
		None
	}

	/// Fetches the previous key.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn prev_key(&mut self, db: &TrieDB<L>) -> Option<TrieKeyItem<TrieHash<L>, CError<L>>> {
		while let Some(raw_item) = self.next_raw_item(db, false) {
			let (key, maybe_extra_nibble, _) = match Self::extract_key_from_raw_item(raw_item) {
				Some(Ok(k)) => k,
				Some(Err(err)) => return Some(Err(err)),
				None => continue,
			};

			if let Some(extra_nibble) = maybe_extra_nibble {
				return Some(Err(Box::new(TrieError::ValueAtIncompleteKey(key, extra_nibble))));
			}

			return Some(Ok(key));
		}
		None
	}

	/// Extracts the key from the result of a raw item retrieval.
	///
	/// Given a raw item, it extracts the key information, including the key bytes, an optional
	/// extra nibble (prefix padding), and the node value.
	fn extract_key_from_raw_item<'a>(
		raw_item: Result<
			(&NibbleVec, Option<&TrieHash<L>>, &'a Arc<OwnedNode<DBValue, L::Location>>),
			TrieHash<L>,
			CError<L>,
		>,
	) -> Option<Result<(Vec<u8>, Option<u8>, Value<'a, L::Location>), TrieHash<L>, CError<L>>> {
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
				None => return None,
			},
			Node::NibbledBranch(partial, _, value) => {
				prefix.append_partial(partial.right());
				match value {
					Some(value) => value,
					None => return None,
				}
			},
			_ => return None,
		};

		let (key_slice, maybe_extra_nibble) = prefix.as_prefix();

		Some(Ok((key_slice.to_vec(), maybe_extra_nibble, value)))
	}
}

/// Iterator for going through all nodes in the trie in pre-order traversal order.
///
/// You can reduce the number of iterations and simultaneously iterate in both directions with two
/// cursors by using `TrieDBNodeDoubleEndedIterator`. You can convert this iterator into a double
/// ended iterator with `into_double_ended_iter`.
pub struct TrieDBNodeIterator<'a, 'cache, L: TrieLayout> {
	db: &'a TrieDB<'a, 'cache, L>,
	raw_iter: TrieDBRawIterator<L>,
}

impl<'a, 'cache, L: TrieLayout> TrieDBNodeIterator<'a, 'cache, L> {
	/// Create a new iterator.
	pub fn new(db: &'a TrieDB<'a, 'cache, L>) -> Result<Self, TrieHash<L>, CError<L>> {
		Ok(Self { raw_iter: TrieDBRawIterator::new(db)?, db })
	}

	/// Restore an iterator from a raw iterator.
	pub fn from_raw(db: &'a TrieDB<'a, 'cache, L>, raw_iter: TrieDBRawIterator<L>) -> Self {
		Self { db, raw_iter }
	}

	/// Convert the iterator to a raw iterator and db.
	pub fn into_raw(self) -> (TrieDBRawIterator<L>, &'a TrieDB<'a, 'cache, L>) {
		(self.raw_iter, self.db)
	}

	/// Fetch value by hash at a current node height
	pub fn fetch_value(
		&self,
		key: &[u8],
		prefix: Prefix,
		location: L::Location,
	) -> Result<DBValue, TrieHash<L>, CError<L>> {
		TrieDBRawIterator::fetch_value(self.db, key, prefix, location)
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	pub fn prefix(&mut self, prefix: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.prefix(self.db, prefix, true)
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	pub fn prefix_then_seek(
		&mut self,
		prefix: &[u8],
		seek: &[u8],
	) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.prefix_then_seek(self.db, prefix, seek)
	}

	/// Access inner hash db.
	pub fn db(&self) -> &dyn node_db::NodeDB<L::Hash, DBValue, L::Location> {
		self.db.db()
	}

	/// Access value of an item.
	pub fn item_from_raw(
		&self,
		item: &(NibbleVec, Option<TrieHash<L>>, Arc<OwnedNode<DBValue, L::Location>>),
	) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		TrieDBRawIterator::value_from_raw(&item.0, &item.2, self.db)
	}
}

impl<'a, 'cache, L: TrieLayout> TrieIterator<L> for TrieDBNodeIterator<'a, 'cache, L> {
	fn seek(&mut self, key: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.seek(self.db, key, true).map(|_| ())
	}
}

impl<'a, 'cache, L: TrieLayout> Iterator for TrieDBNodeIterator<'a, 'cache, L> {
	type Item = Result<
		(NibbleVec, Option<TrieHash<L>>, Arc<OwnedNode<DBValue, L::Location>>),
		TrieHash<L>,
		CError<L>,
	>;

	fn next(&mut self) -> Option<Self::Item> {
		self.raw_iter.next_raw_item(self.db, true).map(|result| {
			result.map(|(nibble, hash, node)| (nibble.clone(), hash.cloned(), node.clone()))
		})
	}
}

/// Double ended iterator for going through all nodes in the trie in pre-order traversal order.
pub struct TrieDBNodeDoubleEndedIterator<'a, 'cache, L: TrieLayout> {
	db: &'a TrieDB<'a, 'cache, L>,
	raw_iter: TrieDBRawIterator<L>,
	back_raw_iter: TrieDBRawIterator<L>,
}

impl<'a, 'cache, L: TrieLayout> TrieDBNodeDoubleEndedIterator<'a, 'cache, L> {
	/// Create a new double ended iterator.
	pub fn new(db: &'a TrieDB<'a, 'cache, L>) -> Result<Self, TrieHash<L>, CError<L>> {
		Ok(Self {
			db,
			raw_iter: TrieDBRawIterator::new(db)?,
			back_raw_iter: TrieDBRawIterator::new(db)?,
		})
	}

	/// Restore an iterator from a raw iterators.
	pub fn from_raw(
		db: &'a TrieDB<'a, 'cache, L>,
		raw_iter: TrieDBRawIterator<L>,
		back_raw_iter: TrieDBRawIterator<L>,
	) -> Self {
		Self { db, raw_iter, back_raw_iter }
	}

	/// Convert the iterator to a raw forward iterator.
	pub fn into_raw(self) -> TrieDBRawIterator<L> {
		self.raw_iter
	}

	/// Convert the iterator to a raw backward iterator.
	pub fn into_raw_back(self) -> TrieDBRawIterator<L> {
		self.back_raw_iter
	}

	/// Fetch value by hash at a current node height
	pub fn fetch_value(
		&self,
		key: &[u8],
		prefix: Prefix,
		location: L::Location,
	) -> Result<DBValue, TrieHash<L>, CError<L>> {
		TrieDBRawIterator::fetch_value(self.db, key, prefix, location)
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	pub fn prefix(&mut self, prefix: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.prefix(self.db, prefix, true)?;
		self.back_raw_iter.prefix(self.db, prefix, false)
	}

	/// Advance the iterator into a prefix, no value out of the prefix will be accessed
	/// or returned after this operation.
	pub fn prefix_then_seek(
		&mut self,
		prefix: &[u8],
		seek: &[u8],
	) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.prefix_then_seek(self.db, prefix, seek)?;
		self.back_raw_iter.prefix_then_seek(self.db, prefix, seek)
	}

	/// Access inner hash db.
	pub fn db(&self) -> &dyn node_db::NodeDB<L::Hash, DBValue, L::Location> {
		self.db.db()
	}
}

impl<L: TrieLayout> TrieDoubleEndedIterator<L> for TrieDBNodeDoubleEndedIterator<'_, '_, L> {}

impl<'a, 'cache, L: TrieLayout> TrieIterator<L> for TrieDBNodeDoubleEndedIterator<'a, 'cache, L> {
	fn seek(&mut self, key: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.seek(self.db, key, true).map(|_| ())?;
		self.back_raw_iter.seek(self.db, key, false).map(|_| ())
	}
}

impl<'a, 'cache, L: TrieLayout> Iterator for TrieDBNodeDoubleEndedIterator<'a, 'cache, L> {
	type Item = Result<
		(NibbleVec, Option<TrieHash<L>>, Arc<OwnedNode<DBValue, L::Location>>),
		TrieHash<L>,
		CError<L>,
	>;

	fn next(&mut self) -> Option<Self::Item> {
		self.raw_iter.next_raw_item(self.db, true).map(|result| {
			result.map(|(nibble, hash, node)| (nibble.clone(), hash.cloned(), node.clone()))
		})
	}
}

impl<'a, 'cache, L: TrieLayout> DoubleEndedIterator
	for TrieDBNodeDoubleEndedIterator<'a, 'cache, L>
{
	fn next_back(&mut self) -> Option<Self::Item> {
		self.back_raw_iter.next_raw_item(self.db, false).map(|result| {
			result.map(|(nibble, hash, node)| (nibble.clone(), hash.cloned(), node.clone()))
		})
	}
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Clone, Copy)]
pub enum OpHash<'a> {
	Fix(&'a [u8]),
	Var(&'a [u8]),
	None,
}

impl<'a> OpHash<'a> {
	pub fn is_some(&self) -> bool {
		!matches!(self, OpHash::None)
	}
	pub fn is_var(&self) -> bool {
		matches!(self, OpHash::Var(..))
	}
}

/// `exclusive_start` is the last returned key from a previous proof.
/// Proof will there for contains seek information for this key. The proof
/// itself may contain value that where already returned by previous proof.
/// `size_limit` is a minimal limit, after being reach
/// child sibling will be added (up to NB_CHILDREN - 1 * stack node depth and stack node depth drop
/// key info). Also limit is only applyied after a first new value is written.
/// Inline value contain in the proof are also added as they got no additional
/// size cost.
///
/// Return `None` if the proof reach the end of the state, or the key to the last value in proof
/// (next range proof should restart from this key).
pub fn range_proof<'a, 'cache, L: TrieLayout, C: RangeProofCodec>(
	db: &'a TrieDB<'a, 'cache, L>,
	mut iter: TrieDBRawIterator<L>,
	output: &mut impl CountedWrite,
	exclusive_start: Option<&[u8]>,
	size_limit: Option<usize>,
) -> Result<Option<Vec<u8>>, TrieHash<L>, CError<L>> {
	let start_written = output.written();

	// current depth encoded.
	let mut cursor = 0;
	let mut seek_to_value = false;
	if let Some(start) = exclusive_start {
		let _ = iter.seek(db, start, true)?;
		let trail_key = NibbleSlice::new(start);
		let mut node_depth = 0;
		for Crumb { node, .. } in iter.trail.iter() {
			let mut is_branch = false;
			let (range_bef, value) = match node.node_plan() {
				NodePlan::Leaf { partial, value, .. } => {
					node_depth += partial.len();
					(0, Some(value))
				},
				NodePlan::Branch { value, .. } => {
					is_branch = true;
					(trail_key.at(node_depth), value.as_ref())
				},
				NodePlan::NibbledBranch { partial, value, .. } => {
					is_branch = true;
					node_depth += partial.len();
					(
						if node_depth >= trail_key.len() {
							// seek dest: nothing already read, no hash
							0
						} else {
							trail_key.at(node_depth)
						},
						value.as_ref(),
					)
				},
				_ => unimplemented!(),
			};

			if node_depth > trail_key.len() {
				return Err(RangeProofError::ShouldSuspendOnValue.into());
			} else if node_depth == trail_key.len() {
				if value.is_none() {
					return Err(RangeProofError::ShouldSuspendOnValue.into());
				}
				seek_to_value = true;
			}

			let mut has_hash = false;
			if value.is_some() {
				has_hash = true;
			}

			if !has_hash {
				match node.node_plan() {
					NodePlan::Branch { children, .. } |
					NodePlan::NibbledBranch { children, .. } =>
						for i in 0..range_bef {
							if children[i as usize].is_some() {
								has_hash = true;
								break;
							}
						},
					_ => (),
				}
			}
			if !has_hash {
				if is_branch {
					node_depth += 1;
				}
				continue;
			}

			let bound = node_depth / nibble_ops::NIBBLE_PER_BYTE +
				if (node_depth % nibble_ops::NIBBLE_PER_BYTE) == 0 { 0 } else { 1 };
			C::push_partial_key(node_depth, cursor, &start[..bound], output)?;
			cursor = node_depth;
			if is_branch {
				node_depth += 1;
			}

			let data = node.data();
			let mut value_node = Some(match value {
				Some(ValuePlan::Node(hash_range)) => OpHash::Fix(&data[hash_range.clone()]),
				Some(ValuePlan::Inline(inline_range)) => OpHash::Var(&data[inline_range.clone()]),
				None => OpHash::None,
			});

			let mut i = 0;
			let iter_possible = core::iter::from_fn(|| loop {
				// value first.
				// TODO should be a known iter but we don't know if inline -> should avoid first bit
				// of is present at least.
				if let Some(value_hash) = value_node.take() {
					return Some(value_hash);
				}
				if i == range_bef {
					return None;
				}
				i += 1;
				match node.node_plan() {
					NodePlan::NibbledBranch { children, .. } |
					NodePlan::Branch { children, .. } =>
						return Some(match &children[i as usize - 1] {
							Some(NodeHandlePlan::Hash(hash_range)) =>
								OpHash::Fix(&data[hash_range.clone()]),
							Some(NodeHandlePlan::Inline(inline_range)) =>
								OpHash::Var(&data[inline_range.clone()]),
							None => OpHash::None,
						}),
					_ => unreachable!(),
				}
			});

			C::push_hashes(output, iter_possible)?;
		}
	}

	while let Some(n) = { iter.next_raw_item(db, true) } {
		let (prefix, hash, node) = n?;

		if seek_to_value {
			seek_to_value = false;
			// skip first value
			continue;
		}

		let pref_len = prefix.len();

		if pref_len < cursor {
			// write prev pop
			let pop_from = cursor;
			cursor = pref_len - 1; // one for branch index
			let to_pop = pop_from - cursor;

			C::encode_with_size(ProofOp::DropPartial, to_pop, output)?;

			//continue;
		}

		let mut prefix = prefix.clone();
		let is_inline = hash.is_none();
		let node_data = node.data();
		let value = match node.node_plan() {
			NodePlan::Leaf { partial, value } => {
				prefix.append_partial(partial.build(node_data).right());
				Some(value.clone())
			},
			NodePlan::Branch { value, .. } => value.clone(),
			NodePlan::NibbledBranch { partial, value, .. } => {
				prefix.append_partial(partial.build(node_data).right());
				value.clone()
			},
			_ => None,
		};

		let key_len = prefix.len();
		let (key_slice, maybe_extra_nibble) = prefix.as_prefix();

		let Some(value) = value else {
			continue;
		};

		// a push

		if let Some(extra_nibble) = maybe_extra_nibble {
			return Err(Box::new(TrieError::ValueAtIncompleteKey(key_slice.to_vec(), extra_nibble)))
		}

		C::push_partial_key(key_len, cursor, key_slice, output)?;
		cursor = key_len;

		let locations = node.locations();
		match value.build(node_data, locations.first().copied().unwrap_or_default()) {
			Value::Node(hash, location) =>
				match TrieDBRawIterator::<L>::fetch_value(db, &hash, (key_slice, None), location) {
					Ok(value) => {
						C::push_value(value.as_slice(), output)?;
					},
					Err(err) => return Err(err),
				},
			Value::Inline(value) => {
				C::push_value(value, output)?;
			},
		}
		if !is_inline &&
			size_limit.map(|l| (output.written() - start_written) >= l).unwrap_or(false)
		{
			let mut depth_from = key_len;
			let mut depth = key_len;
			let mut first = true;
			let mut one_hash_written = false;
			while let Some(Crumb { node, .. }) = iter.trail.pop() {
				let range_aft = if first {
					first = false;
					// key we stop at, if branch then add all children
					0
				} else {
					depth -= 1; // branch index
					prefix.at(depth) + 1
				};
				let mut has_hash = false;
				match node.node_plan() {
					NodePlan::Branch { children, .. } |
					NodePlan::NibbledBranch { children, .. } =>
						for i in (range_aft as usize)..nibble_ops::NIBBLE_LENGTH {
							if children[i].is_some() {
								has_hash = true;
								break;
							}
						},
					_ => (),
				}
				one_hash_written |= has_hash;
				if has_hash && depth < depth_from {
					C::encode_with_size(ProofOp::DropPartial, depth_from - depth, output)?;
					depth_from = depth;
				}
				if has_hash {
					let data = node.data();
					let mut i = range_aft as usize;
					let iter_possible = core::iter::from_fn(|| loop {
						if i == nibble_ops::NIBBLE_LENGTH {
							return None;
						}
						i += 1;
						match node.node_plan() {
							NodePlan::NibbledBranch { children, .. } |
							NodePlan::Branch { children, .. } =>
								return Some(match &children[i - 1] {
									Some(NodeHandlePlan::Hash(hash_range)) =>
										OpHash::Fix(&data[hash_range.clone()]),
									Some(NodeHandlePlan::Inline(inline_range)) =>
										OpHash::Var(&data[inline_range.clone()]),
									None => OpHash::None,
								}),
							_ => unreachable!(),
						}
					});

					C::push_hashes(output, iter_possible)?;
				}
				match node.node_plan() {
					NodePlan::NibbledBranch { partial, .. } | NodePlan::Leaf { partial, .. } => {
						depth -= partial.len();
					},
					NodePlan::Branch { .. } => {},
					_ => (),
				}
			}
			if one_hash_written {
				return Ok(Some(key_slice.to_vec())); // Note this is suboptimal as we can restart at last branch
				                     // with not aft siblings, but makes things
				                     // easier (always align, simple rule)
			} else {
				return Ok(None);
			}
		}
	}

	Ok(None)
}
