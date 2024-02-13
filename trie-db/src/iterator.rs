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

use crate::range_proof::{Bitmap1, CountedWrite, NoWrite, ProofOp, RangeProofCodec, Read, Write};
use core::ops::Range;

use super::{CError, DBValue, Result, Trie, TrieHash, TrieIterator, TrieLayout};
use crate::{
	nibble::{nibble_ops, NibbleSlice, NibbleVec},
	node::{Node, NodeHandle, NodeHandlePlan, NodeOwned, NodePlan, OwnedNode, Value, ValuePlan},
	triedb::TrieDB,
	ProcessEncodedNode, TrieError, TrieItem, TrieKeyItem,
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
struct Crumb<H: Hasher, L> {
	hash: Option<H::Out>,
	node: Arc<OwnedNode<DBValue, L>>,
	status: Status,
}

impl<H: Hasher, L: Copy + Default> Crumb<H, L> {
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
				if x < (nibble_ops::NIBBLE_LENGTH - 1) =>
				Status::AtChild(x + 1),
			_ => Status::Exiting,
		}
	}
}

/// Iterator for going through all nodes in the trie in pre-order traversal order.
pub struct TrieDBRawIterator<L: TrieLayout> {
	trail: Vec<Crumb<L::Hash, L::Location>>,
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
			NodeHandle::Hash(db.root().as_ref(), Default::default()),
			EMPTY_PREFIX,
			true,
		)?;
		r.descend(root_node, root_hash);
		Ok(r)
	}

	/// Create a new iterator, but limited to a given prefix.
	pub fn new_prefixed(db: &TrieDB<L>, prefix: &[u8]) -> Result<Self, TrieHash<L>, CError<L>> {
		let mut iter = TrieDBRawIterator::new(db)?;
		iter.prefix(db, prefix)?;

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

	/// Descend into a payload.
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
					"descend_into_node pushes a crumb onto the trial; \
						thus the trail is non-empty; qed",
				);
				let node = crumb.node.node();

				match node {
					Node::Leaf(slice, _) => {
						if slice < partial {
							crumb.status = Status::Exiting;
							return Ok(false)
						}
						return Ok(slice.starts_with(&partial))
					},
					Node::Extension(slice, child) => {
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
							child,
							prefix.left(),
							true,
						)?
					},
					Node::Branch(children, _value) => {
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
								*child,
								prefix.left(),
								true,
							)?
						} else {
							return Ok(false)
						}
					},
					Node::NibbledBranch(slice, children, _value) => {
						if !partial.starts_with(&slice) {
							if slice < partial {
								crumb.status = Status::Exiting;
								self.key_nibbles.append_partial(slice.right());
								self.key_nibbles.push((nibble_ops::NIBBLE_LENGTH - 1) as u8);
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
								*child,
								prefix.left(),
								true,
							)?
						} else {
							return Ok(false)
						}
					},
					Node::Empty => {
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
	fn prefix(&mut self, db: &TrieDB<L>, prefix: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
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
		db: &TrieDB<L>,
		prefix: &[u8],
		seek: &[u8],
	) -> Result<(), TrieHash<L>, CError<L>> {
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
		db: &TrieDB<L>,
	) -> Option<
		Result<
			(&NibbleVec, Option<&TrieHash<L>>, &Arc<OwnedNode<DBValue, L::Location>>),
			TrieHash<L>,
			CError<L>,
		>,
	> {
		loop {
			let crumb = self.trail.last_mut()?;
			// TODO this call to node is costly and not needed in most case.
			let node = crumb.node.node();

			match (crumb.status, node) {
				(Status::Entering, _) => {
					// This is only necessary due to current borrow checker's limitation.
					let crumb = self.trail.last_mut().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
					crumb.increment();
					return Some(Ok((&self.key_nibbles, crumb.hash.as_ref(), &crumb.node)))
				},
				(Status::Exiting, _) => {
					let crumb = self.trail.pop().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
					match crumb.node.node_plan() {
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
					self.trail.last_mut()?.increment();
				},
				(Status::At, Node::Extension(partial, child)) => {
					self.key_nibbles.append_partial(partial.right());

					match db.get_raw_or_lookup(
						crumb.hash.unwrap_or_default(),
						child,
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
				(Status::At, Node::Branch { .. }) => {
					self.key_nibbles.push(0);
					crumb.increment();
				},
				(Status::At, Node::NibbledBranch(partial, ..)) => {
					self.key_nibbles.append_partial(partial.right());
					self.key_nibbles.push(0);
					crumb.increment();
				},
				(Status::AtChild(i), Node::Branch(children, ..)) |
				(Status::AtChild(i), Node::NibbledBranch(_, children, ..)) => {
					if let Some(child) = &children[i] {
						self.key_nibbles.pop();
						self.key_nibbles.push(i as u8);

						match db.get_raw_or_lookup(
							crumb.hash.unwrap_or_default(),
							*child,
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
	pub fn next_item(&mut self, db: &TrieDB<L>) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
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
		None
	}

	/// Fetches the next key.
	///
	/// Must be called with the same `db` as when the iterator was created.
	pub fn next_key(&mut self, db: &TrieDB<L>) -> Option<TrieKeyItem<TrieHash<L>, CError<L>>> {
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

			let (key_slice, maybe_extra_nibble) = prefix.as_prefix();
			let key = key_slice.to_vec();
			if let Some(extra_nibble) = maybe_extra_nibble {
				return Some(Err(Box::new(TrieError::ValueAtIncompleteKey(key, extra_nibble))))
			}

			return Some(Ok(key))
		}
		None
	}
}

/// Iterator for going through all nodes in the trie in pre-order traversal order.
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
		self.raw_iter.prefix(self.db, prefix)
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
	pub fn db(&self) -> &dyn hash_db::HashDB<L::Hash, DBValue, L::Location> {
		self.db.db()
	}
}

impl<'a, 'cache, L: TrieLayout> TrieIterator<L> for TrieDBNodeIterator<'a, 'cache, L> {
	fn seek(&mut self, key: &[u8]) -> Result<(), TrieHash<L>, CError<L>> {
		self.raw_iter.seek(self.db, key).map(|_| ())
	}
}

impl<'a, 'cache, L: TrieLayout> Iterator for TrieDBNodeIterator<'a, 'cache, L> {
	type Item = Result<
		(NibbleVec, Option<TrieHash<L>>, Arc<OwnedNode<DBValue, L::Location>>),
		TrieHash<L>,
		CError<L>,
	>;

	fn next(&mut self) -> Option<Self::Item> {
		self.raw_iter.next_raw_item(self.db).map(|result| {
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
	// TODO rem?
	pub fn content(&self) -> &[u8] {
		match self {
			OpHash::Fix(s) | OpHash::Var(s) => s,
			OpHash::None => &[],
		}
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
		let _ = iter.seek(db, start)?;
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
				// should be unreachable if stop at value: maybe just error in this case
				seek_to_value = false;
				break;
			} else if node_depth == trail_key.len() {
				if value.is_none() {
					// TODO proper error not starting at a value.
					return Err(Box::new(TrieError::IncompleteDatabase(Default::default())));
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

	while let Some(n) = { iter.next_raw_item(db) } {
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

			let op = ProofOp::DropPartial;

			C::encode_with_size(op, to_pop, output)
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

			//continue;
		}

		let mut prefix = prefix.clone();
		let is_inline = hash.is_none();
		// TODO node.node_plan() instead
		let value = match node.node() {
			Node::Leaf(partial, value) => {
				prefix.append_partial(partial.right());
				Some(value)
			},
			Node::Branch(_, value) => value,
			Node::NibbledBranch(partial, _, value) => {
				prefix.append_partial(partial.right());
				value
			},
			_ => None,
		};

		let key_len = prefix.len();
		let (key_slice, maybe_extra_nibble) = prefix.as_prefix();
		// TODO avoid instanciating key here
		let key = key_slice.to_vec();

		let Some(value) = value else {
			continue;
		};

		// a push

		if let Some(extra_nibble) = maybe_extra_nibble {
			return Err(Box::new(TrieError::ValueAtIncompleteKey(key, extra_nibble)))
		}

		// value push key content TODO make a function
		C::push_partial_key(key_len, cursor, &key, output)?;
		cursor = key_len;

		// TODO non vec value
		let value = match value {
			Value::Node(hash, location) =>
				match TrieDBRawIterator::<L>::fetch_value(db, &hash, (key_slice, None), location) {
					Ok(value) => value,
					Err(err) => return Err(err),
				},
			Value::Inline(value) => value.to_vec(),
		};
		C::push_value(value.as_slice(), output)?;
		if !is_inline &&
			size_limit.map(|l| (output.written() - start_written) >= l).unwrap_or(false)
		{
			let mut depth_from = key_len;
			let mut depth = key_len;
			let mut first = true;
			let mut one_hash_written = false;
			while let Some(Crumb { node, hash, .. }) = iter.trail.pop() {
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
					C::encode_with_size(ProofOp::DropPartial, depth_from - depth, output)
						// TODO right error (when doing no_std writer / reader
						.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
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
				return Ok(Some(key)); // Note this is suboptimal as we can restart at last branch with not aft
				      // siblings, but makes things easier (always align, simple rule)
			} else {
				return Ok(None);
			}
		}
	}

	Ok(None)
}
