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
	nibble::{nibble_ops, NibbleSlice, NibbleVec},
	node::{Node, NodeHandle, NodeOwned, NodePlan, OwnedNode, Value, ValuePlan},
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
			let node = crumb.node.node();

			match (crumb.status, node) {
				(Status::Entering, _) => {
					// This is only necessary due to current borrow checker's limitation.
					let crumb = self.trail.last_mut().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
					crumb.increment();
					return Some(Ok((&self.key_nibbles, crumb.hash.as_ref(), &crumb.node)))
				},
				(Status::Exiting, node) => {
					match node {
						Node::Empty | Node::Leaf { .. } => {},
						Node::Extension(partial, ..) => {
							self.key_nibbles.drop_lasts(partial.len());
						},
						Node::Branch { .. } => {
							self.key_nibbles.pop();
						},
						Node::NibbledBranch(partial, ..) => {
							self.key_nibbles.drop_lasts(partial.len() + 1);
						},
					}
					self.trail.pop().expect("we've just fetched the last element using `last_mut` so this cannot fail; qed");
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

	/// Convert the iterator to a raw iterator.
	pub fn into_raw(self) -> TrieDBRawIterator<L> {
		self.raw_iter
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

//--
// warp sync : bunch of partial and val + left hand hash child + right hand hash child
// random proof: can put hash child anywhere: need index
// p -> p -> p
//

// TODO rename Partial per key
enum ProofOp {
	Partial,			 // slice next, with size as number of nibbles
	Value,         // value next
	DropPartial,   // followed by depth
	ChildHash,     /* index and hash next TODO note that u8 is needed due to possible missing
	                * nibble which is something only for more than binary and allow
	                * value in the middle. */
}

impl ProofOp {
	fn as_u8(&self) -> u8 {
		match self {
			ProofOp::Partial => 0,
			ProofOp::Value => 1,
			ProofOp::DropPartial => 2,
			ProofOp::ChildHash => 3,
		}
	}
	fn from_u8(encoded: u8) -> Option<Self> {
		Some(match encoded {
			0 => ProofOp::Partial,
			1 => ProofOp::Value,
			2 => ProofOp::DropPartial,
			3 => ProofOp::ChildHash,
			_ => return None,
		})
	}
}

// TODO test combine prev_height align and prefix align and height align
// 2^3 -> 8 bad.
// TODO generic function with start and end align over slice, start mask
// and end mask.
fn put_key<L: TrieLayout>(
	size: usize,
	prev_height: usize,
	prefix: &NibbleVec,
	mut partial: NibbleSlice,
	output: &mut impl std::io::Write,
) -> Result<(), TrieHash<L>, CError<L>> {
	if prev_height < prefix.len() {
		let pref_len = prefix.len() - prev_height;
		let end_aligned = (pref_len % nibble_ops::NIBBLE_PER_BYTE) == 0;
		if (prev_height % nibble_ops::NIBBLE_PER_BYTE) == 0 {
			let off = if end_aligned { 0 } else { 1 };
			output
				.write(
					&prefix.inner()
						[prev_height / nibble_ops::NIBBLE_PER_BYTE..prefix.inner().len() - off],
				)
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			if !end_aligned {
				let mut last = prefix.inner()[prefix.inner().len() - 1];
				if partial.len() > 0 {
					let b = partial.at(0);
					partial.advance(1);
					last &= 0xf0;
					last |= b;
				}
				output
					.write(&[last])
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			}
		} else {
			let nb_byte = pref_len / nibble_ops::NIBBLE_PER_BYTE;
			let slice = &prefix.inner()[prev_height / nibble_ops::NIBBLE_PER_BYTE..];
			let off = if !end_aligned { 0 } else { 1 };
			for i in 0..nb_byte - off {
				let mut b = slice[i] << 4;
				b |= slice[i + 1] >> 4;
				output
					.write(&[b])
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			}
			if !end_aligned {
				let mut last = slice[nb_byte - 1] << 4;
				if partial.len() > 0 {
					let b = partial.at(0);
					partial.advance(1);
					last &= 0xf0;
					last |= b;
				}
				output
					.write(&[last])
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			}
		}
	}
	if partial.len() > 0 {
		// TODO
		let (slice, offset) = partial.right_ref();
		let aligned = offset == 0;
		if aligned {
			output
				.write(slice)
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
		} else {
			let nb_byte = partial.len() / nibble_ops::NIBBLE_PER_BYTE;
			for i in 0..nb_byte - 1 {
				let mut b = slice[i] << 4;
				b |= slice[i + 1] >> 4;
				output
					.write(&[b])
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			}
			let b = slice[nb_byte - 1] << 4;
			output
				.write(&[b])
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
		}
	}
	Ok(())
}

fn put_value<L: TrieLayout>(
	value: &[u8],
	output: &mut impl std::io::Write,
) -> Result<(), TrieHash<L>, CError<L>> {
	let op = ProofOp::Value;
	output
		.write(&[op.as_u8()])
		// TODO right error (when doing no_std writer / reader
		.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
	VarInt(value.len() as u32)
		.encode_into(output)
		// TODO right error (when doing no_std writer / reader
		.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

	output
		.write(value)
		.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
	Ok(())
}

// TODO chunk it TODO Write like trait out of std.
fn full_state<'a, 'cache, L: TrieLayout>(
	mut iter: TrieDBNodeIterator<'a, 'cache, L>,
	output: &mut impl std::io::Write,
) -> Result<(), TrieHash<L>, CError<L>> {
	let mut prev_height: usize = 0;
	while let Some(n) = iter.next() {
		let (mut prefix, o_hash, node) = n?;
		match node.node_plan() {
			NodePlan::Empty => {},
			NodePlan::Leaf { partial, value } => {
				let height = prefix.len() + partial.len();
				debug_assert!(height > prev_height);
				let size = height - prev_height;
				// TODO plug scale encode?
				VarInt(size as u32)
					.encode_into(output)
					// TODO right error (when doing no_std writer / reader
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
				let aligned = (size % nibble_ops::NIBBLE_PER_BYTE) == 0;
				let op = ProofOp::Partial;
				output
					.write(&[op.as_u8()])
					// TODO right error (when doing no_std writer / reader
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

				let partial = partial.build(node.data());
				put_key::<L>(size, prev_height, &prefix, partial, output)?;
				prev_height += size;

				let value = value.build(node.data(), node.locations());
				match value {
					Value::Inline(value) => {
						put_value::<L>(value, output)?;
					},
					Value::Node(hash, location) => {
						prefix.append_partial(partial.right());
						let (key_slice, maybe_extra_nibble) = prefix.as_prefix();
						if let Some(extra_nibble) = maybe_extra_nibble {
							return Err(Box::new(TrieError::ValueAtIncompleteKey(
								key_slice.to_vec(),
								extra_nibble,
							)))
						}

						match TrieDBRawIterator::fetch_value(
							iter.db,
							&hash,
							(key_slice, None),
							location[0],
						) {
							Ok(value) => {
								put_value::<L>(value.as_slice(), output)?;
							},
							Err(err) => return Err(err),
						};
					},
				}
			},
			NodePlan::NibbledBranch { partial, children, value } => {
				let height = prefix.len() + partial.len();

				// TODO partial if value present only

				// TODO when restart you just seek: so iterate on crumb to push all children befor
				// index. TODO when exiting return all children after index
			},
			NodePlan::Extension { .. } => unimplemented!(),
			NodePlan::Branch { .. } => unimplemented!(),
		}
	}
	Ok(())
}

// TODO chunk it TODO Write like trait out of std.
pub fn full_state2<'a, 'cache, L: TrieLayout>(
	mut iter: crate::triedb::TrieDBIterator<'a, 'cache, L>,
	output: &mut impl std::io::Write,
) -> Result<(), TrieHash<L>, CError<L>> {
	let mut prev_height: usize = 0;
	// TODO at start key do a seek then iterate on crumb and add key portion plus all children hash
	// along the branches before ix and value hash or value.

	// TODO when exiting on limit: pop all crumb and add siblings hash after index

	let mut prev_key = Vec::new();
	let mut prev_key_len = 0;
	while let Some(n) = iter.next() {
		let (key, value) = n?;
		let key_len = key.len() * nibble_ops::NIBBLE_PER_BYTE;
		let common_depth = nibble_ops::biggest_depth(&prev_key[..], &key[..]);

		if common_depth < prev_key_len {
			let op = ProofOp::DropPartial;
			output
				.write(&[op.as_u8()])
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

			let nb = VarInt((prev_key_len - common_depth) as u32).encode_into(output);
		}
		debug_assert!(common_depth < key_len);
		if common_depth < key_len {
			let to_write = key_len - common_depth;
			let aligned = to_write % nibble_ops::NIBBLE_PER_BYTE == 0;
			let op = ProofOp::Partial;
			output
				.write(&[op.as_u8()])
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			VarInt(to_write as u32)
				.encode_into(output)
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			let start_aligned = common_depth % nibble_ops::NIBBLE_PER_BYTE == 0;
			let start_ix = common_depth / nibble_ops::NIBBLE_PER_BYTE;
			if start_aligned {
				let off = if aligned { 0 } else { 1 };
				output.write(&key[start_ix..key.len() - off]);
				if !aligned {
					output.write(&[nibble_ops::pad_left(key[key.len() - 1])]);
				}
			} else {
				for i in start_ix..key.len() - 1 {
					let mut b = key[i] << 4;
					b |= key[i + 1] >> 4;
					output.write(&[b]);
				}
				if !aligned {
					let b = key[key.len() - 1] << 4;
					output.write(&[b]);
				}
			}
		}
		put_value::<L>(value.as_slice(), output)?;

		prev_key = key;
		prev_key_len = key_len;
	}
	Ok(())
}


pub fn build_from_proof<'a, 'cache, L: TrieLayout>(
	input: &mut impl std::io::Read,
	db: &mut memory_db::MemoryDB<L::Hash, memory_db::PrefixedKey<L::Hash>, DBValue>,
) -> TrieHash<L> {
	unimplemented!()
}

// Limiting size to u32 (could also just use a terminal character).
#[derive(Debug, PartialEq, Eq)]
#[repr(transparent)]
struct VarInt(u32);

impl VarInt {
	fn encoded_len(&self) -> usize {
		if self.0 == 0 {
			return 1
		}
		let len = 32 - self.0.leading_zeros() as usize;
		if len % 7 == 0 {
			len / 7
		} else {
			len / 7 + 1
		}
		/*
		match self.0 {
			l if l < 2 ^ 7 => 1, // leading 0: 25
			l if l < 2 ^ 14 => 2, // leading 0: 18

			l if l < 2 ^ 21 => 3, // 11
			l if l < 2 ^ 28 => 4, // 4
			_ => 5,
		}
		*/
	}

	fn encode_into(&self, out: &mut impl std::io::Write) -> core::result::Result<(), ()> {
		let mut to_encode = self.0;
		for _ in 0..self.encoded_len() - 1 {
			out.write(&[0b1000_0000 | to_encode as u8]).map_err(|_| ())?;
			to_encode >>= 7;
		}
		out.write(&[to_encode as u8]).map_err(|_| ())?;
		Ok(())
	}

	fn decode(encoded: &[u8]) -> core::result::Result<(Self, usize), ()> {
		let mut value = 0u32;
		for (i, byte) in encoded.iter().enumerate() {
			let last = byte & 0b1000_0000 == 0;
			value |= ((byte & 0b0111_1111) as u32) << (i * 7);
			if last {
				return Ok((VarInt(value), i + 1))
			}
		}
		Err(())
	}
}

#[test]
fn varint_encode_decode() {
	let mut buf = crate::query_plan::InMemoryRecorder::default();
	for i in 0..u16::MAX as u32 + 1 {
		VarInt(i).encode_into(&mut buf);
		assert_eq!(buf.buffer.len(), VarInt(i).encoded_len());
		assert_eq!(Ok((VarInt(i), buf.buffer.len())), VarInt::decode(&buf.buffer));
		buf.buffer.clear();
	}
}
//fn state_warp
