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
	pub(crate) fn next_raw_item<O: CountedWrite>(
		&mut self,
		db: &TrieDB<L>,
		mut cb: Option<&mut &mut IterCallback<L, O>>,
	) -> Option<
		Result<
			(&NibbleVec, Option<&TrieHash<L>>, &Arc<OwnedNode<DBValue, L::Location>>),
			TrieHash<L>,
			CError<L>,
		>,
	> {
		self.next_raw_item_with_pop(db, cb, false)
	}

	pub(crate) fn next_raw_item_with_pop<O: CountedWrite>(
		&mut self,
		db: &TrieDB<L>,
		mut cb: Option<&mut &mut IterCallback<L, O>>,
		with_pop: bool,
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
					if let Some(cb) = cb.as_mut() {
						if let Err(e) = cb.on_pop(&crumb, &self.key_nibbles) {
							return Some(Err(e));
						}
					}
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
					if with_pop {
						if let Some(crumb) = self.trail.last_mut() {
							return Some(Ok((&self.key_nibbles, crumb.hash.as_ref(), &crumb.node)))
						} else {
							return None;
						}
					}
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
		self.next_inner::<NoWrite>(db, None)
	}

	/// Same as `next_item` but with callback.
	pub(crate) fn next_item_with_callback<O: CountedWrite>(
		&mut self,
		db: &TrieDB<L>,
		cb: &mut IterCallback<L, O>,
	) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		self.next_inner(db, Some(cb))
	}

	#[inline]
	fn next_inner<O: CountedWrite>(
		&mut self,
		db: &TrieDB<L>,
		mut cb: Option<&mut IterCallback<L, O>>,
	) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		while let Some(raw_item) = self.next_raw_item::<O>(db, cb.as_mut()) {
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
		while let Some(raw_item) = self.next_raw_item::<NoWrite>(db, None) {
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
		self.raw_iter.next_raw_item::<NoWrite>(self.db, None).map(|result| {
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
pub enum ProofOp {
	Partial,     // slice next, with size as number of nibbles. Attached could be size.
	Value,       // value next. Attached could be size.
	DropPartial, // followed by depth. Attached could be size.
	Hashes,      /* followed by consecutive defined hash, then bitmap of maximum 8 possibly
	              * defined hash then defined amongst them, then 8 next and repeat
	              * for possible. Attached could be bitmap.
	              * When structure allows either inline or hash, we add a bit in the bitmap
	              * to indicate if inline or single value hash: the first one */
}

pub trait ProofOpHeadCodec {
	/// return range of value that can be attached to this op.
	/// for bitmap it return bitmap len.
	fn attached_range(op: ProofOp) -> u8;

	/// op can have some data attached (depending on encoding).
	fn encode_op(op: ProofOp, attached: u8) -> u8;

	/// Return op and range attached.
	fn decode_op(encoded: u8) -> (ProofOp, u8);
}

#[derive(Default, Clone)]
// TODO const N expected len??
pub struct Bitmap1(pub u8);

impl Bitmap1 {
	pub fn check(expected_len: usize) -> bool {
		debug_assert!(expected_len > 0);
		debug_assert!(expected_len < 9);
		(0xff >> expected_len) == 0
	}

	pub fn get(&self, i: usize) -> bool {
		debug_assert!(i < 8);
		self.0 & (0b0000_0001 << i) != 0
	}

	// TODO useless??
	pub fn encode<I: Iterator<Item = bool>>(&mut self, has_children: I) {
		for (i, v) in has_children.enumerate() {
			if v {
				self.set(i);
			}
		}
	}

	pub fn set(&mut self, i: usize) {
		debug_assert!(i < 8);
		self.0 |= 0b0000_0001 << i;
	}
}

/*
pub struct BitmapAccesses<'a> {
	possible_inline_value: bool,    // TODO should be constant
	possible_inline_children: bool, // TODO should be constant
	unaccessed_value: bool,
	unaccessed_ranges: &'a [Range<usize>],
}
// TODO memoize/precalculate results for all.
impl<'a> BitmapAccesses<'a> {
	// note 1 is no map and we got None in bit index functions
	fn nb_bits(&self) -> usize {
		let mut count = self.value_offset();
		for range in self.unaccessed_ranges {
			count += range.len() * if self.possible_inline_children { 2 } else { 1 };
		}
		count
	}
	fn value_offset(&self) -> usize {
		if self.unaccessed_value {
			if self.possible_inline_value {
				2
			} else {
				1
			}
		} else {
			0
		}
	}
	fn bit_index_value(&self) -> Option<usize> {
		self.unaccessed_value.then(|| 0)
	}
	fn bit_index_value_type(&self) -> Option<usize> {
		(self.unaccessed_value && self.possible_inline_value).then(|| 1)
	}
	fn index_children(&self, ix: u8) -> Option<usize> {
		let mut found = false;
		let mut offset = 0;
		let ix = ix as usize;
		for range in self.unaccessed_ranges {
			if ix < range.start {
				break;
			}
			if ix >= range.start && ix < range.end {
				found = true;
				offset += ix - range.start;
				break;
			}
			offset += range.len();
		}
		found.then(|| offset)
	}
	fn bit_index_children(&self, ix: u8) -> Option<usize> {
		self.index_children(ix)
			.map(|i| self.value_offset() + if self.possible_inline_children { i * 2 } else { i })
	}
	fn bit_index_children_type(&self, ix: u8) -> Option<usize> {
		self.possible_inline_children
			.then(|| self.bit_index_children(ix).map(|i| i + 1))
			.flatten()
	}
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(Default)]
pub struct BitmapAccesses2 {
	value_index: Option<u8>,
	children_index: [Option<u8>; nibble_ops::NIBBLE_LENGTH],
}

pub fn get_bitmap_accesses<
	const POSSIBLE_INLINE_VALUE: bool,
	const POSSIBLE_INLINE_CHILDREN: bool,
>(
	unaccessed_value: bool,
	unaccessed_ranges: &[Range<usize>],
	value_present: bool,
	children_present: &[bool],
) -> (BitmapAccesses2, BitmapAccesses2) {
	let mut presence = BitmapAccesses2::default();
	let mut is_inline = BitmapAccesses2::default();
	let mut index: u8 = 0;
	if unaccessed_value {
		presence.value_index = Some(index);
		index += 1;
		if POSSIBLE_INLINE_VALUE {
			is_inline.value_index = Some(index);
			index += 1;
		}
	}
	for range in unaccessed_ranges {
		for i in range.start..range.end {
			presence.children_index[i] = Some(index);
			index += 1;
			if POSSIBLE_INLINE_CHILDREN {
				is_inline.children_index[i] = Some(index);
				index += 1;
			}
		}
	}
	(presence, is_inline)
}
*/

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

pub fn encode_hashes<'a, I, I2, L, O>(
	output: &mut O,
	mut iter_defined: I, // TODO remove or use
	mut iter_possible: I2,
	header_bitmap_len: usize, // TODO from trait
	header_init: fn(u8) -> u8, /* TODO from trait
	                           *	#[cfg(debug_assert)]
	                           *	hashes_indexes: (BitmapAccesses2, BitmapAccesses2), */
) -> Result<(), TrieHash<L>, CError<L>>
where
	O: CountedWrite,
	L: TrieLayout + 'a,
	I: Iterator<Item = OpHash<'a>>,
	I2: Iterator<Item = OpHash<'a>>,
{
	let mut nexts: [OpHash; 8] = [OpHash::None; 8];
	let mut header_written = false;
	let mut i_hash = 0;
	let mut i_bitmap = 0;
	let mut hash_len = 0;
	let mut bitmap_len = 0;
	// if bit in previous bitmap (presence to true and type expected next).
	let mut prev_bit: Option<OpHash> = None;
	let mut buff_bef_first = smallvec::SmallVec::<[u8; 4]>::new();

	loop {
		bitmap_len += i_bitmap;
		hash_len += i_hash;
		i_bitmap = 0;
		i_hash = 0;
		let mut bitmap = Bitmap1::default();

		let bound = if !header_written && header_bitmap_len > 0 {
			header_bitmap_len
		} else {
			if !header_written {
				header_written = true;
				let header = header_init(bitmap.0);
				buff_bef_first.push(header);
			}
			8
		};

		if let Some(h) = prev_bit.take() {
			debug_assert!(h.is_some());
			if h.is_var() {
				bitmap.set(i_bitmap);
			}
			i_bitmap += 1;
			if h.is_some() {
				nexts[i_hash] = h;
				i_hash += 1;
			}
		}

		while let Some(h) = iter_possible.next() {
			if h.is_some() {
				bitmap.set(i_bitmap);
				i_bitmap += 1;
				if i_bitmap == bound {
					prev_bit = Some(h);
					break;
				}
				if h.is_var() {
					bitmap.set(i_bitmap);
				}
			}
			i_bitmap += 1;
			if h.is_some() {
				nexts[i_hash] = h;
				i_hash += 1;
			}
			if i_bitmap == bound {
				break;
			}
		}

		if i_bitmap == 0 {
			debug_assert!(header_written && i_hash == 0);
			break
		}

		if !header_written {
			header_written = true;
			let header = header_init(bitmap.0);
			buff_bef_first.push(header);
		} else {
			buff_bef_first.push(bitmap.0);
		}
		output
			.write(&buff_bef_first)
			// TODO right error (when doing no_std writer / reader
			.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
		buff_bef_first.clear();
		for j in 0..i_hash {
			match nexts[j] {
				OpHash::Fix(s) => {
					output
						.write(s)
						// TODO right error (when doing no_std writer / reader
						.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
				},
				OpHash::Var(s) => {
					VarInt(s.len() as u32)
						.encode_into(output)
						// TODO right error (when doing no_std writer / reader
						.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
					output
						.write(s)
						// TODO right error (when doing no_std writer / reader
						.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
				},
				OpHash::None => unreachable!(),
			}
		}
	}
	Ok(())
}

impl ProofOp {
	pub fn as_u8(&self) -> u8 {
		match self {
			ProofOp::Partial => 0,
			ProofOp::Value => 1,
			ProofOp::DropPartial => 2,
			ProofOp::Hashes => 3,
		}
	}
	pub fn from_u8(encoded: u8) -> Option<Self> {
		Some(match encoded {
			0 => ProofOp::Partial,
			1 => ProofOp::Value,
			2 => ProofOp::DropPartial,
			3 => ProofOp::Hashes,
			_ => return None,
		})
	}
}

fn put_value<L: TrieLayout>(
	value: &[u8],
	output: &mut impl CountedWrite,
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

// Call on pop, passed as parameter.
pub(crate) struct IterCallback<'a, L, O> {
	output: &'a mut O,
	start_key: Option<&'a [u8]>,
	start_range_height: usize,
	first: bool,
	write_aft: bool,
	not_first_branch_aft: bool,
	aft_written: bool,
	_ph: core::marker::PhantomData<L>,
}

impl<'a, L, O> IterCallback<'a, L, O> {
	pub fn new(output: &'a mut O, start_key: Option<&'a [u8]>) -> Self {
		let start_range_height =
			start_key.map(|s| s.len() * nibble_ops::NIBBLE_PER_BYTE).unwrap_or(0);
		Self {
			output,
			start_key,
			first: true,
			start_range_height,
			aft_written: false,
			not_first_branch_aft: false,
			write_aft: false,
			_ph: Default::default(),
		}
	}
}

impl<'a, L: TrieLayout, O: CountedWrite> IterCallback<'a, L, O> {
	pub(crate) fn on_pop(
		&mut self,
		crumb: &Crumb<L::Hash, L::Location>,
		key_nibbles: &NibbleVec,
	) -> Result<(), TrieHash<L>, CError<L>> {
		if crumb.hash.is_none() {
			// inline got nothing to add
			if self.write_aft {
				self.not_first_branch_aft = true;
			}
			self.first = false;
			return Ok(());
		}

		let mut value_only = false;
		match crumb.node.node_plan() {
			NodePlan::Branch { .. } | NodePlan::NibbledBranch { .. } => (),
			NodePlan::Leaf { value, partial, .. } => {
				// note that key_nibbles do not contain partial.
				if self.first {
					if let Some(start_key) = self.start_key {
						let mut start_key_nibbles = NibbleSlice::new(start_key);
						if start_key_nibbles.starts_with_vec(key_nibbles) {
							start_key_nibbles.advance(key_nibbles.len());
							if start_key_nibbles == partial.build(crumb.node.data()) {
								value_only = true;
							}
						}
					}
				}
				if !value_only {
					return Ok(());
				}
			},
			_ => return Ok(()),
		}
		self.first = false;

		// exclusive
		let mut range_bef = None;
		// inclusive
		let mut range_aft = nibble_ops::NIBBLE_LENGTH;

		if value_only {
			self.start_range_height = key_nibbles.len();
		} else if self.write_aft && !self.not_first_branch_aft {
			// stop on a branch with value.
			range_aft = 0;
		} else {
			// on branch the key nibbles is not popped yet and contains the index of last accessed
			// node.
			let depth_current_ix = key_nibbles.len();
			let current_ix =
				(depth_current_ix > 0).then(|| key_nibbles.at(depth_current_ix - 1) as usize);

			if let Some(key) = self.start_key.as_ref() {
				if depth_current_ix > 0 && (depth_current_ix - 1) <= self.start_range_height {
					let at = key_nibbles.len() - 1;
					debug_assert!({
						let common =
							crate::nibble::nibble_ops::biggest_depth(key, key_nibbles.inner());
						// can be sup as we may have compare agains byte padded inner).
						common >= at
					});
					let key_slice = NibbleSlice::new(key);
					if at < key_slice.len() {
						range_bef = Some(key_slice.at(at) as usize);
					} else {
						// only case in range is a seek into branch that got popped, in this case
						// at == keyslice.len()
						debug_assert!(at == key_slice.len());
					}
					self.start_range_height = at - 1;
				}
			}
			// inclusive
			if self.write_aft {
				range_aft = current_ix.map(|i| i + 1).unwrap_or(nibble_ops::NIBBLE_LENGTH)
			};
		}

		if self.write_aft {
			self.not_first_branch_aft = true;
		}

		debug_assert!(range_aft >= range_bef.unwrap_or(0));
		// if key is less than start key, we attach the value hash if there is one.
		let mut value_node = None;
		if range_bef.is_some() || value_only {
			// Values before start needed.
			// Other values from exiting (pop are already part of the proof (we range over all
			// them).
			value_node = Some(match crumb.node.node_plan().value_plan() {
				Some(ValuePlan::Node(hash_range)) =>
					OpHash::Fix(&crumb.node.data()[hash_range.clone()]),
				Some(ValuePlan::Inline(inline_range)) =>
					OpHash::Var(&crumb.node.data()[inline_range.clone()]),
				None => OpHash::None,
			});
		}

		let mut i = 0;
		let write_aft = self.write_aft;
		let aft_written = &mut self.aft_written;
		let iter_possible = core::iter::from_fn(|| loop {
			// value first.
			if let Some(value_hash) = value_node.take() {
				return Some(value_hash);
			}
			if i == range_bef.unwrap_or(0) {
				i = range_aft;
			}
			if i == nibble_ops::NIBBLE_LENGTH {
				return None;
			}
			i += 1;
			match crumb.node.node_plan() {
				NodePlan::NibbledBranch { children, .. } | NodePlan::Branch { children, .. } =>
					return Some(match &children[i - 1] {
						Some(NodeHandlePlan::Hash(hash_range)) => {
							if write_aft && i > range_aft {
								*aft_written = true;
							}
							OpHash::Fix(&crumb.node.data()[hash_range.clone()])
						},
						Some(NodeHandlePlan::Inline(inline_range)) => {
							// can result in proof containing all remaining values, but
							// still returning proofs, but makes things easier at the
							// cost of this very corner case.
							if write_aft && i > range_aft {
								*aft_written = true;
							}
							OpHash::Var(&crumb.node.data()[inline_range.clone()])
						},
						None => OpHash::None,
					}),
				_ => unreachable!(),
			}
		});

		encode_hashes::<_, _, L, _>(
			self.output,
			core::iter::empty(),
			iter_possible,
			0,
			hash_header_no_bitmap,
		)?;
		Ok(())
	}
}

fn hash_header_no_bitmap(_: u8) -> u8 {
	ProofOp::Hashes.as_u8()
}

pub fn no_bitmap_hash_header(_: u8) -> u8 {
	0
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
pub fn range_proof<'a, 'cache, L: TrieLayout>(
	db: &'a TrieDB<'a, 'cache, L>,
	mut iter: TrieDBRawIterator<L>,
	output: &mut impl CountedWrite,
	exclusive_start: Option<&[u8]>,
	size_limit: Option<usize>,
) -> Result<Option<Vec<u8>>, TrieHash<L>, CError<L>> {
	let start_written = output.written();

	let mut prev_key_depth = 0; // TODO rename as written cursor depth
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
			push_partial_key::<L>(node_depth, prev_key_depth, &start[..bound], output)?;
			prev_key_depth = node_depth;
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

			encode_hashes::<_, _, L, _>(
				output,
				core::iter::empty(),
				iter_possible,
				0,
				hash_header_no_bitmap,
			)?;
		}
	}

	while let Some(n) = { iter.next_raw_item_with_pop::<NoWrite>(db, None, false) } {
		let (prefix, hash, node) = n?;

		if seek_to_value {
			seek_to_value = false;
			// skip first value
			continue;
		}

		let pref_len = prefix.len();

		if pref_len < prev_key_depth {
			// write prev pop
			let pop_from = prev_key_depth;
			prev_key_depth = pref_len - 1; // one for branch index
			let to_pop = pop_from - prev_key_depth;

			let op = ProofOp::DropPartial;
			output
				.write(&[op.as_u8()])
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

			let nb = VarInt(to_pop as u32)
				.encode_into(output)
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
		push_partial_key::<L>(key_len, prev_key_depth, &key, output)?;
		prev_key_depth = key_len;

		// TODO non vec value
		let value = match value {
			Value::Node(hash, location) =>
				match TrieDBRawIterator::<L>::fetch_value(db, &hash, (key_slice, None), location) {
					Ok(value) => value,
					Err(err) => return Err(err),
				},
			Value::Inline(value) => value.to_vec(),
		};
		put_value::<L>(value.as_slice(), output)?;
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
					let op = ProofOp::DropPartial;
					output
						.write(&[op.as_u8()])
						// TODO right error (when doing no_std writer / reader
						.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

					let nb = VarInt((depth_from - depth) as u32)
						.encode_into(output)
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

					encode_hashes::<_, _, L, _>(
						output,
						core::iter::empty(),
						iter_possible,
						0,
						hash_header_no_bitmap,
					)?;
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

fn push_partial_key<L: TrieLayout>(
	to: usize,
	from: usize,
	key: &[u8],
	output: &mut impl CountedWrite,
) -> Result<(), TrieHash<L>, CError<L>> {
	let to_write = to - from;
	let aligned = to_write % nibble_ops::NIBBLE_PER_BYTE == 0;
	let op = ProofOp::Partial;
	output
		.write(&[op.as_u8()])
		// TODO right error (when doing no_std writer / reader
		.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
	let nb = VarInt(to_write as u32)
		.encode_into(output)
		// TODO right error (when doing no_std writer / reader
		.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
	let start_aligned = from % nibble_ops::NIBBLE_PER_BYTE == 0;
	let start_ix = from / nibble_ops::NIBBLE_PER_BYTE;
	if start_aligned {
		let off = if aligned { 0 } else { 1 };
		let slice_to_write = &key[start_ix..key.len() - off];
		output.write(slice_to_write);
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
	Ok(())
}

#[cfg_attr(feature = "std", derive(Debug))]
#[derive(PartialEq, Eq)]
#[repr(transparent)]
pub struct VarInt(u32);

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

	fn encode_into(&self, out: &mut impl CountedWrite) -> core::result::Result<(), ()> {
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

	pub fn decode_from(input: &mut impl std::io::Read) -> core::result::Result<u32, ()> {
		let mut value = 0u32;
		let mut buff = [0u8];
		let mut i = 0;
		loop {
			input.read_exact(&mut buff[..]).map_err(|_| ())?;
			let byte = buff[0];
			let last = byte & 0b1000_0000 == 0;
			value |= ((byte & 0b0111_1111) as u32) << (i * 7);
			if last {
				return Ok(value);
			}
			i += 1;
		}
		Err(())
	}
}

#[test]
fn varint_encode_decode() {
	let mut buf = Vec::new();
	for i in 0..u16::MAX as u32 + 1 {
		VarInt(i).encode_into(&mut buf);
		assert_eq!(buf.len(), VarInt(i).encoded_len());
		assert_eq!(Ok((VarInt(i), buf.len())), VarInt::decode(&buf));
		buf.clear();
	}
}

pub trait CountedWrite: std::io::Write {
	// size written in write.
	// Warning depending on implementation this
	// is not starting at same size, so should
	// always be used to compare with an initial size.
	fn written(&self) -> usize;
}

pub struct Counted<T: std::io::Write> {
	pub inner: T,
	pub written: usize,
}

impl<T: std::io::Write> From<T> for Counted<T> {
	fn from(inner: T) -> Self {
		Self { inner, written: 0 }
	}
}

// TODO a specific trait. and this for std only
impl<T: std::io::Write> std::io::Write for Counted<T> {
	fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
		let written = self.inner.write(buf)?;
		self.written += written;
		Ok(written)
	}

	fn flush(&mut self) -> std::io::Result<()> {
		self.inner.flush()
	}
}

impl<T: std::io::Write> CountedWrite for Counted<T> {
	fn written(&self) -> usize {
		self.written
	}
}

impl CountedWrite for Vec<u8> {
	fn written(&self) -> usize {
		self.len()
	}
}

#[derive(Clone, Copy)]
pub struct NoWrite;

impl std::io::Write for NoWrite {
	fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
		Err(std::io::ErrorKind::Unsupported.into())
	}

	fn flush(&mut self) -> std::io::Result<()> {
		Ok(())
	}
}

impl CountedWrite for NoWrite {
	fn written(&self) -> usize {
		0
	}
}
