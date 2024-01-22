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
	pub(crate) fn next_raw_item<O>(
		&mut self,
		db: &TrieDB<L>,
		cb: Option<&mut IterCallback<L, O>>,
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
		self.next_inline::<()>(db, None)
	}

	/// Same as `next_item` but with callback.
	pub fn next_item_with_callback<O>(
		&mut self,
		db: &TrieDB<L>,
		cb: IterCallback<L, O>,
	) -> Option<TrieItem<TrieHash<L>, CError<L>>> {
		self.next_inline(db, Some(cb))
	}

	#[inline]
	fn next_inline<O>(
		&mut self,
		db: &TrieDB<L>,
		mut cb: Option<IterCallback<L, O>>,
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
		while let Some(raw_item) = self.next_raw_item::<()>(db, None) {
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
		self.raw_iter.next_raw_item::<()>(self.db, None).map(|result| {
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
	              * for possible. Attached could be bitmap. */
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
pub struct Bitmap1(u8);

impl Bitmap1 {
	pub fn check(expected_len: usize) -> bool {
		debug_assert!(expected_len > 0);
		debug_assert!(expected_len < 9);
		(0xff >> expected_len) == 0
	}

	pub fn get(self, i: usize) -> bool {
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

pub fn encode_hashes<'a, I, I2, L, O>(
	output: &mut O,
	mut iter_defined: I,
	mut iter_possible: I2,
	header_bitmap_len: usize,
	header_init: fn(u8) -> u8,
) -> Result<(), TrieHash<L>, CError<L>>
where
	O: CountedWrite,
	L: TrieLayout + 'a,
	I: Iterator<Item = &'a TrieHash<L>>,
	I2: Iterator<Item = Option<&'a TrieHash<L>>>,
{
	let mut nexts: [Option<&TrieHash<L>>; 8] = [None; 8];
	let mut header_written = false;
	loop {
		let mut i = 0;
		let mut bitmap = Bitmap1::default();
		let bound = if !header_written && header_bitmap_len > 0 { header_bitmap_len } else { 8 };
		while let Some(h) = iter_possible.next() {
			if h.is_some() {
				bitmap.set(i);
			}
			nexts[i] = h;
			i += 1;
			if i == bound {
				break;
			}
		}
		if !header_written {
			header_written = true;
			let header = header_init(bitmap.0);
			output
				.write(&[header])
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			// Note this first iter is empty for trie
			while let Some(h) = iter_defined.next() {
				output
					.write(h.as_ref())
					// TODO right error (when doing no_std writer / reader
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			}
		} else {
			output
				.write(&[bitmap.0])
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
		}
		for _ in 0..i {
			if let Some(h) = nexts[i] {
				output
					.write(h.as_ref())
					// TODO right error (when doing no_std writer / reader
					.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
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
	_ph: core::marker::PhantomData<L>,
}

impl<'a, L: TrieLayout, O: CountedWrite> IterCallback<'a, L, O> {
	fn on_pop(
		&mut self,
		crumb: Crumb<L::Hash, L::Location>,
	) -> Result<bool, TrieHash<L>, CError<L>> {
		unimplemented!()
	}
}

// TODO chunk it TODO Write like trait out of std.
/// `exclusive_start` is the last returned key from a previous proof.
/// Proof will there for contains seek information for this key. The proof
/// itself may contain value that where already returned by previous proof.
/// `size_limit` is a minimal limit, after being reach
/// child sibling will be added (up to NB_CHILDREN - 1 * stack node depth and stack node depth drop
/// key info). Also limit is only applyied after a first new value is written.
/// Inline value contain in the proof are also added as they got no additional
/// size cost.
///
/// Return true if the proof reach end of trie.
pub fn range_proof<'a, 'cache, L: TrieLayout>(
	mut iter: crate::triedb::TrieDBIterator<'a, 'cache, L>,
	output: &mut impl CountedWrite,
	exclusive_start: Option<&[u8]>,
	size_limit: Option<usize>,
) -> Result<bool, TrieHash<L>, CError<L>> {
	let start_written = output.written();
	// TODO when exiting a node: write children and value hash if needed.

	if let Some(start) = exclusive_start {
		iter.seek(start)?;
	}

	let mut prev_key = Vec::new();
	let mut prev_key_len = 0;
	while let Some(n) =
		{ iter.next_with_callback(IterCallback { output, _ph: Default::default() }) }
	{
		let (key, value) = n?;
		let key_len = key.len() * nibble_ops::NIBBLE_PER_BYTE;
		// Note that this is largely suboptimal: could be rewritten to use directly node iterator,
		// but this makes code simple (no need to manage branch skipping).
		let common_depth = nibble_ops::biggest_depth(&prev_key[..], &key[..]);

		if common_depth < prev_key_len {
			let op = ProofOp::DropPartial;
			output
				.write(&[op.as_u8()])
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;

			let nb = VarInt((prev_key_len - common_depth) as u32)
				.encode_into(output)
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
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
			let nb = VarInt(to_write as u32)
				.encode_into(output)
				// TODO right error (when doing no_std writer / reader
				.map_err(|e| Box::new(TrieError::IncompleteDatabase(Default::default())))?;
			let start_aligned = common_depth % nibble_ops::NIBBLE_PER_BYTE == 0;
			let start_ix = common_depth / nibble_ops::NIBBLE_PER_BYTE;
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
		}
		put_value::<L>(value.as_slice(), output)?;
		if size_limit.map(|l| (start_written - output.written()) >= l).unwrap_or(false) {
			unimplemented!("TODO child indexes and pop from crumb, start needed hash from index from start key, after needed from crumb");
			return Ok(false);
		}

		prev_key = key;
		prev_key_len = key_len;
	}
	Ok(true)
}

// Limiting size to u32 (could also just use a terminal character).
#[derive(Debug, PartialEq, Eq)]
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
	let mut buf = crate::query_plan::InMemoryRecorder::default();
	for i in 0..u16::MAX as u32 + 1 {
		VarInt(i).encode_into(&mut buf);
		assert_eq!(buf.buffer.len(), VarInt(i).encoded_len());
		assert_eq!(Ok((VarInt(i), buf.buffer.len())), VarInt::decode(&buf.buffer));
		buf.buffer.clear();
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
