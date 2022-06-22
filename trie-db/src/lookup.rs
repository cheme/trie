// Copyright 2017, 2018 Parity Technologies
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

//! Trie lookup via HashDB.

use crate::{
	nibble::NibbleSlice,
	node::{decode_hash, Node, NodeHandle, NodeHandleOwned, NodeOwned, Value, ValueOwned},
	node_codec::NodeCodec,
	rstd::boxed::Box,
	Bytes, CError, Context, DBValue, Query, Result, TrieAccess, TrieError,
	TrieHash, TrieLayout,
};
use hash_db::{HashDBRef, Hasher, Prefix};

/// Trie lookup helper object.
pub struct Lookup<'a, 'cache, L: TrieLayout, Q: Query<L::Hash>> {
	/// database to query from.
	pub db: &'a dyn HashDBRef<L::Hash, DBValue>,
	/// Query object to record nodes and transform data.
	pub query: Q,
	/// Hash to start at
	pub hash: TrieHash<L>,
	/// TODO
	pub context: &'cache mut dyn Context<L>,
}

impl<'a, 'cache, L, Q> Lookup<'a, 'cache, L, Q>
where
	L: TrieLayout,
	Q: Query<L::Hash>,
{
	/// Load the given value.
	///
	/// This will access the `db` if the value is not already in memory, but then it will put it
	/// into the given `cache` as `NodeOwned::Value`.
	///
	/// Returns the bytes representing the value.
	fn load_value(
		v: Value,
		prefix: Prefix,
		_full_key: &[u8],
		db: &dyn HashDBRef<L::Hash, DBValue>,
		context: &mut dyn Context<L>,
		query: Q,
	) -> Result<Q::Item, TrieHash<L>, CError<L>> {
		match v {
			Value::Inline(value) => Ok(query.decode(&value)),
			Value::Node(hash) => {
				let mut res = TrieHash::<L>::default();
				res.as_mut().copy_from_slice(hash);
				let mut h = TrieHash::<L>::default();
				h.as_mut().copy_from_slice(&hash[..]);
				let value = context.get_or_insert_node(h, true, &mut || {
					if let Some(value) = db.get(&res, prefix) {
						Ok(value)
					} else {
						Err(Box::new(TrieError::IncompleteDatabase(res)))
					}
				})?;
				// TODO have a single value function (and store value as variant of cachenode)
				match value {
					NodeOwned::Value(v, _) => Ok(query.decode(&v)),
					_ => Err(Box::new(TrieError::IncompleteDatabase(res))),
				}
			},
		}
	}

	/// Load the given value.
	///
	/// This will access the `db` if the value is not already in memory, but then it will put it
	/// into the given `cache` as `NodeOwned::Value`.
	///
	/// Returns the bytes representing the value and its hash.
	fn load_owned_value(
		v: ValueOwned<TrieHash<L>>,
		prefix: Prefix,
		full_key: &[u8],
		context: &mut dyn Context<L>,
		db: &dyn HashDBRef<L::Hash, DBValue>,
	) -> Result<(Bytes, TrieHash<L>), TrieHash<L>, CError<L>> {
		match v {
			ValueOwned::Inline(value, hash) => Ok((value.clone(), hash)),
			ValueOwned::Node(hash) => {
				let node = context.get_or_insert_node(hash, true, &mut || {
					db.get(&hash, prefix)
						.ok_or_else(|| Box::new(TrieError::IncompleteDatabase(hash)))
				})?;

				let value = node
					.data()
					.expect(
						"We are caching a `NodeOwned::Value` for a value node \
						hash and this cached node has always data attached; qed",
					)
					.clone();

					// TODO should be at top level (when querying look) only.
					context.record(TrieAccess::Value {
						hash,
						value: value.as_ref().into(),
						full_key,
					});

				Ok((value, hash))
			},
		}
	}

	/// Look up the given `nibble_key`.
	///
	/// If the value is found, it will be passed to the given function to decode or copy.
	///
	/// The given `full_key` should be the full key to the data that is requested. This will
	/// be used when there is a cache to potentially speed up the lookup.
	pub fn look_up(
		self,
		full_key: &[u8],
		nibble_key: NibbleSlice,
	) -> Result<Option<Q::Item>, TrieHash<L>, CError<L>> {
			self.look_up_without_cache(nibble_key, full_key, Self::load_value)
	}

	/// Look up the value hash for the given `nibble_key`.
	///
	/// The given `full_key` should be the full key to the data that is requested. This will
	/// be used when there is a cache to potentially speed up the lookup.
	pub fn look_up_hash(
		self,
		full_key: &[u8],
		nibble_key: NibbleSlice,
	) -> Result<Option<TrieHash<L>>, TrieHash<L>, CError<L>> {
			self.look_up_without_cache(
				nibble_key,
				full_key,
				|v, _, full_key, _, context, _| {
					Ok(match v {
						Value::Inline(v) => {
							let hash = L::Hash::hash(&v);

							// TODO this should be recorded with `cache_value_for_key`.
							context.record(TrieAccess::Value { hash, value: v.into(), full_key });

							hash
						},
						Value::Node(hash_bytes) => {
							let mut hash = TrieHash::<L>::default();
							hash.as_mut().copy_from_slice(hash_bytes);
							hash
						},
					})
				},
			)
	}

	/// When modifying any logic inside this function, you also need to do the same in
	/// [`Self::lookup_without_cache`].
	fn look_up_with_cache_internal<R>(
		&mut self,
		nibble_key: NibbleSlice,
		full_key: &[u8],
		load_value_owned: impl Fn(
			ValueOwned<TrieHash<L>>,
			Prefix,
			&[u8],
			&mut dyn Context<L>,
			&dyn HashDBRef<L::Hash, DBValue>,
		) -> Result<R, TrieHash<L>, CError<L>>,
	) -> Result<Option<R>, TrieHash<L>, CError<L>> {
		let mut partial = nibble_key;
		let mut hash = self.hash;
		let mut key_nibbles = 0;

		// this loop iterates through non-inline nodes.
		for depth in 0.. {
			let db = &mut self.db;
			let mut node = self.context.get_or_insert_node(hash, false, &mut || {
				let node_data = match db.get(&hash, nibble_key.mid(key_nibbles).left()) {
					Some(value) => value,
					None =>
						return Err(Box::new(match depth {
							0 => TrieError::InvalidStateRoot(hash),
							_ => TrieError::IncompleteDatabase(hash),
						})),
				};
				Ok(node_data)
			})?;

			// this loop iterates through all inline children (usually max 1)
			// without incrementing the depth.
			loop {
				let next_node = match node {
					NodeOwned::Leaf(slice, value) =>
						return if partial == *slice {
							let value = (*value).clone();
							drop(node);
							load_value_owned(
								value,
								nibble_key.original_data_as_prefix(),
								full_key,
								self.context,
								self.db,
							)
							.map(Some)
						} else {
							// TODO at look up level
//							self.record(|| TrieAccess::NonExisting { full_key });

							Ok(None)
						},
					NodeOwned::Extension(slice, item) =>
						if partial.starts_with_vec(&slice) {
							partial = partial.mid(slice.len());
							key_nibbles += slice.len();
							item
						} else {
							// TODO at look up level
//							self.record(|| TrieAccess::NonExisting { full_key });

							return Ok(None)
						},
					NodeOwned::Branch(children, value) =>
						if partial.is_empty() {
							return if let Some(value) = value.clone() {
								drop(node);
								load_value_owned(
									value,
									nibble_key.original_data_as_prefix(),
									full_key,
									self.context,
									self.db,
								)
								.map(Some)
							} else {
								// TODO on lookup query
								// self.record(|| TrieAccess::NonExisting { full_key });

								Ok(None)
							}
						} else {
							match &children[partial.at(0) as usize] {
								Some(x) => {
									partial = partial.mid(1);
									key_nibbles += 1;
									x
								},
								None => {
									// TODO on lookup query
									// self.record(|| TrieAccess::NonExisting { full_key });

									return Ok(None)
								},
							}
						},
					NodeOwned::NibbledBranch(slice, children, value) => {
						if !partial.starts_with_vec(&slice) {

							// TODO at lookup level
//							self.record(|| TrieAccess::NonExisting { full_key });

							return Ok(None)
						}

						if partial.len() == slice.len() {
							return if let Some(value) = value.clone() {
								drop(node);
								load_value_owned(
									value,
									nibble_key.original_data_as_prefix(),
									full_key,
									self.context,
									self.db,
								)
								.map(Some)
							} else {
								// TODO on lookup query
								// self.record(|| TrieAccess::NonExisting { full_key });

								Ok(None)
							}
						} else {
							match &children[partial.at(slice.len()) as usize] {
								Some(x) => {
									partial = partial.mid(slice.len() + 1);
									key_nibbles += slice.len() + 1;
									x
								},
								None => {
								// TODO on lookup query
								// self.record(|| TrieAccess::NonExisting { full_key });

									return Ok(None)
								},
							}
						}
					},
					NodeOwned::Empty => {
								// TODO on lookup query
								// self.record(|| TrieAccess::NonExisting { full_key });

						return Ok(None)
					},
					NodeOwned::Value(_, _) => {
						unreachable!(
							"`NodeOwned::Value` can not be reached by using the hash of a node. \
							 `NodeOwned::Value` is only constructed when loading a value into memory, \
							 which needs to have a different hash than any node; qed",
						)
					},
				};

				// check if new node data is inline or hash.
				match next_node {
					NodeHandleOwned::Hash(new_hash) => {
						hash = *new_hash;
						break
					},
					NodeHandleOwned::Inline(inline_node) => {
						node = &inline_node;
					},
				}
			}
		}

		Ok(None)
	}

	/// Look up the given key. If the value is found, it will be passed to the given
	/// function to decode or copy.
	///
	/// When modifying any logic inside this function, you also need to do the same in
	/// [`Self::lookup_with_cache_internal`].
	fn look_up_without_cache<R>(
		self,
		nibble_key: NibbleSlice,
		full_key: &[u8],
		load_value: impl Fn(
			Value,
			Prefix,
			&[u8],
			&dyn HashDBRef<L::Hash, DBValue>,
			&mut dyn Context<L>,
			Q,
		) -> Result<R, TrieHash<L>, CError<L>>,
	) -> Result<Option<R>, TrieHash<L>, CError<L>> {
		let mut partial = nibble_key;
		let mut hash = self.hash;
		let mut key_nibbles = 0;

		// this loop iterates through non-inline nodes.
		for depth in 0.. {
			// TODO get_or_insert_node that record directly
			let node_data = match self.db.get(&hash, nibble_key.mid(key_nibbles).left()) {
				Some(value) => value,
				None =>
					return Err(Box::new(match depth {
						0 => TrieError::InvalidStateRoot(hash),
						_ => TrieError::IncompleteDatabase(hash),
					})),
			};

			// this loop iterates through all inline children (usually max 1)
			// without incrementing the depth.
			let mut node_data = &node_data[..];
			loop {
				let decoded = match L::Codec::decode(node_data) {
					Ok(node) => node,
					Err(e) => return Err(Box::new(TrieError::DecoderError(hash, e))),
				};

				let next_node = match decoded {
					Node::Leaf(slice, value) =>
						return if slice == partial {
							load_value(
								value,
								nibble_key.original_data_as_prefix(),
								full_key,
								self.db,
								self.context,
								self.query,
							)
							.map(Some)
						} else {
							// TODO at call level
//							self.record(|| TrieAccess::NonExisting { full_key });

							Ok(None)
						},
					Node::Extension(slice, item) =>
						if partial.starts_with(&slice) {
							partial = partial.mid(slice.len());
							key_nibbles += slice.len();
							item
						} else {
							// TODO at call level
							//self.record(|| TrieAccess::NonExisting { full_key });

							return Ok(None)
						},
					Node::Branch(children, value) =>
						if partial.is_empty() {
							return if let Some(val) = value {
								load_value(
									val,
									nibble_key.original_data_as_prefix(),
									full_key,
									self.db,
									self.context,
									self.query,
								)
								.map(Some)
							} else {
							// TODO at call level
							//	self.record(|| TrieAccess::NonExisting { full_key });

								Ok(None)
							}
						} else {
							match children[partial.at(0) as usize] {
								Some(x) => {
									partial = partial.mid(1);
									key_nibbles += 1;
									x
								},
								None => {
							// TODO at call level
							//		self.record(|| TrieAccess::NonExisting { full_key });

									return Ok(None)
								},
							}
						},
					Node::NibbledBranch(slice, children, value) => {
						if !partial.starts_with(&slice) {
							// TODO at call level
							//self.record(|| TrieAccess::NonExisting { full_key });

							return Ok(None)
						}

						if partial.len() == slice.len() {
							return if let Some(val) = value {
								load_value(
									val,
									nibble_key.original_data_as_prefix(),
									full_key,
									self.db,
									self.context,
									self.query,
								)
								.map(Some)
							} else {
							// TODO at call level
							//	self.record(|| TrieAccess::NonExisting { full_key });

								Ok(None)
							}
						} else {
							match children[partial.at(slice.len()) as usize] {
								Some(x) => {
									partial = partial.mid(slice.len() + 1);
									key_nibbles += slice.len() + 1;
									x
								},
								None => {
							// TODO at call level
							//		self.record(|| TrieAccess::NonExisting { full_key });

									return Ok(None)
								},
							}
						}
					},
					Node::Empty => {
							// TODO at call level
						//self.record(|| TrieAccess::NonExisting { full_key });

						return Ok(None)
					},
				};

				// check if new node data is inline or hash.
				match next_node {
					NodeHandle::Hash(data) => {
						hash = decode_hash::<L::Hash>(data)
							.ok_or_else(|| Box::new(TrieError::InvalidHash(hash, data.to_vec())))?;
						break
					},
					NodeHandle::Inline(data) => {
						node_data = data;
					},
				}
			}
		}
		Ok(None)
	}
}
