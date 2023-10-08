// Copyright 2017, 2020 Parity Technologies
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

use crate::{
	rstd::boxed::Box, triedb::TrieDB, CError, DBValue, MerkleValue, Query, Result, Trie,
	TrieDBBuilder, TrieHash, TrieItem, TrieIterator, TrieKeyItem, TrieLayout,
};
use hash_db::{HashDBRef, Hasher};

/// A `Trie` implementation which hashes keys and uses a generic `HashDB` backing database.
///
/// Use it as a `Trie` trait object. You can use `raw()` to get the backing `TrieDB` object.
pub struct SecTrieDB<'db, 'cache, L, const N: usize>
where
	L: TrieLayout<N>,
{
	raw: TrieDB<'db, 'cache, L, N>,
}

impl<'db, 'cache, L, const N: usize> SecTrieDB<'db, 'cache, L, N>
where
	L: TrieLayout<N>,
{
	/// Create a new trie with the backing database `db` and `root`.
	///
	/// Initialise to the state entailed by the genesis block.
	/// This guarantees the trie is built correctly.
	pub fn new(db: &'db dyn HashDBRef<L::Hash, DBValue>, root: &'db TrieHash<L, N>) -> Self {
		SecTrieDB { raw: TrieDBBuilder::new(db, root).build() }
	}

	/// Get a reference to the underlying raw `TrieDB` struct.
	pub fn raw(&self) -> &TrieDB<'db, 'cache, L, N> {
		&self.raw
	}

	/// Get a mutable reference to the underlying raw `TrieDB` struct.
	pub fn raw_mut(&mut self) -> &mut TrieDB<'db, 'cache, L, N> {
		&mut self.raw
	}
}

impl<'db, 'cache, L, const N: usize> Trie<L, N> for SecTrieDB<'db, 'cache, L, N>
where
	L: TrieLayout<N>,
{
	fn root(&self) -> &TrieHash<L, N> {
		self.raw.root()
	}

	fn contains(&self, key: &[u8]) -> Result<bool, TrieHash<L, N>, CError<L, N>> {
		self.raw.contains(L::Hash::hash(key).as_ref())
	}

	fn get_hash(&self, key: &[u8]) -> Result<Option<TrieHash<L, N>>, TrieHash<L, N>, CError<L, N>> {
		self.raw.get_hash(key)
	}

	fn get_with<Q: Query<L::Hash>>(
		&self,
		key: &[u8],
		query: Q,
	) -> Result<Option<Q::Item>, TrieHash<L, N>, CError<L, N>> {
		self.raw.get_with(L::Hash::hash(key).as_ref(), query)
	}

	fn lookup_first_descendant(
		&self,
		key: &[u8],
	) -> Result<Option<MerkleValue<TrieHash<L, N>>>, TrieHash<L, N>, CError<L, N>> {
		self.raw.lookup_first_descendant(key)
	}

	fn iter<'a>(
		&'a self,
	) -> Result<
		Box<dyn TrieIterator<L, N, Item = TrieItem<TrieHash<L, N>, CError<L, N>>> + 'a>,
		TrieHash<L, N>,
		CError<L, N>,
	> {
		TrieDB::iter(&self.raw)
	}

	fn key_iter<'a>(
		&'a self,
	) -> Result<
		Box<dyn TrieIterator<L, N, Item = TrieKeyItem<TrieHash<L, N>, CError<L, N>>> + 'a>,
		TrieHash<L, N>,
		CError<L, N>,
	> {
		TrieDB::key_iter(&self.raw)
	}
}
