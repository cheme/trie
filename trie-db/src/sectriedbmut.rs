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
	triedbmut::TrieDBMutBuilder, CError, DBValue, Result, TrieDBMut, TrieHash, TrieLayout, TrieMut,
	Value,
};
use hash_db::{HashDB, Hasher};

/// A mutable `Trie` implementation which hashes keys and uses a generic `HashDB` backing database.
///
/// Use it as a `Trie` or `TrieMut` trait object. You can use `raw()` to get the backing `TrieDBMut`
/// object.
pub struct SecTrieDBMut<'db, L, const N: usize>
where
	L: TrieLayout<N>,
{
	raw: TrieDBMut<'db, L, N>,
}

impl<'db, L, const N: usize> SecTrieDBMut<'db, L, N>
where
	L: TrieLayout<N>,
{
	/// Create a new trie with the backing database `db` and empty `root`
	/// Initialize to the state entailed by the genesis block.
	/// This guarantees the trie is built correctly.
	pub fn new(db: &'db mut dyn HashDB<L::Hash, DBValue>, root: &'db mut TrieHash<L, N>) -> Self {
		SecTrieDBMut { raw: TrieDBMutBuilder::new(db, root).build() }
	}

	/// Create a new trie with the backing database `db` and `root`.
	pub fn from_existing(
		db: &'db mut dyn HashDB<L::Hash, DBValue>,
		root: &'db mut TrieHash<L, N>,
	) -> Self {
		SecTrieDBMut { raw: TrieDBMutBuilder::from_existing(db, root).build() }
	}

	/// Get the backing database.
	pub fn db(&self) -> &dyn HashDB<L::Hash, DBValue> {
		self.raw.db()
	}

	/// Get the backing database.
	pub fn db_mut(&mut self) -> &mut dyn HashDB<L::Hash, DBValue> {
		self.raw.db_mut()
	}
}

impl<'db, L, const N: usize> TrieMut<L, N> for SecTrieDBMut<'db, L, N>
where
	L: TrieLayout<N>,
{
	fn root(&mut self) -> &TrieHash<L, N> {
		self.raw.root()
	}

	fn is_empty(&self) -> bool {
		self.raw.is_empty()
	}

	fn contains(&self, key: &[u8]) -> Result<bool, TrieHash<L, N>, CError<L, N>> {
		self.raw.contains(&L::Hash::hash(key).as_ref())
	}

	fn get<'a, 'key>(
		&'a self,
		key: &'key [u8],
	) -> Result<Option<DBValue>, TrieHash<L, N>, CError<L, N>>
	where
		'a: 'key,
	{
		self.raw.get(&L::Hash::hash(key).as_ref())
	}

	fn insert(
		&mut self,
		key: &[u8],
		value: &[u8],
	) -> Result<Option<Value<L, N>>, TrieHash<L, N>, CError<L, N>> {
		self.raw.insert(&L::Hash::hash(key).as_ref(), value)
	}

	fn remove(&mut self, key: &[u8]) -> Result<Option<Value<L, N>>, TrieHash<L, N>, CError<L, N>> {
		self.raw.remove(&L::Hash::hash(key).as_ref())
	}
}
