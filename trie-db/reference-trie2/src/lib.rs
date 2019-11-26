
extern crate reference_trie;

pub use reference_trie::*;

use memory_db::{MemoryDB, HashKey, PrefixedKey};
use reference_trie::{
	RefTrieDBMutNoExt,
	RefTrieDBMut,
	reference_trie_root,
	calc_root_no_extension,
	compare_no_extension_insert_remove,
};
use trie_db::{TrieMut, DBValue};
use keccak_hasher::KeccakHasher;

pub fn mut_insert(data: &Vec<(Vec<u8>,Vec<u8>)>) -> ([u8; 32], Vec<Vec<u8>>, usize) {
	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	{
	let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
	for a in 0..data.len() {
		t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
	}
	}
	let mut iter_res = Vec::new();
	let mut error = 0;
	{
			let trie = RefTrieDB::new(&memdb, &root).unwrap();
			let mut iter = trie.iter().unwrap();
			let prefix = &b"012"[..];
			iter.seek(prefix).unwrap();

			for x in iter {
				let (key, _) = x.unwrap();
				if key.starts_with(prefix) {
					iter_res.push(key);
				} else {
					error +=1;
				}

			}

	}
	
	(root, iter_res, error)
}
