// Copyright 2017, 2020 Parity Technologies
//
// Licensed under the Apache License, Version .0 (the "License");
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

use trie_db::DBValue;
use memory_db::{MemoryDB, HashKey, PrefixedKey};
use keccak_hasher::KeccakHasher;
use trie_db::partial_db::DepthIndexes;

#[test]
fn trie_root_empty () {
	compare_implementations(vec![])
}

#[test]
fn trie_one_node () {
	compare_implementations(vec![
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8]),
	]);
}

#[test]
fn root_extension_one () {
	compare_implementations(vec![
		(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;32]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
	]);
}

fn test_iter(data: Vec<(Vec<u8>, Vec<u8>)>) {
	use reference_trie::{RefTrieDBMut, RefTrieDB};
	use trie_db::{TrieMut, Trie};

	let mut db = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
	let mut root = Default::default();
	{
		let mut t = RefTrieDBMut::new(&mut db, &mut root);
		for i in 0..data.len() {
			let key: &[u8]= &data[i].0;
			let value: &[u8] = &data[i].1;
			t.insert(key, value).unwrap();
		}
	}
	let t = RefTrieDB::new(&db, &root).unwrap();
	for (i, kv) in t.iter().unwrap().enumerate() {
		let (k, v) = kv.unwrap();
		let key: &[u8]= &data[i].0;
		let value: &[u8] = &data[i].1;
		assert_eq!(k, key);
		assert_eq!(v, value);
	}
	for (k, v) in data.into_iter() {
		assert_eq!(&t.get(&k[..]).unwrap().unwrap()[..], &v[..]);
	}
}

fn test_iter_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
	use reference_trie::{RefTrieDBMutNoExt, RefTrieDBNoExt};
	use trie_db::{TrieMut, Trie};

	let mut db = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
	let mut root = Default::default();
	{
		let mut t = RefTrieDBMutNoExt::new(&mut db, &mut root);
		for i in 0..data.len() {
			let key: &[u8]= &data[i].0;
			let value: &[u8] = &data[i].1;
			t.insert(key, value).unwrap();
		}
	}
	let t = RefTrieDBNoExt::new(&db, &root).unwrap();
	for (i, kv) in t.iter().unwrap().enumerate() {
		let (k, v) = kv.unwrap();
		let key: &[u8]= &data[i].0;
		let value: &[u8] = &data[i].1;
		assert_eq!(k, key);
		assert_eq!(v, value);
	}
	for (k, v) in data.into_iter() {
		assert_eq!(&t.get(&k[..]).unwrap().unwrap()[..], &v[..]);
	}
}

fn compare_implementations(data: Vec<(Vec<u8>, Vec<u8>)>) {
	test_iter(data.clone());
	test_iter_no_extension(data.clone());
	compare_implementations_h(data.clone());
	compare_implementations_prefixed(data.clone());
	compare_implementations_no_extension(data.clone());
	compare_implementations_no_extension_prefixed(data.clone());
	compare_indexing(data.clone());
}

fn compare_indexing(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let mut indexes = std::collections::BTreeMap::new();
	let indexes_conf = DepthIndexes::new(&[
		1, 2, 4, 6, 9,
	]);
	reference_trie::compare_indexing(data, memdb, &mut indexes, &indexes_conf);
}
fn compare_implementations_prefixed(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let hashdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
	reference_trie::compare_implementations(data, memdb, hashdb);
}
fn compare_implementations_h(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
	reference_trie::compare_implementations(data, memdb, hashdb);
}
fn compare_implementations_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
	reference_trie::compare_implementations_no_extension(data, memdb, hashdb);
}
fn compare_implementations_no_extension_prefixed(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let hashdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
	reference_trie::compare_implementations_no_extension(data, memdb, hashdb);
}
fn compare_implementations_no_extension_unordered(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let hashdb = MemoryDB::<KeccakHasher, HashKey<_>, DBValue>::default();
	reference_trie::compare_implementations_no_extension_unordered(data, memdb, hashdb);
}
fn compare_no_extension_insert_remove(data: Vec<(bool, Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	reference_trie::compare_no_extension_insert_remove(data, memdb);
}
fn compare_root(data: Vec<(Vec<u8>, Vec<u8>)>) {
	let memdb = MemoryDB::<_, HashKey<_>, _>::default();
	reference_trie::compare_root(data, memdb);
}
fn compare_unhashed(data: Vec<(Vec<u8>, Vec<u8>)>) {
	reference_trie::compare_unhashed(data);
}
fn compare_unhashed_no_extension(data: Vec<(Vec<u8>, Vec<u8>)>) {
	reference_trie::compare_unhashed_no_extension(data);
}

// Following tests are a bunch of detected issue here for non regression.

#[test]
fn trie_middle_node1 () {
	compare_implementations(vec![
		(vec![1u8, 2u8], vec![8u8;32]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
	]);
}
#[test]
fn trie_middle_node2 () {
	compare_implementations(vec![
		(vec![0u8, 2u8, 3u8, 5u8, 3u8], vec![1u8;32]),
		(vec![1u8, 2u8], vec![8u8;32]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
		(vec![1u8, 2u8, 3u8, 5u8], vec![7u8;32]),
		(vec![1u8, 2u8, 3u8, 5u8, 3u8], vec![7u8;32]),
	]);
}
#[test]
fn root_extension_bis () {
	compare_root(vec![
		(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;32]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
	]);
}
#[test]
fn root_extension_tierce () {
	let d = vec![
		(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;2]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;2]),
	];
	compare_unhashed(d.clone());
	compare_unhashed_no_extension(d);
}
#[test]
fn root_extension_tierce_big () {
	// on more content unhashed would hash
	compare_unhashed(vec![
		(vec![1u8, 2u8, 3u8, 3u8], vec![8u8;32]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;32]),
		(vec![1u8, 6u8, 3u8, 3u8], vec![8u8;32]),
		(vec![6u8, 2u8, 3u8, 3u8], vec![8u8;32]),
		(vec![6u8, 2u8, 3u8, 13u8], vec![8u8;32]),
	]);
}
#[test]
fn trie_middle_node2x () {
	compare_implementations(vec![
		(vec![0u8, 2u8, 3u8, 5u8, 3u8], vec![1u8;2]),
		(vec![1u8, 2u8], vec![8u8;2]),
		(vec![1u8, 2u8, 3u8, 4u8], vec![7u8;2]),
		(vec![1u8, 2u8, 3u8, 5u8], vec![7u8;2]),
		(vec![1u8, 2u8, 3u8, 5u8, 3u8], vec![7u8;2]),
	]);
}
#[test]
fn fuzz1 () {
	compare_implementations(vec![
		(vec![01u8], vec![42u8, 9]),
		(vec![01u8, 0u8], vec![0u8, 0]),
		(vec![255u8, 2u8], vec![1u8, 0]),
	]);
}
#[test]
fn fuzz2 () {
	compare_implementations(vec![
		(vec![0, 01u8], vec![42u8, 9]),
		(vec![0, 01u8, 0u8], vec![0u8, 0]),
		(vec![0, 255u8, 2u8], vec![1u8, 0]),
	]);
}
#[test]
fn fuzz3 () {
	compare_implementations(vec![
		(vec![0], vec![196, 255]),
		(vec![48], vec![138, 255]),
		(vec![67], vec![0, 0]),
		(vec![128], vec![255, 0]),
		(vec![247], vec![0, 196]),
		(vec![255], vec![0, 0]),
	]);
}
#[test]
fn fuzz_no_extension1 () {
	compare_implementations(vec![
		(vec![0], vec![128, 0]),
		(vec![128], vec![0, 0]),
	]);
}
#[test]
fn fuzz_no_extension2 () {
	compare_implementations(vec![
		(vec![0], vec![6, 255]),
		(vec![6], vec![255, 186]),
		(vec![255], vec![186, 255]),
	]);
}
#[test]
fn fuzz_no_extension5 () {
	compare_implementations(vec![
		(vec![0xaa], vec![0xa0]),
		(vec![0xaa, 0xaa], vec![0xaa]),
		(vec![0xaa, 0xbb], vec![0xab]),
		(vec![0xbb], vec![0xb0]),
		(vec![0xbb, 0xbb], vec![0xbb]),
		(vec![0xbb, 0xcc], vec![0xbc]),
	]);
}
#[test]
fn fuzz_no_extension3 () {
	compare_implementations(vec![
		(vec![0], vec![0, 0]),
		(vec![11, 0], vec![0, 0]),
		(vec![11, 252], vec![11, 0]),
	]);

	compare_implementations_no_extension_unordered(vec![
		(vec![11, 252], vec![11, 0]),
		(vec![11, 0], vec![0, 0]),
		(vec![0], vec![0, 0]),
	]);
}
#[test]
fn fuzz_no_extension4 () {
	compare_implementations_no_extension(vec![
		(vec![0x01, 0x56], vec![0x1]),
		(vec![0x02, 0x42], vec![0x2]),
		(vec![0x02, 0x50], vec![0x3]),
	]);
}
#[test]
fn fuzz_no_extension_insert_remove_1 () {
	let data = vec![
		(false, vec![0], vec![251, 255]),
		(false, vec![0, 1], vec![251, 255]),
		(false, vec![0, 1, 2], vec![255; 32]),
		(true, vec![0, 1], vec![0, 251]),
	];
	compare_no_extension_insert_remove(data);
}
#[test]
fn fuzz_no_extension_insert_remove_2 () {
	let data = vec![
		(false, vec![0x00], vec![0xfd, 0xff]),
		(false, vec![0x10, 0x00], vec![1;32]),
		(false, vec![0x11, 0x10], vec![0;32]),
		(true, vec![0x10, 0x00], vec![])
	];
	compare_no_extension_insert_remove(data);
}
#[test]
fn two_bytes_nibble_length () {
	let data = vec![
		(vec![00u8], vec![0]),
		(vec![01u8;64], vec![0;32]),
	];
	compare_implementations_no_extension(data.clone());
	compare_implementations_no_extension_prefixed(data.clone());
}
#[test]
#[should_panic]
fn too_big_nibble_length_old () {
	compare_implementations_h(vec![
		(vec![01u8;64], vec![0;32]),
	]);
}
#[test]
fn too_big_nibble_length_new () {
	compare_implementations_no_extension(vec![
		(vec![01u8;((u16::max_value() as usize + 1) / 2) + 1], vec![0;32]),
	]);
}
#[test]
fn polka_re_test () {
	compare_implementations(vec![
		(vec![77, 111, 111, 55, 111, 104, 121, 97], vec![68, 97, 105, 55, 105, 101, 116, 111]),
		(vec![101, 105, 67, 104, 111, 111, 66, 56], vec![97, 56, 97, 113, 117, 53, 97]),
		(vec![105, 97, 48, 77, 101, 105, 121, 101], vec![69, 109, 111, 111, 82, 49, 97, 105]),
	]);
}

fn compare_index_calc(
	data: Vec<(Vec<u8>, Vec<u8>)>,
	change: Vec<(Vec<u8>, Option<Vec<u8>>)>,
	depth_indexes: Vec<u32>,
	nb_node_fetch: Option<usize>,
) {
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let mut indexes = std::collections::BTreeMap::new();
	let indexes_conf = DepthIndexes::new(&depth_indexes[..]);
	reference_trie::compare_index_calc(data, change, memdb, &mut indexes, &indexes_conf, nb_node_fetch);
}

#[test]
fn compare_index_calculations() {
	let empty = vec![];
	let one_level_branch = vec![
		(b"test".to_vec(), vec![2u8; 32]),
		(b"tett".to_vec(), vec![3u8; 32]),
		(b"teut".to_vec(), vec![4u8; 32]),
		(b"tevtc".to_vec(), vec![5u8; 32]),
		(b"tewtb".to_vec(), vec![6u8; 32]),
		(b"tezta".to_vec(), vec![6u8; 32]),
	];
	let two_level_branch = vec![
		(b"test".to_vec(), vec![2u8; 32]),
		(b"testi".to_vec(), vec![2u8; 32]),
		(b"tett".to_vec(), vec![3u8; 32]),
		(b"tetti".to_vec(), vec![3u8; 32]),
		(b"teut".to_vec(), vec![4u8; 32]),
		(b"teuti".to_vec(), vec![4u8; 32]),
		(b"tevtc".to_vec(), vec![5u8; 32]),
		(b"tevtci".to_vec(), vec![5u8; 32]),
		(b"tewtb".to_vec(), vec![6u8; 32]),
		(b"tewtbi".to_vec(), vec![6u8; 32]),
		(b"tezta".to_vec(), vec![6u8; 32]),
		(b"teztai".to_vec(), vec![6u8; 32]),
	];

	let inputs = vec![
		(empty.clone(), vec![], vec![], Some(0)),
		(empty.clone(), vec![], vec![2, 5], Some(0)),
		(empty.clone(), vec![(b"te".to_vec(), None)], vec![2, 5], Some(0)),
		(empty.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![], Some(0)),
		(empty.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![8, 20], Some(0)),

		(one_level_branch.clone(), vec![], vec![], Some(6)),
		// 6 as read ahead.
		(one_level_branch.clone(), vec![], vec![2, 5], Some(6)),
		(one_level_branch.clone(), vec![], vec![5], Some(6)),
		(one_level_branch.clone(), vec![], vec![6], Some(7)),
		(one_level_branch.clone(), vec![], vec![7], Some(7)),
		(one_level_branch.clone(), vec![], vec![6, 7], Some(7)),

		// insert before indexes
		// index one child
		(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![5], Some(6)),
		// index on children
		(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![7], Some(7)),
		(two_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![7], Some(8)),
		// index after children
		(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![10], Some(7)),
		(one_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![10], Some(7)),
		(two_level_branch.clone(), vec![(b"te".to_vec(), Some(vec![12; 32]))], vec![10], Some(11)),

		// insert onto indexes
		// insert after indexes
	];
	for (data, change, depth_indexes, nb_fetch) in inputs.into_iter() {
		compare_index_calc(data, change, depth_indexes, nb_fetch);
	}
}

#[test]
fn check_indexing() {
	use trie_db::BackingByteVec;
	use trie_db::partial_db::Index;
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let data = vec![
		(b"alfa".to_vec(), vec![0; 32]),
		(b"bravo".to_vec(), vec![1; 32]),
//		(b"do".to_vec(), vec![2; 32]),
		(b"dog".to_vec(), vec![3; 32]),
		(b"doge".to_vec(), vec![4; 32]),
		(b"horse".to_vec(), vec![5; 32]),
		(b"house".to_vec(), vec![6; 32]),
	];

	let mut indexes = std::collections::BTreeMap::new();
	let indexes_conf = DepthIndexes::new(&[
		6,
	]);
	let mut expected = vec![
		// alf
		(vec![0, 0, 0, 6, 97, 108, 102], false),
		// bra
		(vec![0, 0, 0, 6, 98, 114, 97], false),
		// dog
		(vec![0, 0, 0, 6, 100, 111, 103], true),
		// hor
		(vec![0, 0, 0, 6, 104, 111, 114], false),
		// hou
		(vec![0, 0, 0, 6, 104, 111, 117], false),
	];
	reference_trie::compare_indexing(data.clone(), memdb.clone(), &mut indexes, &indexes_conf);
	for index in indexes.into_iter().rev() {
		assert_eq!(expected.pop().unwrap(), (index.0.to_vec(), (index.1).on_index));
	}
	assert!(expected.is_empty());

	let mut indexes = std::collections::BTreeMap::new();
	let indexes_conf = DepthIndexes::new(&[
		4, 8,
	]);
	let mut expected = vec![
		// alf
		(vec![0, 0, 0, 4, 97, 108], false),
		// bra
		(vec![0, 0, 0, 4, 98, 114], false),
		// dog
		(vec![0, 0, 0, 4, 100, 111], false),
		// hou hor branch (7)
		(vec![0, 0, 0, 4, 104, 111], false),
		// doge
		(vec![0, 0, 0, 8, 100, 111, 103, 101], true),
		// hou
		(vec![0, 0, 0, 8, 104, 111, 114, 115], false),
		(vec![0, 0, 0, 8, 104, 111, 117, 115], false),
	];
	reference_trie::compare_indexing(data.clone(), memdb.clone(), &mut indexes, &indexes_conf);
	for index in indexes.into_iter().rev() {
		assert_eq!(expected.pop().unwrap(), (index.0.to_vec(), (index.1).on_index));
	}
	assert!(expected.is_empty());


//	panic!("{:?}", indexes);
	let mut indexes = std::collections::BTreeMap::<BackingByteVec, Index>::new();
	let indexes_conf = DepthIndexes::new(&[
		1, 2, 4, 6, 8,
	]);
	reference_trie::compare_indexing(data.clone(), memdb.clone(), &mut indexes, &indexes_conf);
	
	let mut expected = vec![
		(vec![0, 0, 0, 1, 6], true), // root on nibble 1
		// 2, a
		(vec![0, 0, 0, 2, 97], false),
		(vec![0, 0, 0, 2, 98], false),
		(vec![0, 0, 0, 2, 100], false),
		(vec![0, 0, 0, 2, 104], false),
		// h
		(vec![0, 0, 0, 6, 104, 111, 114], false),
		(vec![0, 0, 0, 6, 104, 111, 117], false),
		// doge
		(vec![0, 0, 0, 8, 100, 111, 103, 101], true),
	];
	for index in indexes.into_iter().rev() {
		assert_eq!(expected.pop().unwrap(), (index.0.to_vec(), (index.1).on_index));
	}
	println!("{:?}", expected);
	assert!(expected.is_empty());
}
