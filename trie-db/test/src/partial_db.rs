// Copyright 2020, 2020 Parity Technologies
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

//! Tests for trie_db partial_db mod.

use trie_db::partial_db::{DepthIndexes, Index, IndexBackend,
	KVBackendIter, KVBackend, RootIndexIterator2};
use std::collections::BTreeMap;
use std::cmp::Ordering;
use trie_db::nibble_ops;
use trie_db::BackingByteVec;
use trie_db::LeftNibbleSlice;
use trie_db::NibbleVec;

/// A filled (up to a maximum non include size key) key value backend.
/// Second usize is a width (255 for all keys).
struct TestKVBackend(usize, u8);

/// A filled (up to a maximum non include size key) key value backend.
struct TestKVBackendIter(Vec<u8>, usize, bool, u8);

impl KVBackend for TestKVBackend {
	fn read(&self, key: &[u8]) -> Option<Vec<u8>> {
		if key.len() < self.0 {
			Some(vec![1u8])
		} else {
			None
		}
	}
	fn write(&mut self, _key: &[u8], _value: &[u8]) {
		unreachable!("Unsupported")
	}
	fn remove(&mut self, _key: &[u8]) {
		unreachable!("Unsupported")
	}
	fn iter<'a>(&'a self) -> KVBackendIter<'a> {
		self.iter_from(&[])
	}
	fn iter_from<'a>(&'a self, start: &[u8]) -> KVBackendIter<'a> {
		Box::new(TestKVBackendIter(start.to_vec(), self.0, false, self.1))
	}
}

impl Iterator for TestKVBackendIter {
	type Item = (Vec<u8>, Vec<u8>);
	fn next(&mut self) -> Option<Self::Item> {
		if self.1 == 0 {
			return None;
		}
		let key = self.0.clone();


		if self.2 {
			// going upward
			loop {
				let len = self.0.len();
				let last = self.0[len - 1];
				if last == self.3 {
					self.0.pop();
					if self.0.is_empty() {
						self.1 = 0;
						break;
					}
				} else {
					self.0[len - 1] += 1;
					self.2 = false;
					break;
				}
			}
		} else {
			// going downward
			if self.0.len() == self.1 - 1 {
				self.2 = true;
				return self.next();
			} else {
				self.0.push(0u8);
			}
		}
		Some((key, vec![1u8]))
	}
}

#[test]
fn test_root_iter() {
	let width = 16;
	let mut kvbackend = TestKVBackend(4, width);
	let mut kvbackenditer = kvbackend.iter();
	let mut nb = 0;
	for (k, v) in kvbackenditer {
		nb += 1;
//			println!("{:?} at {:?}", k, ix);
	}
	let mut index_backend: BTreeMap<BackingByteVec, Index> = Default::default();
	let idepth1: usize = 2;
	let depth_index = DepthIndexes::new(&[idepth1 as u32]);
	let mut index_deleted = Vec::new(); 
	let mut value_deleted = Vec::new(); 
	let mut root_iter = RootIndexIterator2::<_, _, Vec<u8>, _>::new(
		&kvbackend,
		&index_backend,
		&depth_index,
		std::iter::empty(),
		&mut index_deleted,
		&mut value_deleted,
	);
	let mut nb2 = 0;
	for (k, v) in root_iter {
		nb2 += 1;
	}
	assert_eq!(nb, nb2);
	let mut index_backend: BTreeMap<BackingByteVec, Index> = Default::default();
	let index1 = vec![0];
	let index2 = vec![5];
	index_backend.write(idepth1, LeftNibbleSlice::new_len(index1.as_slice(), idepth1), Index{hash: Default::default(), on_index: false});
	index_backend.write(idepth1, LeftNibbleSlice::new_len(index2.as_slice(), idepth1), Index{hash: Default::default(), on_index: true});
	let mut index_deleted = Vec::new(); 
	let mut value_deleted = Vec::new(); 
	let mut root_iter = RootIndexIterator2::<_, _, Vec<u8>, _>::new(
		&kvbackend,
		&index_backend,
		&depth_index,
		std::iter::empty(),
		&mut index_deleted,
		&mut value_deleted,
	);
	let mut nb3 = 0;
	for (k, v) in root_iter {
		if let Item::Index(..) = v {
		} else {
			let common_depth = nibble_ops::biggest_depth(
				&k.inner()[..],
				index1.as_slice(),
			);
			assert!(common_depth < 2);
			let common_depth = nibble_ops::biggest_depth(
				&k.inner()[..],
				index2.as_slice(),
			);
			assert!(common_depth < 2);
		}
		nb3 += 1;
	}
	assert_ne!(nb2, nb3);
	let depth_index = DepthIndexes::new(&[3, 6]);
	let mut index_backend: BTreeMap<BackingByteVec, Index> = Default::default();
	let index1 = vec![0, 0];
	let index11 = vec![0, 1, 0];
	let index12 = vec![0, 1, 5];
	index_backend.write(3, LeftNibbleSlice::new_len(index1.as_slice(), 3), Index{ hash: Default::default(), on_index: true});
	index_backend.write(6, LeftNibbleSlice::new_len(index11.as_slice(), 6), Index{ hash: Default::default(), on_index: false});
	index_backend.write(6, LeftNibbleSlice::new_len(index12.as_slice(), 6), Index{ hash: Default::default(), on_index: true});
	let mut index_deleted = Vec::new(); 
	let mut value_deleted = Vec::new(); 
	let mut root_iter = RootIndexIterator2::<_, _, Vec<u8>, _>::new(
		&kvbackend,
		&index_backend,
		&depth_index,
		std::iter::empty(),
		&mut index_deleted,
		&mut value_deleted,
	);
	let mut nb3 = 0;
	for (k, v) in root_iter {
		if let Item::Index(..) = v {
		} else {
			let common_depth = nibble_ops::biggest_depth(
				&k.inner()[..],
				index1.as_slice(),
			);
			assert!(common_depth < 3);
		}
		nb3 += 1;
	}
	assert_ne!(nb2, nb3);
	let mut index_deleted = Vec::new(); 
	let mut value_deleted = Vec::new(); 
	let mut root_iter = RootIndexIterator2::<_, _, Vec<u8>, _>::new(
		&kvbackend,
		&index_backend,
		&depth_index,
		// change to stack second layer iter
		vec![(index1.clone(), None)].into_iter(),
		&mut index_deleted,
		&mut value_deleted,
	);
	let mut nb3 = 0;
	let mut nb4 = 0;
	for (k, v) in root_iter {
		if let Item::Index(..) = v {
		} else {
			let common_depth = nibble_ops::biggest_depth(
				&k.inner()[..],
				index11.as_slice(),
			);
			assert!(common_depth < 6);
			let common_depth = nibble_ops::biggest_depth(
				&k.inner()[..],
				index12.as_slice(),
			);
			assert!(common_depth < 6);
			nb3 += 1;
		}
		nb4 += 1;
	}
	assert_ne!(nb2, nb3);
	assert_eq!(nb2, nb4);
}

#[test]
fn test_root_index_1() {
	let width = 16;
	let mut kvbackend = TestKVBackend(4, width);
	let mut kvbackenditer = kvbackend.iter();
	let mut nb = 0;
	let mut indexes_backend = BTreeMap::new();

	let indexes = DepthIndexes::new(&[1, 2, 3]);
	reference_trie::build_index(&mut indexes_backend, &indexes, kvbackenditer);
	//panic!("{:?}, {:?}", indexes_backend, indexes_backend.len())
}
#[test]
fn test_root_index_2() {
	let mut kvbackenditer = vec![
		(vec![1;32], vec![0;32]),
		(vec![1;64], vec![3;32]),
	];
	let mut nb = 0;
	let mut indexes_backend = BTreeMap::new();

	let indexes = DepthIndexes::new(&[]);
	let root_1 = reference_trie::build_index(&mut indexes_backend, &indexes, kvbackenditer.clone().into_iter());
	let mut indexes_backend = BTreeMap::new();
	let indexes = DepthIndexes::new(&[65]);
	let root_2 = reference_trie::build_index(&mut indexes_backend, &indexes, kvbackenditer.into_iter());
	assert_eq!(root_1, root_2);
//		panic!("{:?}, {:?}, {:?}", indexes_backend, indexes_backend.len(), root_1 == root_2);
}

#[test]
fn test_root_index_runs() {
	test_root_index(&[32], 500, 4);
//	test_root_index(&[15], 500, 60);
//	test_root_index(&[1, 2, 3, 4, 5, 15, 20], 500, 160);
	test_root_index(&[15, 25, 30], 50, 600);
//	test_root_index(&[15, 25, 30], 1, 600_000);
}

#[cfg(test)]
fn test_root_index(indexes: &'static [u32], nb_iter: usize, count: u32) {
	use trie_standardmap::*;

	let mut seed: [u8; 32] = Default::default();
	for _ in 0..nb_iter {
		// TODO should move to iter_build
		let x = StandardMap {
			alphabet: Alphabet::Custom(b"@QWERTYUIOPASDFGHJKLZXCVBNM[/]^_".to_vec()),
			min_key: 5,
			journal_key: 32 - 5,
			value_mode: ValueMode::Index,
			count,
		}.make_with(&mut seed);

		use memory_db::{MemoryDB, HashKey, PrefixedKey};
		use keccak_hasher::KeccakHasher;

		let indexes_conf = DepthIndexes::new(indexes);
		let memdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, Vec<u8>>::default();
		let mut indexes = std::collections::BTreeMap::new();
		let change = Vec::new();
		let data: BTreeMap<_, _> = x.into_iter().collect();
		let data: Vec<_> = data.into_iter().collect();
		reference_trie::compare_index_calc(data, change, memdb, &mut indexes, &indexes_conf, None);
	}
}

type Item = trie_db::partial_db::Item<Vec<u8>>;

#[test]
fn test_fix_set_root_iter() {
	type Expected = (Vec<u8>, usize, Item);
	fn check_expected(expected: Option<Expected>, value: (NibbleVec, Item)) {
		let expected = expected.expect("to many iterator values");
		assert_eq!(LeftNibbleSlice::new_len(expected.0.as_slice(), expected.1), value.0.as_slice());
		match (expected.2, value.1) {
			(Item::Index(_), Item::Index(_)) => (),
			(Item::Value(a), Item::Value(b)) => assert_eq!(a, b),
			(Item::StoredValue(a), Item::StoredValue(b)) => assert_eq!(a, b),
			_ => panic!("Mismatch item"),
		}
	}

	fn check(
		set: crate::iter_build::IndexingSet,
		mut expected: Vec<Expected>,
		mut changes: Vec::<(Vec<u8>, Option<Vec<u8>>)>,
	) {
		let (mem_db, indexes, indexes_conf) = set;
		expected.reverse();
		let mut deleted_indexes = Vec::new();
		let mut deleted_values = Vec::new();
		let mut root_iter = RootIndexIterator2::new(
			&mem_db,
			&indexes,
			&indexes_conf,
			changes.into_iter(), // TODO change api to pass IntoIter as param!!!
			&mut deleted_indexes,
			&mut deleted_values,
		);

		while let Some(iter) = root_iter.next() {
			check_expected(expected.pop(), iter);
		}
		assert!(expected.is_empty());
/*		while let Some(item) = root_iter.next() {
			println!("{:?}", item);
		}
		panic!("disp");*/
	}

	// Set1 : unique index

	let set = crate::iter_build::indexing_set_1(Default::default());
	let expected_set_1 = vec![
		(vec![97, 108, 102], 6, Item::Index(Default::default())),
		(vec![98, 114, 97], 6, Item::Index(Default::default())),
		(vec![100, 111, 103], 6, Item::Index(Default::default())),
		(vec![104, 111, 114], 6, Item::Index(Default::default())),
		(vec![104, 111, 117], 6, Item::Index(Default::default())),
	];
	check(set, expected_set_1.clone(), Default::default());

	let set = crate::iter_build::indexing_set_1(vec![
		(b"d".to_vec(), vec![2; 32]),
		(b"er".to_vec(), vec![3; 32]),
		(b"housf".to_vec(), vec![6; 32]),
		(b"i".to_vec(), vec![6; 32]),
	]);
	let expected = vec![
		(vec![97, 108, 102], 6, Item::Index(Default::default())),
		(vec![98, 114, 97], 6, Item::Index(Default::default())),
		(vec![100], 2, Item::StoredValue(vec![2; 32])),
		(vec![100, 111, 103], 6, Item::Index(Default::default())),
		(vec![101, 114], 4, Item::StoredValue(vec![3; 32])),
		(vec![104, 111, 114], 6, Item::Index(Default::default())),
		(vec![104, 111, 117], 6, Item::Index(Default::default())),
		(vec![105], 2, Item::StoredValue(vec![6; 32])),
	];
	check(set, expected, Default::default());

	let set = crate::iter_build::indexing_set_1(Default::default());
	let changes = vec![
		(b"alfa".to_vec(), Some(vec![1; 32])),
	];
	let mut expected = expected_set_1.clone();
	// single value instead of index
	expected[0] = (b"alfa".to_vec(), 8, Item::Value(vec![1; 32]));
	check(set.clone(), expected, changes);

	// insert before index, break partial.
	let changes = vec![
		(b"do".to_vec(), Some(vec![1; 32])),
	];
	let mut expected = expected_set_1.clone();
	expected[2] = (b"do".to_vec(), 4, Item::Value(vec![1; 32]));
	expected.insert(3, (b"dog".to_vec(), 6, Item::StoredValue(vec![3; 32])));
	expected.insert(4, (b"doge".to_vec(), 8, Item::StoredValue(vec![4; 32])));
	check(set.clone(), expected, changes);

	// insert at index
	let changes = vec![
		(b"dog".to_vec(), Some(vec![1; 32])),
	];
	let mut expected = expected_set_1.clone();
	expected[2] = (b"dog".to_vec(), 6, Item::Value(vec![1; 32]));
	expected.insert(3, (b"doge".to_vec(), 8, Item::StoredValue(vec![4; 32])));
	check(set.clone(), expected, changes);

	// insert after index at child
	let changes = vec![
		(b"doge".to_vec(), Some(vec![1; 32])),
	];
	let mut expected = expected_set_1.clone();
	expected[2] = (b"dog".to_vec(), 6, Item::StoredValue(vec![3; 32]));
	expected.insert(3, (b"doge".to_vec(), 8, Item::Value(vec![1; 32])));
	check(set.clone(), expected, changes);


	// insert after index with child
	let changes = vec![
		(b"hous".to_vec(), Some(vec![8; 32])),
	];
	let mut expected = expected_set_1.clone();
	expected[4] = (b"hous".to_vec(), 8, Item::Value(vec![8; 32]));
	expected.push((b"house".to_vec(), 10, Item::StoredValue(vec![6; 32])));
	check(set.clone(), expected, changes);

	// remove value after ix parent do not change
	let changes = vec![
		(b"bravo".to_vec(), None),
	];
	let mut expected = expected_set_1.clone();
	expected.remove(1);
	//panic!("{:?}", expected);
	check(set.clone(), expected, changes);

	// remove value after ix parent merge (removing a child from a two child branch without value).
	let changes = vec![
		(b"horse".to_vec(), None),
	];
	let mut expected = expected_set_1.clone();
	expected.remove(3);
	expected[3] = (b"house".to_vec(), 10, Item::StoredValue(vec![6; 32]));
	//panic!("{:?}", expected);
	check(set.clone(), expected, changes);

	// TODO same test but removing house

	// TODO test with insert in prefix of a value: aa (iter stored a), ba (iter stored b)
	// and also in prefix post of a value: am, bs.

	// TODO test removing doge then parent got value so no need to fuse branch: do  it on set 2!!
	
	// Set 2: Two indexes 

	let set = crate::iter_build::indexing_set_2(Default::default());
	let expected_set_2 = vec![
		(vec![97, 108], 4, Item::Index(Default::default())),
		(vec![98, 114], 4, Item::Index(Default::default())),
		(vec![100, 111], 4, Item::Index(Default::default())),
		(vec![104, 111], 4, Item::Index(Default::default())),
	];
	check(set.clone(), expected_set_2.clone(), Default::default());

	// insert between index and empty second ix
	let changes = vec![
		(b"alf".to_vec(), Some(vec![9; 32])),
	];
	let mut expected = expected_set_2.clone();
	expected[0] = (b"alf".to_vec(), 6, Item::Value(vec![9; 32]));
	expected.insert(1, (b"alfa".to_vec(), 8, Item::StoredValue(vec![0; 32])));
	check(set.clone(), expected, changes);

	// insert at first index with senconcd
	let changes = vec![
		(b"dog".to_vec(), Some(vec![9; 32])),
	];
	let mut expected = expected_set_2.clone();
	expected[2] = (b"dog".to_vec(), 6, Item::Value(vec![9; 32]));
	expected.insert(3, (b"doge".to_vec(), 8, Item::Index(Default::default())));
	check(set.clone(), expected, changes);

	// insert between first index and second
	let changes = vec![
		(b"hor".to_vec(), Some(vec![9; 32])),
	];
	let mut expected = expected_set_2.clone();
	expected[3] = (b"hor".to_vec(), 6, Item::Value(vec![9; 32]));
	expected.insert(4, (b"horse".to_vec(), 10, Item::StoredValue(vec![5; 32])));
	expected.insert(5, (b"hous".to_vec(), 8, Item::Index(Default::default())));
	check(set.clone(), expected, changes);
}
