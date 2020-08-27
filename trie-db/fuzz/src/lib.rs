// Copyright 2019 Parity Technologies
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


use hash_db::Hasher;
use keccak_hasher::KeccakHasher;
use memory_db::{HashKey, MemoryDB, PrefixedKey};
use reference_trie::{
	calc_root_no_extension,
	calc_root_no_extension2,
	calc_root_build_no_extension2,
	compare_no_extension_insert_remove,
	ExtensionLayout,
	NoExtensionLayout,
	proof::{generate_proof, verify_proof},
	reference_trie_root,
	RefTrieDBMut,
	RefTrieDBMutNoExt,
	RefTrieDBNoExt,
	TrieDBIterator,
};
use std::convert::TryInto;
use trie_db::{DBValue, Trie, TrieDB, TrieDBMut, TrieLayout, TrieMut};

fn fuzz_to_data(input: &[u8]) -> Vec<(Vec<u8>,Vec<u8>)> {
	let mut result = Vec::new();
	// enc = (minkeylen, maxkeylen (min max up to 32), datas)
	// fix data len 2 bytes
	let mut minkeylen = if let Some(v) = input.get(0) {
		let mut v = *v & 31u8;
		v = v + 1;
		v
	} else { return result; };
	let mut maxkeylen = if let Some(v) = input.get(1) {
		let mut v = *v & 31u8;
		v = v + 1;
		v
	} else { return result; };

	if maxkeylen < minkeylen {
		let v = minkeylen;
		minkeylen = maxkeylen;
		maxkeylen = v;
	}
	let mut ix = 2;
	loop {
		let keylen = if let Some(v) = input.get(ix) {
			let mut v = *v & 31u8;
			v = v + 1;
			v = std::cmp::max(minkeylen, v);
			v = std::cmp::min(maxkeylen, v);
			v as usize
		} else { break };
		let key = if input.len() > ix + keylen {
			input[ix..ix+keylen].to_vec()
		} else { break };
		ix += keylen;
		let val = if input.len() > ix + 2 {
			input[ix..ix+2].to_vec()
		} else { break };
		result.push((key,val));
	}
	result
}

fn fuzz_removal(data: Vec<(Vec<u8>,Vec<u8>)>) -> Vec<(bool, Vec<u8>,Vec<u8>)> {
	let mut res = Vec::new();
	let mut existing = None;
	for (a, d) in data.into_iter().enumerate() {
		if existing == None {
			existing = Some(a%2);
		}
		if existing.unwrap() == 0 {
			if a % 9 == 6
			|| a % 9 == 7
			|| a % 9 == 8 {
				// a random removal some time
				res.push((true, d.0, d.1));
				continue;
			}
		}
		res.push((false, d.0, d.1));
	}
	res
}

pub fn fuzz_that_reference_trie_root(input: &[u8]) {
	let data = data_sorted_unique(fuzz_to_data(input));
	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
	for a in 0..data.len() {
		t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
	}
	assert_eq!(*t.root(), reference_trie_root(data));
}

pub fn fuzz_that_reference_trie_root_fix_length(input: &[u8]) {
	let data = data_sorted_unique(fuzz_to_data_fix_length(input));
	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
	for a in 0..data.len() {
		t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
	}
	assert_eq!(*t.root(), reference_trie_root(data));
}

fn fuzz_to_data_fix_length(input: &[u8]) -> Vec<(Vec<u8>,Vec<u8>)> {
	let mut result = Vec::new();
	let mut ix = 0;
	loop {
		let keylen = 32;
		let key = if input.len() > ix + keylen {
			input[ix..ix+keylen].to_vec()
		} else { break };
		ix += keylen;
		let val = if input.len() > ix + 2 {
			input[ix..ix+2].to_vec()
		} else { break };
		result.push((key, val));
	}
	result
}

fn data_sorted_unique<V>(input: Vec<(Vec<u8>, V)>) -> Vec<(Vec<u8>, V)> {
	let mut m = std::collections::BTreeMap::new();
	for (k, v) in input.into_iter() {
		let _ = m.insert(k, v); // latest value for uniqueness
	}
	m.into_iter().collect()
}

fn data_sorted_unique_extend(input: Vec<(Vec<u8>, Vec<u8>)>) -> Vec<(Vec<u8>, Vec<u8>)> {
	let mut m = std::collections::BTreeMap::new();
	for (k, mut v) in input.into_iter() {
		v.resize(32, v[0]);
		let _ = m.insert(k, v); // latest value for uniqueness
	}
	m.into_iter().collect()
}


pub fn fuzz_that_compare_implementations(input: &[u8]) {
	let data = data_sorted_unique(fuzz_to_data(input));
	//println!("data:{:?}", &data);
	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let hashdb = MemoryDB::<KeccakHasher, PrefixedKey<_>, DBValue>::default();
	reference_trie::compare_implementations(data, memdb, hashdb);
}

pub fn fuzz_that_unhashed_no_extension(input: &[u8]) {
	let data = data_sorted_unique(fuzz_to_data(input));
	reference_trie::compare_unhashed_no_extension(data);
}

pub fn fuzz_that_no_extension_insert(input: &[u8]) {
	let data = fuzz_to_data(input);
	//println!("data{:?}", data);
	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
	for a in 0..data.len() {
		t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
	}
	// we are testing the RefTrie code here so we do not sort or check uniqueness
	// before.
	let data = data_sorted_unique(fuzz_to_data(input));
	//println!("data{:?}", data);
	assert_eq!(*t.root(), calc_root_no_extension(data));
}

pub fn fuzz_that_no_extension_insert2(input: &[u8]) {
	let data = fuzz_to_data(input);
	//println!("data{:?}", data);
	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
	for a in 0..data.len() {
		t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
	}
	// we are testing the RefTrie code here so we do not sort or check uniqueness
	// before.
	let data = data_sorted_unique(fuzz_to_data(input));
	//println!("data{:?}", data);
	assert_eq!(*t.root(), calc_root_no_extension2(data));
}

pub fn fuzz_that_no_extension_insert3(input: &[u8]) {
	let data = fuzz_to_data(input);
	//println!("data{:?}", data);
	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	let ref_root = {
		let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
		for a in 0..data.len() {
			let mut v = data[a].1.clone();
			v.resize(32, v[0]);
			t.insert(&data[a].0[..], &v[..]).unwrap();
		}
		t.root().clone()
	};
	// we are testing the RefTrie code here so we do not sort or check uniqueness
	// before.
	let data = data_sorted_unique(fuzz_to_data(input));
	//println!("data{:?}", data);
	let mut memdb2 = MemoryDB::<_, PrefixedKey<_>, _>::default();
	assert_eq!(ref_root, calc_root_build_no_extension2(data, &mut memdb2));
	let mut error = 0;
	{
		let trie = RefTrieDBNoExt::new(&memdb2, &ref_root).unwrap();
		for x in trie.iter().unwrap() {
			if x.is_err() {
				error +=1;
			}
		}
	}
	if error != 0 {
		{
			let trie = RefTrieDBNoExt::new(&memdb, &ref_root).unwrap();
			println!("ok: {:?}", trie);
		}

		{
			let trie = RefTrieDBNoExt::new(&memdb2, &ref_root).unwrap();
			println!("ko: {:?}", trie);
		}
	}
	assert_eq!(error, 0);

}


pub fn fuzz_that_no_extension_insert_remove(input: &[u8]) {
	let data = fuzz_to_data(input);
	let data = fuzz_removal(data);

	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	compare_no_extension_insert_remove(data, memdb);
}

pub fn fuzz_seek_iter(input: &[u8]) {
	let data = data_sorted_unique(fuzz_to_data_fix_length(input));

	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	{
		let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
		for a in 0..data.len() {
			t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
		}
	}

	// fuzzing around a fix prefix of 6 nibble.
	let prefix = &b"012"[..];

	let mut iter_res2 = Vec::new();
	for a in data {
		if a.0.starts_with(prefix) {
			iter_res2.push(a.0);
		}
	}

	let mut iter_res = Vec::new();
	let mut error = 0;
	{
			let trie = RefTrieDBNoExt::new(&memdb, &root).unwrap();
			let mut iter =  trie.iter().unwrap();
			if let Ok(_) = iter.seek(prefix) {
			} else {
				error += 1;
			}

			for x in iter {
				if let Ok((key, _)) = x {
				if key.starts_with(prefix) {
					iter_res.push(key);
				} else {
					break;
				}
				} else {
					error +=1;
				}
			}
	}

	assert_eq!(iter_res, iter_res2);
	assert_eq!(error, 0);
}

pub fn fuzz_prefix_iter(input: &[u8]) {
	let data = data_sorted_unique(fuzz_to_data_fix_length(input));

	let mut memdb = MemoryDB::<_, HashKey<_>, _>::default();
	let mut root = Default::default();
	{
		let mut t = RefTrieDBMutNoExt::new(&mut memdb, &mut root);
		for a in 0..data.len() {
			t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
		}
	}

	// fuzzing around a fix prefix of 6 nibble.
	let prefix = &b"012"[..];

	let mut iter_res2 = Vec::new();
	for a in data {
		if a.0.starts_with(prefix) {
			iter_res2.push(a.0);
		}
	}

	let mut iter_res = Vec::new();
	let mut error = 0;
	{
			let trie = RefTrieDBNoExt::new(&memdb, &root).unwrap();
			let iter = TrieDBIterator::new_prefixed(&trie, prefix).unwrap();

			for x in iter {
				if let Ok((key, _)) = x {
				if key.starts_with(prefix) {
					iter_res.push(key);
				} else {
					println!("error out of range");
					error +=1;
				}
				} else {
					error +=1;
				}
			}
	}

	assert_eq!(iter_res, iter_res2);
	assert_eq!(error, 0);
}

pub fn fuzz_that_verify_accepts_valid_proofs(input: &[u8]) {
	let mut data = fuzz_to_data(input);
	// Split data into 3 parts:
	// - the first 1/3 is added to the trie and not included in the proof
	// - the second 1/3 is added to the trie and included in the proof
	// - the last 1/3 is not added to the trie and the proof proves non-inclusion of them
	let mut keys = data[(data.len() / 3)..]
		.iter()
		.map(|(key, _)| key.clone())
		.collect::<Vec<_>>();
	data.truncate(data.len() * 2 / 3);

	let data = data_sorted_unique(data);
	keys.sort();
	keys.dedup();

	let (root, proof, items) = test_generate_proof::<ExtensionLayout>(data, keys);
	assert!(verify_proof::<ExtensionLayout, _, _, _>(&root, &proof, items.iter()).is_ok());
}

pub fn fuzz_that_verify_rejects_invalid_proofs(input: &[u8]) {
	if input.len() < 4 {
		return;
	}

	let random_int = u32::from_le_bytes(
		input[0..4].try_into().expect("slice is 4 bytes")
	) as usize;

	let mut data = fuzz_to_data(&input[4..]);
	// Split data into 3 parts:
	// - the first 1/3 is added to the trie and not included in the proof
	// - the second 1/3 is added to the trie and included in the proof
	// - the last 1/3 is not added to the trie and the proof proves non-inclusion of them
	let mut keys = data[(data.len() / 3)..]
		.iter()
		.map(|(key, _)| key.clone())
		.collect::<Vec<_>>();
	data.truncate(data.len() * 2 / 3);

	let data = data_sorted_unique(data);
	keys.sort();
	keys.dedup();

	if keys.is_empty() {
		return;
	}

	let (root, proof, mut items) = test_generate_proof::<ExtensionLayout>(data, keys);

	// Make one item at random incorrect.
	let items_idx = random_int % items.len();
	match &mut items[items_idx] {
		(_, Some(value)) if random_int % 2 == 0 => value.push(0),
		(_, value) if value.is_some() => *value = None,
		(_, value) => *value = Some(DBValue::new()),
	}
	assert!(verify_proof::<ExtensionLayout, _, _, _>(&root, &proof, items.iter()).is_err());
}

pub fn fuzz_indexing_root_calc(input: &[u8], indexes: Option<reference_trie::DepthIndexes>, with_rem: bool, big_values: bool) {
	if input.len() < 7 {
		return;
	}
	let indexes_conf = indexes.unwrap_or_else(|| {
		let mut depth_indexes = Vec::new();
		let nb = input[0] % 3 + 1;
		for i in 0..nb {
			 // runing on small depth, no 0 as implicit
			depth_indexes.push((input[(i + 1) as usize] % 32) as u32 + 1);
		}
		depth_indexes.sort();
		reference_trie::DepthIndexes::new(&depth_indexes[..])
	});
	let split = input[5] as usize;
	let rem = input[6] as usize;
	let mut data = fuzz_to_data(&input[5..]);
	if data.len() == 0 {
		return;
	}
	let split = split % data.len();
	let splitted = data.len() - split;
	let rem = if with_rem && splitted > 0 {
		rem % splitted
	} else {
		0
	};
	let mut change: Vec<(Vec<u8>, Option<Vec<u8>>)> = data[split..].iter().enumerate().map(|(i, (k, v))| {
		if i <= data.len() - rem {
			(k.clone(), Some(v.clone()))
		} else {
			(k.clone(), None) // this is dropping bits in a way that is probably bad for fuzzer 
		}
	}).collect();
	data.truncate(split);

	if with_rem {
		// also remove existing node
		let to_rem_existing = rem / 2;
		if data.len() > to_rem_existing {
			for i in 0..to_rem_existing {
				change.push((data[i].0.clone(), None));
			}
		}
	}

	let memdb = MemoryDB::<_, PrefixedKey<_>, _>::default();
	let mut indexes = std::collections::BTreeMap::new();
	data = if big_values {
		data_sorted_unique_extend(data)
	} else {
		data_sorted_unique(data)
	};
	change = data_sorted_unique(change);
	reference_trie::compare_index_calc(data, change, memdb, &mut indexes, &indexes_conf, None);
}

fn test_generate_proof<L: TrieLayout>(
	entries: Vec<(Vec<u8>, Vec<u8>)>,
	keys: Vec<Vec<u8>>,
) -> (<L::Hash as Hasher>::Out, Vec<Vec<u8>>, Vec<(Vec<u8>, Option<DBValue>)>)
{
	// Populate DB with full trie from entries.
	let (db, root) = {
		let mut db = <MemoryDB<L::Hash, HashKey<_>, _>>::default();
		let mut root = Default::default();
		{
			let mut trie = <TrieDBMut<L>>::new(&mut db, &mut root);
			for (key, value) in entries {
				trie.insert(&key, &value).unwrap();
			}
		}
		(db, root)
	};

	// Generate proof for the given keys..
	let trie = <TrieDB<L>>::new(&db, &root).unwrap();
	let proof = generate_proof::<_, L, _, _>(&trie, keys.iter()).unwrap();
	let items = keys.into_iter()
		.map(|key| {
			let value = trie.get(&key).unwrap();
			(key,value)
		})
		.collect();

	(root, proof, items)
}

#[test]
fn test_failure1() {
	let data = [
		vec![0x11,0x0,0xa,0x41,0x80,0x80,0x0,0x41,0xfd,0x7,0x9,0xc,0x1,0xa,0x0,0x79,0xd2,0x3b,
			0x1,0x1,0x0,0x1,0xa,0x1,0x9,],
		vec![0x1,0x0,0x80,0x60,0x20,0x0,0x60,0x20,0x0,0xff,0xfc,0x0,0x80,0xff,0xff,0xfc,0x9a],
		vec![0x1,0xfb,0x0,0xff,0x32,0x20,0x0,0x0,0x20,0xe,0x3d,0xec,0x0,],
		vec![0xf5,0xe4,0x41,0x40,0x80,0x80,0x80,0x0,0x1,0x0,0x80,0x0,0xff,0xff,0x1,0x80,0xf5,],
		vec![0x18,0x0,0x0,0x0,0x0,0x20,0x80,0x41,0xff,0xff,0xff,0xff,0x1,0x0,0xa,0xa,0xa,],
		vec![0xa6,0xff,0x27,0xf7,0x3,0x1,0x0,0xff,0x27,0xf7,0xff,0xc5,0xc5,0xee,],
		vec![0xff,0xff,0xff,0x0,0xe3,0x80,0xe9,0x0,0x80,0x0,0x0,0x80,0x0,0x0,0x80,0x0,0xf0],
		vec![0x1,0xfb,0x0,0xff,0x32,0x20,0x0,0x0,0x20,0xe,0x3d,0xec,0x0,],
		vec![0x10,0xd,0x0,0x0,0x0,0x0,0x20,0x27,0x0,0xe,0x0,0x0,0x0,],
		vec![0xc,0x16,0xa,0xa,0x0,0x0,0x80,0x0,0xff,0xa,0x0,0xe,0xff,],
	];
	for data in data.iter() {
		fuzz_indexing_root_calc(data.as_slice(), None, true, false);
		fuzz_indexing_root_calc(data.as_slice(), None, true, true);
	}
}

#[test]
fn test_failure2() {
	let data = [
		vec![0x0,0x0,0x4,0x0,0x40,0x4,0x0],
	];
	for data in data.iter() {
		fuzz_that_no_extension_insert2(data.as_slice());
	}
}

#[test]
fn test_failure3() {
	let data = [
		vec![0x0,0x0,0xba,0x9,0x0,0x19,0x28,],
		vec![0x0,0x0,0xba,0x9,0x0,0x28,],
	];
	for data in data.iter() {
		fuzz_that_no_extension_insert3(data.as_slice());
	}
}

#[test]
fn test_failure4() {
	let data = [
		vec![0x0,0x0,0x98,0x0,0x0,0x20,0xfe,0xfe,0xfe,0xfe,0xfe,0xfe,0xfe,0x0,0x9f,0x0,0x0,0x60,0x0,0x0,0x0,0x20,0xfe,0xfe,0xfe,0xfe,0x1,0xff,0xfe,0xfe,0xfe,0x0,0x9f,0x0,0x0,0x40,0x2c,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x36,0x0,0x0,0x0,0x0,0xe2,0x1],
	];
	for data in data.iter() {
		fuzz_indexing_root_calc(
			data,
			Some(reference_trie::DepthIndexes::new(&[2])),
			true,
			false,
		);
	}
}
