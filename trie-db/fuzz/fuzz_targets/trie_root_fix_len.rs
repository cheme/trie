

#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate trie_db;
extern crate memory_db;
extern crate reference_trie;
use memory_db::MemoryDB;
use reference_trie::{RefTrieDBMut, ref_trie_root,};
use trie_db::{TrieMut};

fn fuzz_to_data(input: &[u8]) -> Vec<(Vec<u8>,Vec<u8>)> {
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
      result.push((key,val));
    }
    result
  }

  fn data_sorted_unique(input: Vec<(Vec<u8>,Vec<u8>)>) -> Vec<(Vec<u8>,Vec<u8>)> {
    let mut m = std::collections::BTreeMap::new();
    for (k,v) in input.into_iter() {
      let _  = m.insert(k,v); // latest value for uniqueness
    }
    m.into_iter().collect()
  }
	fn fuzz_that(input: &[u8]) {
    let data = data_sorted_unique(fuzz_to_data(input));
    let mut memdb = MemoryDB::default();
		let mut root = Default::default();
		let mut t = RefTrieDBMut::new(&mut memdb, &mut root);
    for a in 0..data.len() {
		  t.insert(&data[a].0[..], &data[a].1[..]).unwrap();
    }
		assert_eq!(*t.root(), ref_trie_root(data));
	}



fuzz_target!(|data: &[u8]| {
    // fuzzed code goes here
    fuzz_that(data);
});
