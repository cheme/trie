#![no_main]

use trie_db_fuzz::fuzz_indexing_root_calc;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
		fuzz_indexing_root_calc(data, None, true);
});
