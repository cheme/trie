#![no_main]

use trie_db_fuzz::fuzz_that_no_extension_insert2;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
	// fuzzed code goes here
	fuzz_that_no_extension_insert2(data);
});
