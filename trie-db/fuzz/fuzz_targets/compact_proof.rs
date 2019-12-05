#![no_main]

use trie_db_fuzz::fuzz_proof_reduction;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
	fuzz_proof_reduction(data);
});
