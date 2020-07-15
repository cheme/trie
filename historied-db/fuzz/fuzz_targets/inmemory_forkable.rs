#![no_main]
use libfuzzer_sys::fuzz_target;
fuzz_target!(|data: &[u8]| {
	historied_db::test::fuzz::inmemory_forkable(data, false, true)
});
