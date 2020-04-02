// Copyright 2020, 2020 Parity Technologies
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

//! Fuzz code for external fuzzer. Managing it in the
//! project allow to run non regression test over previously
//! problematic fuzzer inputs.

use crate::{
	Management, StateDB, ForkableManagement, ManagementRef, StateDBRef,
};
use crate::test::simple_impl::StateInput;

type InMemoryMgmt = crate::historied::tree_management::TreeManagement<StateInput, usize, usize, u16>;
struct FuzzerState {
	/// in memory historied datas to test
	in_memory_db: crate::historied::BTreeMap<Vec<u8>, u16, crate::historied::tree::MemoryOnly<usize, usize, u16>>,
	/// in memory state management
	in_memory_mgmt: InMemoryMgmt,
	/// simple reference
	simple: crate::test::simple_impl::Db<Vec<u8>, u16>, 
	/// limit of state to u8, hash from 1 to 255 are valid.
	next_hash: u8, // TODO rename next state
}

impl FuzzerState {
	fn new() -> Self {
		let mut in_memory_mgmt = InMemoryMgmt::default();
		in_memory_mgmt.map_root_state(StateInput(0));
		FuzzerState {
			in_memory_db: crate::historied::BTreeMap::new(),
			in_memory_mgmt,
			simple: crate::test::simple_impl::Db::init().0,
			next_hash: 1,
		}
	}

	fn apply(&mut self, action: FuzzerAction) {
		match action {
			FuzzerAction::SetValueLatest(key, value) => self.set_value_latest(key, value),
			FuzzerAction::SetValueAt(key, value, at) => self.set_value_at(key, value, at),
			FuzzerAction::AppendLatest => self.append_latest(),
			FuzzerAction::AppendAt(at) => self.append_at(at),
			FuzzerAction::DropLatest => self.drop_latest(),
			FuzzerAction::DropAt(at) => self.drop_at(at),
		}
	}

	fn set_value_latest(&mut self, key: u8, value: u16) {
		let at_simple = self.simple.latest_state();
		self.simple.emplace(vec![key], value, &at_simple);
		let at = self.in_memory_mgmt.latest_state();
		self.in_memory_db.emplace(vec![key], value, &at);
	}

	fn set_value_at(&mut self, key: u8, value: u16, at: u8) {
		let state = StateInput((at % self.next_hash) as usize);
		let at_simple = self.simple.get_db_state_mut(&state);
		let at = self.in_memory_mgmt.get_db_state_mut(&state);
		assert_eq!(at.is_some(), at_simple.is_some());
		at_simple.map(|at_simple|
			self.simple.emplace(vec![key], value, &at_simple)
		);
		at.map(|at|
			self.in_memory_db.emplace(vec![key], value, &at)
		);
	}

	fn append_latest(&mut self) {
		if self.next_hash < NUMBER_POSSIBLE_STATES {
			let new_state = StateInput(self.next_hash as usize);
			let at_simple = self.simple.latest_state_fork();
			let s_simple = self.simple.append_external_state(new_state.clone(), &at_simple);
			let at = self.in_memory_mgmt.latest_state_fork();
			let s = self.in_memory_mgmt.append_external_state(new_state, &at);
			if s.is_some() {
				self.next_hash += 1;
			}
			assert_eq!(s.is_some(), s_simple.is_some());
		}
	}

	fn append_at(&mut self, at: u8) {
		if self.next_hash < NUMBER_POSSIBLE_STATES {
			let new_state = StateInput(self.next_hash as usize);
			// keep in range
			let at = StateInput((at % self.next_hash) as usize);
			let at_simple = self.simple.get_db_state_for_fork(&at);
			let at = self.in_memory_mgmt.get_db_state_for_fork(&at);
			assert_eq!(at.is_some(), at_simple.is_some());
			let s_simple = at_simple.and_then(|at| self.simple.append_external_state(new_state.clone(), &at));
			let s = at.and_then(|at| self.in_memory_mgmt.append_external_state(new_state, &at));
			if s.is_some() {
				self.next_hash += 1;
			}
			assert_eq!(s.is_some(), s_simple.is_some());
		}
	}

	fn drop_latest(&mut self) {
		let at_simple = self.simple.latest_state_fork();
		let mut dropped_simple = self.simple.drop_state(&at_simple, true);
		let at = self.in_memory_mgmt.latest_state_fork();
		let mut dropped = self.in_memory_mgmt.drop_state(&at, true);
		dropped_simple.as_mut().map(|d| d.sort());
		dropped.as_mut().map(|d| d.sort());
		assert_eq!(dropped, dropped_simple)
	}

	fn drop_at(&mut self, at: u8) {
		let at = StateInput((at % self.next_hash) as usize);
		let at_simple = self.simple.get_db_state_for_fork(&at);
		let at = self.in_memory_mgmt.get_db_state_for_fork(&at);
		assert_eq!(at.is_some(), at_simple.is_some());
		let mut dropped_simple = at_simple.and_then(|at| self.simple.drop_state(&at, true));
		let mut dropped = at.and_then(|at| self.in_memory_mgmt.drop_state(&at, true));
		dropped_simple.as_mut().map(|d| d.sort());
		dropped.as_mut().map(|d| d.sort());
		assert_eq!(dropped, dropped_simple)
	}

	fn compare(&self) {
		for state in 0..self.next_hash {
			let state = StateInput(state as usize);
			let query_simple = self.simple.get_db_state(&state);
			let query = self.in_memory_mgmt.get_db_state(&state);
			assert_eq!(query.is_some(), query_simple.is_some());
			if query.is_some() {
				let query = query.unwrap();
				let query_simple = query_simple.unwrap();
				for key in 0..NUMBER_POSSIBLE_KEYS {
					let value_simple = self.simple.get(&vec![key], &query_simple);
					let value = self.in_memory_db.get(&vec![key], &query);
					assert_eq!(value, value_simple);
				}
			}
		}
	}
}

const NUMBER_POSSIBLE_STATES: u8 = 255;
const NUMBER_POSSIBLE_KEYS: u8 = 4;
const NUMBER_POSSIBLE_VALUES: u8 = 16;

#[derive(Debug)]
enum FuzzerAction {
	/// Key is u8 but it.
	SetValueLatest(u8, u16),
	/// Key is u8 but it.
	SetValueAt(u8, u16, u8),
	/// append state at latest.
	AppendLatest,
	/// append state at given state.
	AppendAt(u8),
	/// drop latest.
	DropLatest,
	/// drop from.
	DropAt(u8),
}

impl FuzzerAction {
	fn next_action(data: &mut &[u8]) -> Option<Self> {
		if data.len() == 0 {
			return None;
		}
		match data[0] % 6 {
			0 => {
				if data.len() < 3 {
					return None;
				}
				let result = FuzzerAction::SetValueLatest(
					data[1] % NUMBER_POSSIBLE_KEYS,
					(data[2] % NUMBER_POSSIBLE_VALUES) as u16,
				);
				*data = &data[3..];
				Some(result)
			},
			1 => {
				if data.len() < 4 {
					return None;
				}
				let result = FuzzerAction::SetValueAt(
					data[1] % NUMBER_POSSIBLE_KEYS,
					(data[2] % NUMBER_POSSIBLE_VALUES) as u16,
					data[3],
				);
				*data = &data[4..];
				Some(result)
			},
			2 => {
				*data = &data[1..];
				Some(FuzzerAction::AppendLatest)
			},
			3 => {
				if data.len() < 2 {
					return None;
				}
				let result = FuzzerAction::AppendAt(data[1]);
				*data = &data[2..];
				Some(result)
			},
			4 => {
				*data = &data[1..];
				Some(FuzzerAction::DropLatest)
			},
			5 => {
				if data.len() < 2 {
					return None;
				}
				let result = FuzzerAction::DropAt(data[1]);
				*data = &data[2..];
				Some(result)
			},
			_ => unreachable!("modulo above"),
		}
	}

	#[cfg(test)]
	/// For debugging purpose.
	fn into_actions(data: &[u8]) -> Vec<FuzzerAction> {
		let data = &mut &data[..];
		let mut actions = Vec::new();
		while let Some(action) = FuzzerAction::next_action(data) {
			actions.push(action);
		}
		actions
	}
}

/// Entry point for fuzzing in memory forkable scenario.
pub fn inmemory_forkable(data: &[u8]) {
	let mut fuzz_state = FuzzerState::new();
	let data = &mut &data[..];
	while let Some(action) = FuzzerAction::next_action(data) {
		fuzz_state.apply(action);
	}
	fuzz_state.compare();
}

#[test]
fn inmemory_forkable_no_regression() {
	let inputs = [
		&[][..],
		&[32, 50, 244, 0][..],
		&[32, 5, 0, 65][..],
		&[30, 65, 161][..],
		&[181, 226, 244, 157][..],
		&[219, 50, 32, 50][..],
		&[242, 7, 4, 2, 117, 125][..],
	];
	for input in inputs.iter() {
		println!("{:?}", FuzzerAction::into_actions(input));
		inmemory_forkable(input);
	}
}
