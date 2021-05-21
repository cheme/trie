// Copyright 2017, 2018 Parity Technologies
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

//! Reference implementation of a streamer.

use std::fmt;
use std::iter::once;
use std::marker::PhantomData;
use std::ops::Range;
use parity_scale_codec::{Decode, Input, Output, Encode, Compact, Error as CodecError};
use trie_root::{Hasher, MetaHasher};

use trie_db::{
	node::{NibbleSlicePlan, NodePlan, Value, ValuePlan, NodeHandlePlan},
	triedbmut::ChildReference,
	DBValue,
	trie_visit,
	TrieBuilder,
	TrieRoot,
	Partial,
	Meta,
	ChildrenDecoded,
	GlobalMeta,
};
use std::borrow::Borrow;

use trie_db::{
	nibble_ops, NodeCodec,
	Trie, TrieConfiguration, TrieDB, TrieDBMut,
	TrieLayout, TrieMut,
};
pub use trie_root::TrieStream;
pub mod node {
	pub use trie_db::node::Node;
}

/// Reference hasher is a keccak hasher.
pub type RefHasher = keccak_hasher::KeccakHasher;

/// Apply a test method on every test layouts.
#[macro_export]
macro_rules! test_layouts {
	($test:ident, $test_internal:ident) => {
		#[test]
		fn $test() {
			$test_internal::<reference_trie::CheckMetaHasher>();
			$test_internal::<reference_trie::CheckMetaHasherNoExt>();
			$test_internal::<reference_trie::NoExtensionLayout>();
			$test_internal::<reference_trie::ExtensionLayout>();
		}
	};
}

/// Apply a test method on every test layouts.
#[macro_export]
macro_rules! test_layouts_no_meta {
	($test:ident, $test_internal:ident) => {
		#[test]
		fn $test() {
			$test_internal::<reference_trie::NoExtensionLayout>();
			$test_internal::<reference_trie::ExtensionLayout>();
		}
	};
}


/// Trie layout using extension nodes.
#[derive(Default, Clone)]
pub struct ExtensionLayout;

impl TrieLayout for ExtensionLayout {
	const USE_EXTENSION: bool = true;
	const ALLOW_EMPTY: bool = false;
	type Hash = RefHasher;
	type Codec = ReferenceNodeCodec<RefHasher>;
	type MetaHasher = hash_db::NoMeta;
	type Meta = ();

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		()
	}
}

impl TrieConfiguration for ExtensionLayout { }

/// Trie layout without extension nodes, allowing
/// generic hasher.
pub struct GenericNoExtensionLayout<H>(PhantomData<H>);

impl<H> Default for GenericNoExtensionLayout<H> {
	fn default() -> Self {
		GenericNoExtensionLayout(PhantomData)
	}
}

impl<H> Clone for GenericNoExtensionLayout<H> {
	fn clone(&self) -> Self {
		GenericNoExtensionLayout(PhantomData)
	}
}

impl<H: Hasher> TrieLayout for GenericNoExtensionLayout<H> {
	const USE_EXTENSION: bool = false;
	const ALLOW_EMPTY: bool = false;
	type Hash = H;
	type Codec = ReferenceNodeCodecNoExt<H>;
	type MetaHasher = hash_db::NoMeta;
	type Meta = ();

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		()
	}
}

/// Trie that allows empty values.
#[derive(Default, Clone)]
pub struct AllowEmptyLayout;

impl TrieLayout for AllowEmptyLayout {
	const USE_EXTENSION: bool = true;
	const ALLOW_EMPTY: bool = true;
	type Hash = RefHasher;
	type Codec = ReferenceNodeCodec<RefHasher>;
	type MetaHasher = hash_db::NoMeta;
	type Meta = ();

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		()
	}
}

/// Trie that use a dumb value function over its storage.
/// TODO consider removal
#[derive(Default, Clone)]
pub struct CheckMetaHasher;

impl TrieLayout for CheckMetaHasher {
	const USE_EXTENSION: bool = true;
	const ALLOW_EMPTY: bool = false;
	const USE_META: bool = true;

	type Hash = RefHasher;
	type Codec = ReferenceNodeCodec<RefHasher>;
	type MetaHasher = TestMetaHasher<RefHasher>;
	type Meta = ValueRange;

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		false
	}
}

/// Trie that use a dumb value function over its storage.
#[derive(Default, Clone)]
pub struct CheckMetaHasherNoExt(pub bool);

impl TrieLayout for CheckMetaHasherNoExt {
	const USE_EXTENSION: bool = false;
	const ALLOW_EMPTY: bool = false;
	const USE_META: bool = true;
	const READ_ROOT_STATE_META: bool = true;

	type Hash = RefHasher;
	type Codec = ReferenceNodeCodecNoExt<RefHasher>;
	type MetaHasher = TestMetaHasher<RefHasher>;
	type Meta = ValueRange;

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		self.0
	}
	fn initialize_from_root_meta(&mut self, root_meta: &Self::Meta) {
		if root_meta.recorded_do_value_hash {
			self.0 = true;
		}
	}
	fn set_root_meta(root_meta: &mut Self::Meta, global_meta: GlobalMeta<Self>) {
		if global_meta {
			root_meta.recorded_do_value_hash = true;
		}
	}
}

/// Test value function: prepend optional encoded size of value
pub struct TestMetaHasher<H>(PhantomData<H>);

/// Test value function: prepend optional encoded size of value.
/// Also allow indicating that value is a hash of value.
pub struct TestMetaHasherProof<H>(PhantomData<H>);

impl<H: Hasher> hash_db::MetaHasher<H, DBValue> for TestMetaHasher<H> {
	type Meta = ValueRange;
	type GlobalMeta = bool;

	fn hash(value: &[u8], meta: &Self::Meta) -> H::Out {
		match &meta {
			ValueMeta { range: Some(range), contain_hash: false, do_value_hash, .. } => {
				if *do_value_hash {
					let value = inner_hashed_value::<H>(value, Some((range.start, range.end)));
					H::hash(value.as_slice())
				} else {
					H::hash(value)
				}
			},
			ValueMeta { range: Some(_range), contain_hash: true, .. } => {
				// value contains a hash of data (already inner_hashed_value).
				H::hash(value)
			},
			_ => {
				H::hash(value)
			},
		}
	}

	fn stored_value(value: &[u8], mut meta: Self::Meta) -> DBValue {
		let mut stored = meta.range.as_ref().map(|range| (range.start as u32, range.end as u32)).encode();
		if meta.contain_hash {
			// already contain hash, just flag it.
			stored.push(DEAD_HEADER_META_HASHED_VALUE);
			stored.extend_from_slice(value);
			return stored;
		}
		if meta.do_value_hash && meta.unused_value {
			if let Some(range) = meta.range.as_ref() {
				if range.end - range.start >= INNER_HASH_TRESHOLD {
					// Waring this assume that encoded value does not start by this, so it is tightly coupled
					// with the header type of the codec: only for optimization.
					stored.push(DEAD_HEADER_META_HASHED_VALUE);
					let range = meta.range.as_ref().expect("Tested in condition");
					meta.contain_hash = true; // useless but could be with meta as &mut
					// store hash instead of value.
					let value = inner_hashed_value::<H>(value, Some((range.start, range.end)));
					stored.extend_from_slice(value.as_slice());
					return stored;
				}
			}
		}
		stored.extend_from_slice(value);
		stored
	}

	fn stored_value_owned(value: DBValue, meta: Self::Meta) -> DBValue {
		Self::stored_value(value.as_slice(), meta)
	}

	fn extract_value(mut stored: &[u8], global_meta: Self::GlobalMeta) -> (&[u8], Self::Meta) {
		// handle empty trie optimisation.
		if stored == &[0] {
			let mut meta = Self::Meta::default();
			meta.do_value_hash = global_meta;
			return (stored, meta);
		}
		let input = &mut stored;
		let range: Option<(u32, u32)> = Decode::decode(input).ok().flatten();
		let mut contain_hash = false;
		if input.get(0) == Some(&DEAD_HEADER_META_HASHED_VALUE) {
			debug_assert!(range.is_some());
			contain_hash = true;
			*input = &input[1..];
		}
		let range = range.map(|range| range.0 as usize .. range.1 as usize);
		let mut meta = ValueMeta {
			range,
			unused_value: contain_hash,
			contain_hash,
			do_value_hash: false,
			recorded_do_value_hash: false,
		};
		// get recorded_do_value_hash
		let _offset = meta.read_state_meta(stored)
			.expect("State meta reading failure.");
		//let stored = &stored[offset..];
		meta.do_value_hash = meta.recorded_do_value_hash || global_meta;
		(stored, meta)
	}

	fn extract_value_owned(mut stored: DBValue, global_meta: Self::GlobalMeta) -> (DBValue, Self::Meta) {
		let len = stored.len();
		let (v, meta) = Self::extract_value(stored.as_slice(), global_meta);
		let removed = len - v.len();
		(stored.split_off(removed), meta)
	}
}

#[derive(Default, Clone)]
pub struct ValueMeta {
	pub range: Option<core::ops::Range<usize>>,
	// When `do_value_hash` is true, try to
	// store this behavior in top node
	// encoded (need to be part of state).
	pub recorded_do_value_hash: bool,
	// Does current encoded contains a hash instead of
	// a value (information stored in meta for proofs).
	pub contain_hash: bool,
	// Flag indicating if value hash can run.
	// When defined for a node it gets active
	// for all children node
	pub do_value_hash: bool,
	// Record if a value was accessed, this is
	// set as accessed by defalult, but can be
	// change on access explicitely: `HashDB::get_with_meta`.
	// and reset on access explicitely: `HashDB::access_from`.
	pub unused_value: bool,
}

/// Test Meta input. TODO remove alias
pub type ValueRange = ValueMeta;

/// Treshold for using hash of value instead of value
/// in encoded trie node.
pub const INNER_HASH_TRESHOLD: usize = 1;

impl Meta for ValueRange {
	/// If true apply inner hashing of value
	/// starting from this trie branch.
	type GlobalMeta = bool;

	/// If true apply inner hashing of value
	/// starting from this trie branch.
	type StateMeta = bool;

	fn set_state_meta(&mut self, state_meta: Self::StateMeta) {
		self.recorded_do_value_hash = state_meta;
		self.do_value_hash = state_meta;
	}

	fn has_state_meta(&self) -> bool {
		self.recorded_do_value_hash
	}

	fn read_state_meta(&mut self, data: &[u8]) -> Result<usize, &'static str> {
		let offset = if data[0] == ENCODED_META_NO_EXT {
			if data.len() < 2 {
				return Err("Invalid encoded meta.");
			}
			match data[1] {
				ALLOW_HASH_META => {
					self.recorded_do_value_hash = true;
					self.do_value_hash = true;
					2
				},
				_ => return Err("Invalid encoded meta."),
			}
		} else {
			0
		};
		Ok(offset)
	}

	fn write_state_meta(&self) -> Vec<u8> {
		if self.recorded_do_value_hash {
			// Note that this only works for some codecs (if the first byte do not overlay).
			[ENCODED_META_NO_EXT, ALLOW_HASH_META].to_vec()
		} else {
			Vec::new()
		}
	}

	fn meta_for_new(
		input: Self::GlobalMeta,
	) -> Self {
		let mut result = Self::default();
		result.do_value_hash = input;
		result
	}

	fn meta_for_existing_inline_node(
		input: Self::GlobalMeta,
	) -> Self {
		Self::meta_for_new(input)
	}

	fn meta_for_empty(
		input: Self::GlobalMeta,
	) -> Self {
		Self::meta_for_new(input)
	}

	fn encoded_value_callback(
		&mut self,
		value_plan: ValuePlan,
	) {
		let (contain_hash, range) = match value_plan {
			ValuePlan::Value(range) => (false, range),
			ValuePlan::HashedValue(range, _size) => (true, range),
			ValuePlan::NoValue => return,
		};

		self.range = Some(range);
		self.contain_hash = contain_hash;
	}

	fn decoded_callback(
		&mut self,
		_node_plan: &trie_db::node::NodePlan,
	) {
	}

	fn contains_hash_of_value(&self) -> bool {
		self.contain_hash
	}

	fn do_value_hash(&self) -> bool {
		self.unused_value
	}
}

impl ValueRange {
	pub fn set_accessed_value(&mut self, accessed: bool) {
		self.unused_value = !accessed;
	}
}

/// Representation with inner hash.
pub fn inner_hashed_value<H: Hasher>(x: &[u8], range: Option<(usize, usize)>) -> Vec<u8> {
	if let Some((start, end)) = range {
		let len = x.len();
		if start < len && end == len {
			// terminal inner hash
			let hash_end = H::hash(&x[start..]);
			let mut buff = vec![0; x.len() + hash_end.as_ref().len() - (end - start)];
			buff[..start].copy_from_slice(&x[..start]);
			buff[start..].copy_from_slice(hash_end.as_ref());
			return buff;
		}
		if start == 0 && end < len {
			// start inner hash
			let hash_start = H::hash(&x[..start]);
			let hash_len = hash_start.as_ref().len();
			let mut buff = vec![0; x.len() + hash_len - (end - start)];
			buff[..hash_len].copy_from_slice(hash_start.as_ref());
			buff[hash_len..].copy_from_slice(&x[end..]);
			return buff;
		}
		if start < len && end < len {
			// middle inner hash
			let hash_middle = H::hash(&x[start..end]);
			let hash_len = hash_middle.as_ref().len();
			let mut buff = vec![0; x.len() + hash_len - (end - start)];
			buff[..start].copy_from_slice(&x[..start]);
			buff[start..start + hash_len].copy_from_slice(hash_middle.as_ref());
			buff[start + hash_len..].copy_from_slice(&x[end..]);
			return buff;
		}
	}
	// if anything wrong default to hash
	x.to_vec()
}


#[derive(Clone, Copy)]
pub enum Version {
	Old,
	New,
}

impl Default for Version {
	fn default() -> Self {
		// freshly created nodes are New.
		Version::New
	}
}

/// previous layout in updatable scenario.
#[derive(Default, Clone)]
pub struct Old;

impl TrieLayout for Old {
	const USE_EXTENSION: bool = false;
	const ALLOW_EMPTY: bool = false;
	type Hash = RefHasher;
	type Codec = ReferenceNodeCodecNoExt<RefHasher>;
	type MetaHasher = hash_db::NoMeta;
	type Meta = ();

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		()
	}
}

/// Trie that use a dumb value function over its storage.
#[derive(Default, Clone)]
pub struct Updatable(Version);

impl Updatable {
	/// Old trie codec.
	pub fn old() -> Self {
		Updatable(Version::Old)
	}
	/// New trie codec.
	pub fn new() -> Self {
		Updatable(Version::New)
	}
}

impl TrieLayout for Updatable {
	const USE_EXTENSION: bool = false;
	const ALLOW_EMPTY: bool = false;
	const USE_META: bool = true;

	type Hash = RefHasher;
	type Codec = ReferenceNodeCodecNoExt<RefHasher>;
	type MetaHasher = TestUpdatableMetaHasher<RefHasher>;
	type Meta = VersionedValueRange;

	fn layout_meta(&self) -> <Self::Meta as Meta>::GlobalMeta {
		self.0
	}
}

/// Test Meta input.
#[derive(Default, Clone)]
pub struct VersionedValueRange {
	range: Option<core::ops::Range<usize>>,
	old_remaining_children: Option<Vec<u8>>,
	version: Version,
}

impl Meta for VersionedValueRange {
	type GlobalMeta = Version;

	type StateMeta = ();

	fn set_state_meta(&mut self, _state_meta: Self::StateMeta) {
	}

	fn has_state_meta(&self) -> bool {
		false
	}

	fn read_state_meta(&mut self, _data: &[u8]) -> Result<usize, &'static str> {
		Ok(0)
	}

	fn write_state_meta(&self) -> Vec<u8> {
		Vec::new()
	}

	fn meta_for_new(
		input: Self::GlobalMeta,
	) -> Self {
		let old_remaining_children = if matches!(input, Version::Old) {
			Some(Vec::new())
		} else {
			None
		};
		VersionedValueRange { range: None, version: input, old_remaining_children }
	}

	fn meta_for_existing_inline_node(
		input: Self::GlobalMeta,
	) -> Self {
		let old_remaining_children = if matches!(input, Version::Old) {
			Some(Vec::new())
		} else {
			None
		};
		VersionedValueRange { range: None, version: input, old_remaining_children }
	}

	fn meta_for_empty(
		input: Self::GlobalMeta,
	) -> Self {
		// empty is same for new and old, using new
		VersionedValueRange { range: None, version: input, old_remaining_children: None }
	}

	fn encoded_value_callback(
		&mut self,
		value_plan: ValuePlan,
	) {
		if matches!(self.version, Version::New) {
			let range = match value_plan {
				ValuePlan::Value(range) => range,
				ValuePlan::HashedValue(_range, _size) => unimplemented!(),
				ValuePlan::NoValue => return,
			};

			if range.end - range.start >= INNER_HASH_TRESHOLD {
				self.range = Some(range);
			}
		}
	}

	fn decoded_callback(
		&mut self,
		node_plan: &trie_db::node::NodePlan,
	) {
		if matches!(self.version, Version::Old) {
			if self.old_remaining_children.is_none() {
				let mut non_inline_children = Vec::new();
				for (index, child) in node_plan.inline_children().enumerate() {
					if matches!(child, ChildrenDecoded::Hash) {
						// overflow for radix > 256, ok with current hex trie only implementation.
						non_inline_children.push(index as u8);
					}
				}
				self.old_remaining_children = Some(non_inline_children);
			}
		}
	}

	fn contains_hash_of_value(&self) -> bool {
		false
	}

	fn do_value_hash(&self) -> bool {
		false
	}
}

/// Test value function: prepend optional encoded size of value
pub struct TestUpdatableMetaHasher<H>(PhantomData<H>);

impl<H: Hasher> hash_db::MetaHasher<H, DBValue> for TestUpdatableMetaHasher<H> {
	type Meta = VersionedValueRange;
	type GlobalMeta = Version;

	fn hash(value: &[u8], meta: &Self::Meta) -> H::Out {
		if matches!(meta.version, Version::New) {
			if let Some(range) = meta.range.as_ref() {
				assert!(matches!(meta.version,Version::New));
				let value = inner_hashed_value::<H>(value, Some((range.start, range.end)));
				return H::hash(value.as_slice());
			}
		}
		H::hash(value)
	}

	fn stored_value(value: &[u8], meta: Self::Meta) -> DBValue {
		if let Version::Old = meta.version {
			// non empty empty trie byte for old node
			let mut stored = Vec::with_capacity(value.len() + 20);
			stored.push(EMPTY_TRIE); // 1 byte
			stored.extend_from_slice(meta.old_remaining_children.encode().as_slice()); // max 18 byt
			stored.extend_from_slice(value);
			stored
		} else {
			value.to_vec()
		}
	}

	fn stored_value_owned(value: DBValue, meta: Self::Meta) -> DBValue {
		Self::stored_value(value.as_slice(), meta)
	}

	fn extract_value(mut stored: &[u8], _meta: Self::GlobalMeta) -> (&[u8], Self::Meta) {
		let len = stored.len();
		let input = &mut stored;
		// if len == 1 it is new empty trie.
		let (version, old_remaining_children) = if input[0] == EMPTY_TRIE && input.len() > 1 {
			*input = &input[1..];
			(Version::Old, Decode::decode(input).ok().flatten())
		} else {
			(Version::New, None)
		};
		let read_bytes = len - input.len();
		let stored = &stored[read_bytes..];
		(stored, VersionedValueRange {
			old_remaining_children,
			range: None,
			version,
		})
	}

	fn extract_value_owned(mut stored: DBValue, _meta: Self::GlobalMeta) -> (DBValue, Self::Meta) {
		// TODO factor with extract_value
		let len = stored.len();
		let input = &mut stored.as_slice();
		// if len == 1 it is new empty trie.
		let (version, old_remaining_children) = if input[0] == EMPTY_TRIE && input.len() > 1 {
			*input = &input[1..];
			(Version::Old, Decode::decode(input).ok().flatten())
		} else {
			(Version::New, None)
		};
		let read_bytes = len - input.len();
		let stored = stored.split_off(read_bytes);
		(stored, VersionedValueRange {
			old_remaining_children,
			range: None,
			version,
		})
	}
}

impl<H: Hasher> TrieConfiguration for GenericNoExtensionLayout<H> { }

/// Trie layout without extension nodes.
pub type NoExtensionLayout = GenericNoExtensionLayout<RefHasher>;

/// Children bitmap codec for radix 16 trie.
pub struct Bitmap(u16);

const BITMAP_LENGTH: usize = 2;

impl Bitmap {

	fn decode(data: &[u8]) -> Result<Self, CodecError> {
		Ok(u16::decode(&mut &data[..])
			.map(|v| Bitmap(v))?)
	}

	fn value_at(&self, i: usize) -> bool {
		self.0 & (1u16 << i) != 0
	}

	fn encode<I: Iterator<Item = bool>>(has_children: I , output: &mut [u8]) {
		let mut bitmap: u16 = 0;
		let mut cursor: u16 = 1;
		for v in has_children {
			if v { bitmap |= cursor }
			cursor <<= 1;
		}
		output[0] = (bitmap % 256) as u8;
		output[1] = (bitmap / 256) as u8;
	}
}

pub type RefTrieDB<'a> = trie_db::TrieDB<'a, ExtensionLayout>;
pub type RefTrieDBMut<'a> = trie_db::TrieDBMut<'a, ExtensionLayout>;
pub type RefTrieDBMutNoExt<'a> = trie_db::TrieDBMut<'a, NoExtensionLayout>;
pub type RefTrieDBMutAllowEmpty<'a> = trie_db::TrieDBMut<'a, AllowEmptyLayout>;
pub type RefFatDB<'a> = trie_db::FatDB<'a, ExtensionLayout>;
pub type RefFatDBMut<'a> = trie_db::FatDBMut<'a, ExtensionLayout>;
pub type RefSecTrieDB<'a> = trie_db::SecTrieDB<'a, ExtensionLayout>;
pub type RefSecTrieDBMut<'a> = trie_db::SecTrieDBMut<'a, ExtensionLayout>;
pub type RefLookup<'a, Q> = trie_db::Lookup<'a, ExtensionLayout, Q>;
pub type RefLookupNoExt<'a, Q> = trie_db::Lookup<'a, NoExtensionLayout, Q>;

pub fn reference_trie_root<T: TrieLayout, I, A, B>(input: I) -> <T::Hash as Hasher>::Out where
	I: IntoIterator<Item = (A, B)>,
	A: AsRef<[u8]> + Ord + fmt::Debug,
	B: AsRef<[u8]> + fmt::Debug,
{
	if T::USE_EXTENSION {
		trie_root::trie_root::<T::Hash, T::MetaHasher, ReferenceTrieStream, _, _, _>(input, Default::default())
	} else {
		trie_root::trie_root_no_extension::<T::Hash, T::MetaHasher, ReferenceTrieStreamNoExt, _, _, _>(input, Default::default())
	}
}

fn data_sorted_unique<I, A: Ord, B>(input: I) -> Vec<(A, B)>
	where
		I: IntoIterator<Item = (A, B)>,
{
	let mut m = std::collections::BTreeMap::new();
	for (k,v) in input {
		let _ = m.insert(k,v); // latest value for uniqueness
	}
	m.into_iter().collect()
}

pub fn reference_trie_root_iter_build<T, I, A, B>(input: I) -> <T::Hash as Hasher>::Out where
	T: TrieLayout,
	I: IntoIterator<Item = (A, B)>,
	A: AsRef<[u8]> + Ord + fmt::Debug,
	B: AsRef<[u8]> + fmt::Debug,
{
	let mut cb = trie_db::TrieRoot::<T>::default();
	trie_visit(data_sorted_unique(input), &mut cb, &T::default());
	cb.root.unwrap_or_default()
}

fn reference_trie_root_unhashed<I, A, B>(input: I) -> Vec<u8> where
	I: IntoIterator<Item = (A, B)>,
	A: AsRef<[u8]> + Ord + fmt::Debug,
	B: AsRef<[u8]> + fmt::Debug,
{
	trie_root::unhashed_trie::<RefHasher, hash_db::NoMeta, ReferenceTrieStream, _, _, _>(input, Default::default())
}

fn reference_trie_root_unhashed_no_extension<I, A, B>(input: I) -> Vec<u8> where
	I: IntoIterator<Item = (A, B)>,
	A: AsRef<[u8]> + Ord + fmt::Debug,
	B: AsRef<[u8]> + fmt::Debug,
{
	trie_root::unhashed_trie_no_extension::<RefHasher, TestMetaHasher<RefHasher>, ReferenceTrieStreamNoExt, _, _, _>(input, Default::default())
}

const EMPTY_TRIE: u8 = 0;
const LEAF_NODE_OFFSET: u8 = 1;
const EXTENSION_NODE_OFFSET: u8 = 128;
const BRANCH_NODE_NO_VALUE: u8 = 254;
const BRANCH_NODE_WITH_VALUE: u8 = 255;
const LEAF_NODE_OVER: u8 = EXTENSION_NODE_OFFSET - LEAF_NODE_OFFSET;
const EXTENSION_NODE_OVER: u8 = BRANCH_NODE_NO_VALUE - EXTENSION_NODE_OFFSET;
const LEAF_NODE_LAST: u8 = EXTENSION_NODE_OFFSET - 1;
const EXTENSION_NODE_LAST: u8 = BRANCH_NODE_NO_VALUE - 1;

// Constant use with no extensino trie codec.
const NIBBLE_SIZE_BOUND_NO_EXT: usize = u16::max_value() as usize;
const FIRST_PREFIX: u8 = 0b_00 << 6;
const LEAF_PREFIX_MASK_NO_EXT: u8 = 0b_01 << 6;
const BRANCH_WITHOUT_MASK_NO_EXT: u8 = 0b_10 << 6;
const BRANCH_WITH_MASK_NO_EXT: u8 = 0b_11 << 6;
const EMPTY_TRIE_NO_EXT: u8 = FIRST_PREFIX | 0b_00;
// TODO rem pub
pub const ENCODED_META_NO_EXT: u8 = FIRST_PREFIX | 0b_10_10;
// TODO rem pub
pub const ALLOW_HASH_META: u8 = 1;
const DEAD_HEADER_META_HASHED_VALUE: u8 = FIRST_PREFIX | 0b_11_10;

/// Create a leaf/extension node, encoding a number of nibbles. Note that this
/// cannot handle a number of nibbles that is zero or greater than 125 and if
/// you attempt to do so *IT WILL PANIC*.
fn fuse_nibbles_node<'a>(nibbles: &'a [u8], leaf: bool) -> impl Iterator<Item = u8> + 'a {
	debug_assert!(
		nibbles.len() < LEAF_NODE_OVER.min(EXTENSION_NODE_OVER) as usize,
		"nibbles length too long. what kind of size of key are you trying to include in the trie!?!"
	);
	let first_byte = if leaf {
		LEAF_NODE_OFFSET
	} else {
		EXTENSION_NODE_OFFSET
	} + nibbles.len() as u8;

	once(first_byte)
		.chain(if nibbles.len() % 2 == 1 { Some(nibbles[0]) } else { None })
		.chain(nibbles[nibbles.len() % 2..].chunks(2).map(|ch| ch[0] << 4 | ch[1]))
}

enum NodeKindNoExt {
	Leaf,
	BranchNoValue,
	BranchWithValue,
}

/// Create a leaf or branch node header followed by its encoded partial nibbles.
fn fuse_nibbles_node_no_extension<'a>(
	nibbles: &'a [u8],
	kind: NodeKindNoExt,
) -> impl Iterator<Item = u8> + 'a {
	let size = ::std::cmp::min(NIBBLE_SIZE_BOUND_NO_EXT, nibbles.len());

	let iter_start = match kind {
		NodeKindNoExt::Leaf => size_and_prefix_iterator(size, LEAF_PREFIX_MASK_NO_EXT),
		NodeKindNoExt::BranchNoValue => size_and_prefix_iterator(size, BRANCH_WITHOUT_MASK_NO_EXT),
		NodeKindNoExt::BranchWithValue => size_and_prefix_iterator(size, BRANCH_WITH_MASK_NO_EXT),
	};
	iter_start
		.chain(if nibbles.len() % 2 == 1 { Some(nibbles[0]) } else { None })
		.chain(nibbles[nibbles.len() % 2..].chunks(2).map(|ch| ch[0] << 4 | ch[1]))
}

/// Encoding of branch header and children bitmap (for trie stream radix 16).
/// For stream variant with extension.
fn branch_node(has_value: bool, has_children: impl Iterator<Item = bool>) -> [u8; 3] {
	let mut result = [0, 0, 0];
	branch_node_buffered(has_value, has_children, &mut result[..]);
	result
}

/// Encoding of branch header and children bitmap for any radix.
/// For codec/stream variant with extension.
fn branch_node_buffered<I: Iterator<Item = bool>>(
	has_value: bool,
	has_children: I,
	output: &mut[u8],
) {
	let first = if has_value {
		BRANCH_NODE_WITH_VALUE
	} else {
		BRANCH_NODE_NO_VALUE
	};
	output[0] = first;
	Bitmap::encode(has_children, &mut output[1..]);
}

/// Encoding of children bitmap (for trie stream radix 16).
/// For stream variant without extension.
fn branch_node_bit_mask(has_children: impl Iterator<Item = bool>) -> (u8, u8) {
	let mut bitmap: u16 = 0;
	let mut cursor: u16 = 1;
	for v in has_children {
		if v { bitmap |= cursor }
		cursor <<= 1;
	}
	((bitmap % 256 ) as u8, (bitmap / 256 ) as u8)
}

/// Reference implementation of a `TrieStream` with extension nodes.
#[derive(Default, Clone)]
pub struct ReferenceTrieStream {
	buffer: Vec<u8>
}

impl TrieStream for ReferenceTrieStream {
	type GlobalMeta = ();

	fn new(_meta: ()) -> Self {
		ReferenceTrieStream {
			buffer: Vec::new()
		}
	}

	fn append_empty_data(&mut self) {
		self.buffer.push(EMPTY_TRIE);
	}

	fn append_leaf(&mut self, key: &[u8], value: &[u8]) {
		self.buffer.extend(fuse_nibbles_node(key, true));
		value.encode_to(&mut self.buffer);
	}

	fn begin_branch(
		&mut self,
		maybe_key: Option<&[u8]>,
		maybe_value: Option<&[u8]>,
		has_children: impl Iterator<Item = bool>,
	) {
		self.buffer.extend(&branch_node(maybe_value.is_some(), has_children));
		if let Some(partial) = maybe_key {
			// should not happen
			self.buffer.extend(fuse_nibbles_node(partial, false));
		}
		if let Some(value) = maybe_value {
			value.encode_to(&mut self.buffer);
		}
	}

	fn append_extension(&mut self, key: &[u8]) {
		self.buffer.extend(fuse_nibbles_node(key, false));
	}

	fn append_substream<H: Hasher>(&mut self, other: Self) {
		let data = other.out();
		match data.len() {
			0..=31 => data.encode_to(&mut self.buffer),
			_ => H::hash(&data).as_ref().encode_to(&mut self.buffer),
		}
	}

	fn out(self) -> Vec<u8> { self.buffer }

	fn hash_root<H: Hasher>(self) -> H::Out {
		H::hash(&self.buffer)
	}
}

/// Reference implementation of a `TrieStream` without extension.
#[derive(Default, Clone)]
pub struct ReferenceTrieStreamNoExt {
	buffer: Vec<u8>,
	inner_value_hashing: bool,
	current_value_range: Option<Range<usize>>,
}

impl TrieStream for ReferenceTrieStreamNoExt {
	type GlobalMeta = bool;

	fn new(meta: bool) -> Self {
		ReferenceTrieStreamNoExt {
			buffer: Vec::new(),
			inner_value_hashing: meta,
			current_value_range: None,
		}
	}

	fn append_empty_data(&mut self) {
		self.buffer.push(EMPTY_TRIE_NO_EXT);
	}

	fn append_leaf(&mut self, key: &[u8], value: &[u8]) {
		self.buffer.extend(fuse_nibbles_node_no_extension(key, NodeKindNoExt::Leaf));
		Compact(value.len() as u32).encode_to(&mut self.buffer);
		self.current_value_range = Some(self.buffer.len()..self.buffer.len() + value.len());
		self.buffer.extend_from_slice(value);
	}

	fn begin_branch(
		&mut self,
		maybe_key: Option<&[u8]>,
		maybe_value: Option<&[u8]>,
		has_children: impl Iterator<Item = bool>
	) {
		if let Some(partial) = maybe_key {
			if maybe_value.is_some() {
				self.buffer.extend(
					fuse_nibbles_node_no_extension(partial, NodeKindNoExt::BranchWithValue)
				);
			} else {
				self.buffer.extend(
					fuse_nibbles_node_no_extension(partial, NodeKindNoExt::BranchNoValue)
				);
			}
			let bitmap = branch_node_bit_mask(has_children);
			self.buffer.extend([bitmap.0, bitmap.1].iter());
		} else {
			// should not happen
			self.buffer.extend(&branch_node(maybe_value.is_some(), has_children));
		}
		if let Some(value) = maybe_value {
			value.encode_to(&mut self.buffer);
		}
	}

	fn append_extension(&mut self, _key: &[u8]) {
		// should not happen
	}

	fn append_substream<H: Hasher>(&mut self, other: Self) {
		let inner_value_hashing = other.inner_value_hashing;
		let range = other.current_value_range.clone();
		let data = other.out();
		match data.len() {
			0..=31 => data.encode_to(&mut self.buffer),
			_ => {
				if inner_value_hashing
					&& range.as_ref().map(|r| r.end - r.start >= INNER_HASH_TRESHOLD).unwrap_or_default() {
					let meta = ValueMeta {
						range: range,
						unused_value: false,
						contain_hash: false,
						do_value_hash: true,
						recorded_do_value_hash: false,
					};
					<TestMetaHasher<H> as MetaHasher<H, Vec<u8>>>::hash(&data, &meta).as_ref().encode_to(&mut self.buffer);
				} else {
					H::hash(&data).as_ref().encode_to(&mut self.buffer);
				}
			},
		}
	}

	fn hash_root<H: Hasher>(self) -> H::Out {
		let inner_value_hashing = self.inner_value_hashing;
		let range = self.current_value_range;
		let data = self.buffer;
		if inner_value_hashing
			&& range.as_ref().map(|r| r.end - r.start >= INNER_HASH_TRESHOLD).unwrap_or_default() {
			let meta = ValueMeta {
				range: range,
				unused_value: false,
				contain_hash: false,
				do_value_hash: true,
				recorded_do_value_hash: true,
			};
			// Add the recorded_do_value_hash to encoded
			let mut encoded = meta.write_state_meta();
			let encoded = if encoded.len() > 0 {
				encoded.extend(data);
				encoded
			} else {
				data
			};
			<TestMetaHasher<H> as MetaHasher<H, Vec<u8>>>::hash(&encoded, &meta)
		} else {
			H::hash(&data)
		}
	}

	fn out(self) -> Vec<u8> { self.buffer }
}

/// A node header.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NodeHeader {
	Null,
	Branch(bool),
	Extension(usize),
	Leaf(usize),
}

/// A node header no extension.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum NodeHeaderNoExt {
	Null,
	Branch(bool, usize),
	Leaf(usize),
}

impl Encode for NodeHeader {
	fn encode_to<T: Output + ?Sized>(&self, output: &mut T) {
		match self {
			NodeHeader::Null => output.push_byte(EMPTY_TRIE),
			NodeHeader::Branch(true) => output.push_byte(BRANCH_NODE_WITH_VALUE),
			NodeHeader::Branch(false) => output.push_byte(BRANCH_NODE_NO_VALUE),
			NodeHeader::Leaf(nibble_count) =>
				output.push_byte(LEAF_NODE_OFFSET + *nibble_count as u8),
			NodeHeader::Extension(nibble_count) =>
				output.push_byte(EXTENSION_NODE_OFFSET + *nibble_count as u8),
		}
	}
}

/// Encode and allocate node type header (type and size), and partial value.
/// It uses an iterator over encoded partial bytes as input.
fn size_and_prefix_iterator(size: usize, prefix: u8) -> impl Iterator<Item = u8> {
	let size = ::std::cmp::min(NIBBLE_SIZE_BOUND_NO_EXT, size);

	let l1 = std::cmp::min(62, size);
	let (first_byte, mut rem) = if size == l1 {
		(once(prefix + l1 as u8), 0)
	} else {
		(once(prefix + 63), size - l1)
	};
	let next_bytes = move || {
		if rem > 0 {
			if rem < 256 {
				let result = rem - 1;
				rem = 0;
				Some(result as u8)
			} else {
				rem = rem.saturating_sub(255);
				Some(255)
			}
		} else {
			None
		}
	};
	first_byte.chain(::std::iter::from_fn(next_bytes))
}

fn encode_size_and_prefix(size: usize, prefix: u8, out: &mut (impl Output + ?Sized)) {
	for b in size_and_prefix_iterator(size, prefix) {
		out.push_byte(b)
	}
}

fn decode_size<I: Input>(first: u8, input: &mut I) -> Result<usize, CodecError> {
	let mut result = (first & 255u8 >> 2) as usize;
	if result < 63 {
		return Ok(result);
	}
	result -= 1;
	while result <= NIBBLE_SIZE_BOUND_NO_EXT {
		let n = input.read_byte()? as usize;
		if n < 255 {
			return Ok(result + n + 1);
		}
		result += 255;
	}
	Err("Size limit reached for a nibble slice".into())
}

impl Encode for NodeHeaderNoExt {
	fn encode_to<T: Output + ?Sized>(&self, output: &mut T) {
		match self {
			NodeHeaderNoExt::Null => output.push_byte(EMPTY_TRIE_NO_EXT),
			NodeHeaderNoExt::Branch(true, nibble_count)	=>
				encode_size_and_prefix(*nibble_count, BRANCH_WITH_MASK_NO_EXT, output),
			NodeHeaderNoExt::Branch(false, nibble_count) =>
				encode_size_and_prefix(*nibble_count, BRANCH_WITHOUT_MASK_NO_EXT, output),
			NodeHeaderNoExt::Leaf(nibble_count) =>
				encode_size_and_prefix(*nibble_count, LEAF_PREFIX_MASK_NO_EXT, output),
		}
	}
}

impl Decode for NodeHeader {
	fn decode<I: Input>(input: &mut I) -> Result<Self, CodecError> {
		Ok(match input.read_byte()? {
			EMPTY_TRIE => NodeHeader::Null,
			BRANCH_NODE_NO_VALUE => NodeHeader::Branch(false),
			BRANCH_NODE_WITH_VALUE => NodeHeader::Branch(true),
			i @ LEAF_NODE_OFFSET ..= LEAF_NODE_LAST =>
				NodeHeader::Leaf((i - LEAF_NODE_OFFSET) as usize),
			i @ EXTENSION_NODE_OFFSET ..= EXTENSION_NODE_LAST =>
				NodeHeader::Extension((i - EXTENSION_NODE_OFFSET) as usize),
		})
	}
}

impl Decode for NodeHeaderNoExt {
	fn decode<I: Input>(input: &mut I) -> Result<Self, CodecError> {
		let i = input.read_byte()?;
		if i == EMPTY_TRIE_NO_EXT {
			return Ok(NodeHeaderNoExt::Null);
		}
		match i & (0b11 << 6) {
			LEAF_PREFIX_MASK_NO_EXT =>
				Ok(NodeHeaderNoExt::Leaf(decode_size(i, input)?)),
			BRANCH_WITHOUT_MASK_NO_EXT =>
				Ok(NodeHeaderNoExt::Branch(false, decode_size(i, input)?)),
			BRANCH_WITH_MASK_NO_EXT =>
				Ok(NodeHeaderNoExt::Branch(true, decode_size(i, input)?)),
			// do not allow any special encoding
			_ => Err("Unknown type of node".into()),
		}
	}
}

/// Simple reference implementation of a `NodeCodec`.
#[derive(Default, Clone)]
pub struct ReferenceNodeCodec<H>(PhantomData<H>);

/// Simple reference implementation of a `NodeCodec`.
/// Even if implementation follows initial specification of
/// https://github.com/w3f/polkadot-re-spec/issues/8, this may
/// not follow it in the future, it is mainly the testing codec without extension node.
#[derive(Default, Clone)]
pub struct ReferenceNodeCodecNoExt<H>(PhantomData<H>);

fn partial_to_key(partial: Partial, offset: u8, over: u8) -> Vec<u8> {
	let number_nibble_encoded = (partial.0).0 as usize;
	let nibble_count = partial.1.len() * nibble_ops::NIBBLE_PER_BYTE + number_nibble_encoded;
	assert!(nibble_count < over as usize);
	let mut output = vec![offset + nibble_count as u8];
	if number_nibble_encoded > 0 {
		output.push(nibble_ops::pad_right((partial.0).1));
	}
	output.extend_from_slice(&partial.1[..]);
	output
}

fn partial_from_iterator_to_key<I: Iterator<Item = u8>>(
	partial: I,
	nibble_count: usize,
	offset: u8,
	over: u8,
) -> Vec<u8> {
	assert!(nibble_count < over as usize);
	let mut output = Vec::with_capacity(1 + (nibble_count / nibble_ops::NIBBLE_PER_BYTE));
	output.push(offset + nibble_count as u8);
	output.extend(partial);
	output
}

fn partial_from_iterator_encode<I: Iterator<Item = u8>>(
	partial: I,
	nibble_count: usize,
	node_kind: NodeKindNoExt,
) -> Vec<u8> {
	let nibble_count = ::std::cmp::min(NIBBLE_SIZE_BOUND_NO_EXT, nibble_count);

	let mut output = Vec::with_capacity(3 + (nibble_count / nibble_ops::NIBBLE_PER_BYTE));
	match node_kind {
		NodeKindNoExt::Leaf =>
			NodeHeaderNoExt::Leaf(nibble_count).encode_to(&mut output),
		NodeKindNoExt::BranchWithValue =>
			NodeHeaderNoExt::Branch(true, nibble_count).encode_to(&mut output),
		NodeKindNoExt::BranchNoValue =>
			NodeHeaderNoExt::Branch(false, nibble_count).encode_to(&mut output),
	};
	output.extend(partial);
	output
}

fn partial_encode(partial: Partial, node_kind: NodeKindNoExt) -> Vec<u8> {
	let number_nibble_encoded = (partial.0).0 as usize;
	let nibble_count = partial.1.len() * nibble_ops::NIBBLE_PER_BYTE + number_nibble_encoded;

	let nibble_count = ::std::cmp::min(NIBBLE_SIZE_BOUND_NO_EXT, nibble_count);

	let mut output = Vec::with_capacity(3 + partial.1.len());
	match node_kind {
		NodeKindNoExt::Leaf =>
			NodeHeaderNoExt::Leaf(nibble_count).encode_to(&mut output),
		NodeKindNoExt::BranchWithValue =>
			NodeHeaderNoExt::Branch(true, nibble_count).encode_to(&mut output),
		NodeKindNoExt::BranchNoValue =>
			NodeHeaderNoExt::Branch(false, nibble_count).encode_to(&mut output),
	};
	if number_nibble_encoded > 0 {
		output.push(nibble_ops::pad_right((partial.0).1));
	}
	output.extend_from_slice(&partial.1[..]);
	output
}

struct ByteSliceInput<'a> {
	data: &'a [u8],
	offset: usize,
}

impl<'a> ByteSliceInput<'a> {
	fn new(data: &'a [u8]) -> Self {
		ByteSliceInput {
			data,
			offset: 0,
		}
	}

	fn take(&mut self, count: usize) -> Result<Range<usize>, CodecError> {
		if self.offset + count > self.data.len() {
			return Err("out of data".into());
		}

		let range = self.offset..(self.offset + count);
		self.offset += count;
		Ok(range)
	}
}

impl<'a> Input for ByteSliceInput<'a> {
	fn remaining_len(&mut self) -> Result<Option<usize>, CodecError> {
		let remaining = if self.offset <= self.data.len() {
			Some(self.data.len() - self.offset)
		} else {
			None
		};
		Ok(remaining)
	}

	fn read(&mut self, into: &mut [u8]) -> Result<(), CodecError> {
		let range = self.take(into.len())?;
		into.copy_from_slice(&self.data[range]);
		Ok(())
	}

	fn read_byte(&mut self) -> Result<u8, CodecError> {
		if self.offset + 1 > self.data.len() {
			return Err("out of data".into());
		}

		let byte = self.data[self.offset];
		self.offset += 1;
		Ok(byte)
	}
}

// NOTE: what we'd really like here is:
// `impl<H: Hasher> NodeCodec<H> for RlpNodeCodec<H> where <KeccakHasher as Hasher>::Out: Decodable`
// but due to the current limitations of Rust const evaluation we can't do
// `const HASHED_NULL_NODE: <KeccakHasher as Hasher>::Out = <KeccakHasher as Hasher>::Out( … … )`.
// Perhaps one day soon?
impl<H: Hasher, M: Meta> NodeCodec<M> for ReferenceNodeCodec<H> {
	type Error = CodecError;
	type HashOut = H::Out;

	fn hashed_null_node() -> <H as Hasher>::Out {
		H::hash(<Self as NodeCodec<M>>::empty_node_no_meta())
	}

	fn decode_plan_inner(data: &[u8]) -> ::std::result::Result<NodePlan, Self::Error> {
		let mut input = ByteSliceInput::new(data);
		match NodeHeader::decode(&mut input)? {
			NodeHeader::Null => Ok(NodePlan::Empty),
			NodeHeader::Branch(has_value) => {
				let bitmap_range = input.take(BITMAP_LENGTH)?;
				let bitmap = Bitmap::decode(&data[bitmap_range])?;

				let value = if has_value {
					let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
					ValuePlan::Value(input.take(count)?)
				} else {
					ValuePlan::NoValue
				};
				let mut children = [
					None, None, None, None, None, None, None, None,
					None, None, None, None, None, None, None, None,
				];
				for i in 0..nibble_ops::NIBBLE_LENGTH {
					if bitmap.value_at(i) {
						let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
						let range = input.take(count)?;
						children[i] = Some(if count == H::LENGTH {
							NodeHandlePlan::Hash(range)
						} else {
							NodeHandlePlan::Inline(range)
						});
					}
				}
				Ok(NodePlan::Branch { value, children })
			}
			NodeHeader::Extension(nibble_count) => {
				let partial = input.take(
					(nibble_count + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE
				)?;
				let partial_padding = nibble_ops::number_padding(nibble_count);
				let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
				let range = input.take(count)?;
				let child = if count == H::LENGTH {
					NodeHandlePlan::Hash(range)
				} else {
					NodeHandlePlan::Inline(range)
				};
				Ok(NodePlan::Extension {
					partial: NibbleSlicePlan::new(partial, partial_padding),
					child
				})
			}
			NodeHeader::Leaf(nibble_count) => {
				let partial = input.take(
					(nibble_count + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE
				)?;
				let partial_padding = nibble_ops::number_padding(nibble_count);
				let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
				let value = input.take(count)?;
				Ok(NodePlan::Leaf {
					partial: NibbleSlicePlan::new(partial, partial_padding),
					value: ValuePlan::Value(value),
				})
			}
		}
	}

	fn is_empty_node(data: &[u8]) -> bool {
		data == <Self as NodeCodec<M>>::empty_node_no_meta()
	}

	fn empty_node(_meta: &mut M) -> Vec<u8> {
		vec![EMPTY_TRIE]
	}

	fn empty_node_no_meta() -> &'static[u8] {
		&[EMPTY_TRIE]
	}

	fn leaf_node(partial: Partial, value: Value, meta: &mut M) -> Vec<u8> {
		let mut output = partial_to_key(partial, LEAF_NODE_OFFSET, LEAF_NODE_OVER);
		match value {
			Value::Value(value) => {
				Compact(value.len() as u32).encode_to(&mut output);
				let start = output.len();
				output.extend_from_slice(value);
				let end = output.len();
				meta.encoded_value_callback(ValuePlan::Value(start..end));
			},
			_ => unimplemented!("unsupported"),
		}
		output
	}

	fn extension_node(
		partial: impl Iterator<Item = u8>,
		number_nibble: usize,
		child: ChildReference<Self::HashOut>,
		_meta: &mut M,
	) -> Vec<u8> {
		let mut output = partial_from_iterator_to_key(
			partial,
			number_nibble,
			EXTENSION_NODE_OFFSET,
			EXTENSION_NODE_OVER,
		);
		match child {
			ChildReference::Hash(h) => h.as_ref().encode_to(&mut output),
			ChildReference::Inline(inline_data, len) =>
				(&AsRef::<[u8]>::as_ref(&inline_data)[..len]).encode_to(&mut output),
		};
		output
	}

	fn branch_node(
		children: impl Iterator<Item = impl Borrow<Option<ChildReference<Self::HashOut>>>>,
		maybe_value: Value,
		meta: &mut M,
	) -> Vec<u8> {
		let mut output = vec![0; BITMAP_LENGTH + 1];
		let mut prefix: [u8; 3] = [0; 3];
		let have_value = match maybe_value {
			Value::Value(value) => {
				Compact(value.len() as u32).encode_to(&mut output);
				let start = output.len();
				output.extend_from_slice(value);
				let end = output.len();
				meta.encoded_value_callback(ValuePlan::Value(start..end));
				true
			},
			Value::NoValue => {
				meta.encoded_value_callback(ValuePlan::NoValue);
				false
			},
			_ => unimplemented!("unsupported"),
		};
		let has_children = children.map(|maybe_child| match maybe_child.borrow() {
			Some(ChildReference::Hash(h)) => {
				h.as_ref().encode_to(&mut output);
				true
			}
			&Some(ChildReference::Inline(inline_data, len)) => {
				inline_data.as_ref()[..len].encode_to(&mut output);
				true
			}
			None => false,
		});
		branch_node_buffered(have_value, has_children, prefix.as_mut());
		output[0..BITMAP_LENGTH + 1].copy_from_slice(prefix.as_ref());
		output
	}

	fn branch_node_nibbled(
		_partial:	impl Iterator<Item = u8>,
		_number_nibble: usize,
		_children: impl Iterator<Item = impl Borrow<Option<ChildReference<Self::HashOut>>>>,
		_maybe_value: Value,
		_meta: &mut M,
	) -> Vec<u8> {
		unreachable!()
	}
}

impl<H: Hasher> ReferenceNodeCodecNoExt<H> {
	fn decode_plan_inner2<M: Meta>(
		data: &[u8],
		contains_hash: bool,
		meta: Option<&mut M>,
	) -> std::result::Result<NodePlan, CodecError> {
		if data.len() < 1 {
			return Err(CodecError::from("Empty encoded node."));
		}
		let offset = if let Some(meta) = meta {
			meta.read_state_meta(data)?
		} else {
			0
		};
		let mut input = ByteSliceInput::new(data);
		let _ = input.take(offset)?;

		Ok(match NodeHeaderNoExt::decode(&mut input)? {
			NodeHeaderNoExt::Null => NodePlan::Empty,
			NodeHeaderNoExt::Branch(has_value, nibble_count) => {
				let padding = nibble_count % nibble_ops::NIBBLE_PER_BYTE != 0;
				// check that the padding is valid (if any)
				if padding && nibble_ops::pad_left(data[input.offset]) != 0 {
					return Err(CodecError::from("Bad format"));
				}
				let partial = input.take(
					(nibble_count + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE
				)?;
				let partial_padding = nibble_ops::number_padding(nibble_count);
				let bitmap_range = input.take(BITMAP_LENGTH)?;
				let bitmap = Bitmap::decode(&data[bitmap_range])?;
				let value = if has_value {
					let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
					if contains_hash {
						ValuePlan::HashedValue(input.take(H::LENGTH)?, count)
					} else {
						ValuePlan::Value(input.take(count)?)
					}
				} else {
					ValuePlan::NoValue
				};
				let mut children = [
					None, None, None, None, None, None, None, None,
					None, None, None, None, None, None, None, None,
				];
				for i in 0..nibble_ops::NIBBLE_LENGTH {
					if bitmap.value_at(i) {
						let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
						let range = input.take(count)?;
						children[i] = Some(if count == H::LENGTH {
							NodeHandlePlan::Hash(range)
						} else {
							NodeHandlePlan::Inline(range)
						});
					}
				}
				NodePlan::NibbledBranch {
					partial: NibbleSlicePlan::new(partial, partial_padding),
					value,
					children,
				}
			}
			NodeHeaderNoExt::Leaf(nibble_count) => {
				let padding = nibble_count % nibble_ops::NIBBLE_PER_BYTE != 0;
				// check that the padding is valid (if any)
				if padding && nibble_ops::pad_left(data[input.offset]) != 0 {
					return Err(CodecError::from("Bad format"));
				}
				let partial = input.take(
					(nibble_count + (nibble_ops::NIBBLE_PER_BYTE - 1)) / nibble_ops::NIBBLE_PER_BYTE
				)?;
				let partial_padding = nibble_ops::number_padding(nibble_count);
				let count = <Compact<u32>>::decode(&mut input)?.0 as usize;
				let value = if contains_hash {
					ValuePlan::HashedValue(input.take(H::LENGTH)?, count)
				} else {
					ValuePlan::Value(input.take(count)?)
				};

				NodePlan::Leaf {
					partial: NibbleSlicePlan::new(partial, partial_padding),
					value,
				}
			}
		})
	}
}


impl<H: Hasher, M: Meta> NodeCodec<M> for ReferenceNodeCodecNoExt<H> {
	type Error = CodecError;
	type HashOut = <H as Hasher>::Out;

	fn hashed_null_node() -> <H as Hasher>::Out {
		H::hash(<Self as NodeCodec<M>>::empty_node_no_meta())
	}

	fn decode_plan(data: &[u8], meta: &mut M) -> Result<NodePlan, Self::Error> {
		let contains_hash = meta.contains_hash_of_value();
		Self::decode_plan_inner2(data, contains_hash, Some(meta)).map(|plan| {
			meta.decoded_callback(&plan);
			plan
		})
	}

	fn decode_plan_inner(data: &[u8]) -> std::result::Result<NodePlan, Self::Error> {
		let meta: Option<&mut M> = None;
		Self::decode_plan_inner2(data, false, meta)
	}

	fn is_empty_node(data: &[u8]) -> bool {
		data == <Self as NodeCodec<M>>::empty_node_no_meta()
	}

	fn empty_node(meta: &mut M) -> Vec<u8> {
		let mut output = meta.write_state_meta();
		output.extend_from_slice(&[EMPTY_TRIE_NO_EXT]);
		output
	}

	fn empty_node_no_meta() -> &'static [u8] {
		&[EMPTY_TRIE_NO_EXT]
	}

	fn leaf_node(partial: Partial, value: Value, meta: &mut M) -> Vec<u8> {
		let mut output = meta.write_state_meta();
		output.append(&mut partial_encode(partial, NodeKindNoExt::Leaf));
		match value {
			Value::Value(value) => {
				Compact(value.len() as u32).encode_to(&mut output);
				let start = output.len();
				output.extend_from_slice(value);
				let end = output.len();
				meta.encoded_value_callback(ValuePlan::Value(start..end));
			},
			Value::HashedValue(hash, size) => {
				debug_assert!(hash.len() == H::LENGTH);
				Compact(size as u32).encode_to(&mut output);
				let start = output.len();
				output.extend_from_slice(hash);
				let end = output.len();
				meta.encoded_value_callback(ValuePlan::HashedValue(start..end, size));
			},
			Value::NoValue => unreachable!(),
		}
		output
	}

	fn extension_node(
		_partial: impl Iterator<Item = u8>,
		_nbnibble: usize,
		_child: ChildReference<<H as Hasher>::Out>,
		_met: &mut M,
	) -> Vec<u8> {
		unreachable!()
	}

	fn branch_node(
		_children: impl Iterator<Item = impl Borrow<Option<ChildReference<<H as Hasher>::Out>>>>,
		_maybe_value: Value,
		_meta: &mut M,
	) -> Vec<u8> {
		unreachable!()
	}

	fn branch_node_nibbled(
		partial: impl Iterator<Item = u8>,
		number_nibble: usize,
		children: impl Iterator<Item = impl Borrow<Option<ChildReference<Self::HashOut>>>>,
		maybe_value: Value,
		meta: &mut M,
	) -> Vec<u8> {
		let mut output = meta.write_state_meta();
		output.append(&mut if let Value::NoValue = &maybe_value {
			partial_from_iterator_encode(
				partial,
				number_nibble,
				NodeKindNoExt::BranchNoValue,
			)
		} else {
			partial_from_iterator_encode(
				partial,
				number_nibble,
				NodeKindNoExt::BranchWithValue,
			)
		});
		let bitmap_index = output.len();
		let mut bitmap: [u8; BITMAP_LENGTH] = [0; BITMAP_LENGTH];
		(0..BITMAP_LENGTH).for_each(|_| output.push(0));
		match maybe_value {
			Value::Value(value) => {
				Compact(value.len() as u32).encode_to(&mut output);
				let start = output.len();
				output.extend_from_slice(value);
				let end = output.len();
				meta.encoded_value_callback(ValuePlan::Value(start..end));
			},
			Value::HashedValue(hash, size) => {
				debug_assert!(hash.len() == H::LENGTH);
				Compact(size as u32).encode_to(&mut output);
				let start = output.len();
				output.extend_from_slice(hash);
				let end = output.len();
				meta.encoded_value_callback(ValuePlan::HashedValue(start..end, size));
			},
			Value::NoValue => (),
		}

		Bitmap::encode(children.map(|maybe_child| match maybe_child.borrow() {
			Some(ChildReference::Hash(h)) => {
				h.as_ref().encode_to(&mut output);
				true
			}
			&Some(ChildReference::Inline(inline_data, len)) => {
				inline_data.as_ref()[..len].encode_to(&mut output);
				true
			}
			None => false,
		}), bitmap.as_mut());
		output[bitmap_index..bitmap_index + BITMAP_LENGTH]
			.copy_from_slice(&bitmap.as_ref()[..BITMAP_LENGTH]);
		output
	}
}

/// Compare trie builder and in memory trie.
pub fn compare_implementations<T, DB> (
	data: Vec<(Vec<u8>, Vec<u8>)>,
	mut memdb: DB,
	mut hashdb: DB,
)
	where
		T: TrieLayout,
		DB : hash_db::HashDB<T::Hash, DBValue, T::Meta, GlobalMeta<T>> + Eq,
{
	let root_new = calc_root_build::<T, _, _, _, _>(data.clone(), &mut hashdb);
	let root = {
		let mut root = Default::default();
		let mut t = TrieDBMut::<T>::new(&mut memdb, &mut root);
		for i in 0..data.len() {
			t.insert(&data[i].0[..], &data[i].1[..]).unwrap();
		}
		t.commit();
		*t.root()
	};
	if root_new != root {
		{
			let db : &dyn hash_db::HashDB<_, _, _, _> = &hashdb;
			let t = TrieDB::<T>::new(&db, &root_new).unwrap();
			println!("{:?}", t);
			for a in t.iter().unwrap() {
				println!("a:{:x?}", a);
			}
		}
		{
			let db : &dyn hash_db::HashDB<_, _, _, _> = &memdb;
			let t = TrieDB::<T>::new(&db, &root).unwrap();
			println!("{:?}", t);
			for a in t.iter().unwrap() {
				println!("a:{:x?}", a);
			}
		}
	}

	assert_eq!(root, root_new);
	// compare db content for key fuzzing
	assert!(memdb == hashdb);
}

/// Compare trie builder and trie root implementations.
pub fn compare_root<T: TrieLayout, DB: hash_db::HashDB<T::Hash, DBValue, T::Meta, GlobalMeta<T>>>(
	data: Vec<(Vec<u8>, Vec<u8>)>,
	mut memdb: DB,
) {
	let root_new = reference_trie_root_iter_build::<T, _, _, _>(data.clone());
	let root = {
		let mut root = Default::default();
		let mut t = trie_db::TrieDBMut::<T>::new(&mut memdb, &mut root);
		for i in 0..data.len() {
			t.insert(&data[i].0[..], &data[i].1[..]).unwrap();
		}
		*t.root()
	};

	assert_eq!(root, root_new);
}

/// Compare trie builder and trie root unhashed implementations.
pub fn compare_unhashed(
	data: Vec<(Vec<u8>, Vec<u8>)>,
) {
	let root_new = {
		let mut cb = trie_db::TrieRootUnhashed::<ExtensionLayout>::default();
		trie_visit(data.clone().into_iter(), &mut cb, &ExtensionLayout);
		cb.root.unwrap_or(Default::default())
	};
	let root = reference_trie_root_unhashed(data);

	assert_eq!(root, root_new);
}

/// Compare trie builder and trie root unhashed implementations.
/// This uses the variant without extension nodes.
pub fn compare_unhashed_no_extension(
	data: Vec<(Vec<u8>, Vec<u8>)>,
) {
	let root_new = {
		let mut cb = trie_db::TrieRootUnhashed::<NoExtensionLayout>::default();
		trie_visit(data.clone().into_iter(), &mut cb, &NoExtensionLayout::default());
		cb.root.unwrap_or(Default::default())
	};
	let root = reference_trie_root_unhashed_no_extension(data);

	assert_eq!(root, root_new);
}

/// Trie builder root calculation utility.
pub fn calc_root<T, I, A, B>(
	data: I,
) -> <T::Hash as Hasher>::Out
	where
		T: TrieLayout,
		I: IntoIterator<Item = (A, B)>,
		A: AsRef<[u8]> + Ord + fmt::Debug,
		B: AsRef<[u8]> + fmt::Debug,
{
	let mut cb = TrieRoot::<T>::default();
	trie_visit(data.into_iter(), &mut cb, &T::default());
	cb.root.unwrap_or_default()
}

/// Trie builder trie building utility.
pub fn calc_root_build<T, I, A, B, DB>(
	data: I,
	hashdb: &mut DB
) -> <T::Hash as Hasher>::Out
	where
		T: TrieLayout,
		I: IntoIterator<Item = (A, B)>,
		A: AsRef<[u8]> + Ord + fmt::Debug,
		B: AsRef<[u8]> + fmt::Debug,
		DB: hash_db::HashDB<T::Hash, DBValue, T::Meta, GlobalMeta<T>>,
{
	let mut cb = TrieBuilder::<T, DB>::new(hashdb);
	trie_visit(data.into_iter(), &mut cb, &T::default());
	cb.root.unwrap_or_default()
}

/// `compare_implementations_no_extension` for unordered input (trie_root does
/// ordering before running when trie_build expect correct ordering).
pub fn compare_implementations_unordered<T, DB> (
	data: Vec<(Vec<u8>, Vec<u8>)>,
	mut memdb: DB,
	mut hashdb: DB,
)
	where
		T: TrieLayout,
		DB : hash_db::HashDB<T::Hash, DBValue, T::Meta, GlobalMeta<T>> + Eq,
{
	let mut b_map = std::collections::btree_map::BTreeMap::new();
	let root = {
		let mut root = Default::default();
		let mut t = TrieDBMut::<T>::new(&mut memdb, &mut root);
		for i in 0..data.len() {
			t.insert(&data[i].0[..], &data[i].1[..]).unwrap();
			b_map.insert(data[i].0.clone(), data[i].1.clone());
		}
		*t.root()
	};
	let root_new = {
		let mut cb = TrieBuilder::<T, DB>::new(&mut hashdb);
		trie_visit(b_map.into_iter(), &mut cb, &T::default());
		cb.root.unwrap_or_default()
	};

	if root != root_new {
		{
			let db : &dyn hash_db::HashDB<_, _, _, _> = &memdb;
			let t = TrieDB::<T>::new(&db, &root).unwrap();
			println!("{:?}", t);
			for a in t.iter().unwrap() {
				println!("a:{:?}", a);
			}
		}
		{
			let db : &dyn hash_db::HashDB<_, _, _, _> = &hashdb;
			let t = TrieDB::<T>::new(&db, &root_new).unwrap();
			println!("{:?}", t);
			for a in t.iter().unwrap() {
				println!("a:{:?}", a);
			}
		}
	}

	assert_eq!(root, root_new);
}

/// Testing utility that uses some periodic removal over
/// its input test data.
pub fn compare_insert_remove<T, DB: hash_db::HashDB<T::Hash, DBValue, T::Meta, GlobalMeta<T>>>(
	data: Vec<(bool, Vec<u8>, Vec<u8>)>,
	mut memdb: DB,
)
	where
		T: TrieLayout,
		DB : hash_db::HashDB<T::Hash, DBValue, T::Meta, GlobalMeta<T>> + Eq,
{

	let mut data2 = std::collections::BTreeMap::new();
	let mut root = Default::default();
	let mut a = 0;
	{
		let mut t = TrieDBMut::<T>::new(&mut memdb, &mut root);
		t.commit();
	}
	while a < data.len() {
		// new triemut every 3 element
		root = {
			let mut t = TrieDBMut::<T>::from_existing(&mut memdb, &mut root).unwrap();
			for _ in 0..3 {
				if data[a].0 {
					// remove
					t.remove(&data[a].1[..]).unwrap();
					data2.remove(&data[a].1[..]);
				} else {
					// add
					t.insert(&data[a].1[..], &data[a].2[..]).unwrap();
					data2.insert(&data[a].1[..], &data[a].2[..]);
				}

				a += 1;
				if a == data.len() {
					break;
				}
			}
			t.commit();
			*t.root()
		};
	}
	let mut t = TrieDBMut::<T>::from_existing(&mut memdb, &mut root).unwrap();
	// we are testing the RefTrie code here so we do not sort or check uniqueness
	// before.
	assert_eq!(*t.root(), calc_root::<T, _, _, _>(data2));
}

#[cfg(test)]
mod tests {
	use super::*;
	use trie_db::node::Node;

	#[test]
	fn test_encoding_simple_trie() {
		for prefix in [
			LEAF_PREFIX_MASK_NO_EXT,
			BRANCH_WITHOUT_MASK_NO_EXT,
			BRANCH_WITH_MASK_NO_EXT,
		].iter() {
			for i in (0..1000).chain(NIBBLE_SIZE_BOUND_NO_EXT - 2..NIBBLE_SIZE_BOUND_NO_EXT + 2) {
				let mut output = Vec::new();
				encode_size_and_prefix(i, *prefix, &mut output);
				let input = &mut &output[..];
				let first = input.read_byte().unwrap();
				assert_eq!(first & (0b11 << 6), *prefix);
				let v = decode_size(first, input);
				assert_eq!(Ok(std::cmp::min(i, NIBBLE_SIZE_BOUND_NO_EXT)), v);
			}
		}
	}

	#[test]
	fn too_big_nibble_length() {
		// + 1 for 0 added byte of nibble encode
		let input = vec![0u8; (NIBBLE_SIZE_BOUND_NO_EXT as usize + 1) / 2 + 1];
		let enc = <ReferenceNodeCodecNoExt<RefHasher> as NodeCodec<_>>
		::leaf_node(((0, 0), &input), Value::Value(&[1]), &mut ());
		let dec = <ReferenceNodeCodecNoExt<RefHasher> as NodeCodec<_>>
		::decode(&enc, &mut ()).unwrap();
		let o_sl = if let Node::Leaf(sl, _) = dec {
			Some(sl)
		} else { None };
		assert!(o_sl.is_some());
	}

	#[test]
	fn size_encode_limit_values() {
		let sizes = [0, 1, 62, 63, 64, 317, 318, 319, 572, 573, 574];
		let encs = [
			vec![0],
			vec![1],
			vec![0x3e],
			vec![0x3f, 0],
			vec![0x3f, 1],
			vec![0x3f, 0xfe],
			vec![0x3f, 0xff, 0],
			vec![0x3f, 0xff, 1],
			vec![0x3f, 0xff, 0xfe],
			vec![0x3f, 0xff, 0xff, 0],
			vec![0x3f, 0xff, 0xff, 1],
		];
		for i in 0..sizes.len() {
			let mut enc = Vec::new();
			encode_size_and_prefix(sizes[i], 0, &mut enc);
			assert_eq!(enc, encs[i]);
			let s_dec = decode_size(encs[i][0], &mut &encs[i][1..]);
			assert_eq!(s_dec, Ok(sizes[i]));
		}
	}
}
