use crate::{
	nibble::NibbleSlice,
	recorder::Recorder,
	rstd::{result::Result, vec::Vec},
	CError, Result as TrieResult, Trie, TrieHash, TrieLayout,
};
use hash_db::Hasher;

/// Generate an eip-1186 compatible proof for key-value pairs in a trie given a key.
pub fn generate_proof<T, L>(
	trie: &T,
	key: &[u8],
) -> TrieResult<(Vec<Vec<u8>>, Option<Vec<u8>>), TrieHash<L>, CError<L>>
where
	T: Trie<L>,
	L: TrieLayout,
{
	let mut recorder = Recorder::new();
	let item = trie.get_with(key, &mut recorder)?;
	let proof: Vec<Vec<u8>> = recorder.drain().into_iter().map(|r| r.data).collect();
	Ok((proof, item))
}

/// Errors that may occur during proof verification. Most of the errors types simply indicate that
/// the proof is invalid with respect to the statement being verified, and the exact error type can
/// be used for debugging.
#[derive(PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum VerifyError<'a, HO, CE> {
	/// The proof does not contain any value for the given key
	/// the error carries the nibbles left after traversing the trie
	NonExistingValue(NibbleSlice<'a>),
	/// The proof contains a value for the given key
	/// while we were expecting to find a non-existence proof
	ExistingValue(Vec<u8>),
	/// The proof indicates that the trie contains a different value.
	/// the error carries the value contained in the trie
	ValueMismatch(Vec<u8>),
	/// The proof is missing trie nodes required to verify.
	IncompleteProof,
	/// The node hash computed from the proof is not matching.
	HashMismatch(HO),
	/// One of the proof nodes could not be decoded.
	DecodeError(CE),
	/// Error in converting a plain hash into a HO
	HashDecodeError(&'a [u8]),
}

#[cfg(feature = "std")]
impl<'a, HO: std::fmt::Debug, CE: std::error::Error> std::fmt::Display for VerifyError<'a, HO, CE> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
		match self {
			VerifyError::NonExistingValue(key) =>
				write!(f, "Key does not exist in trie: reaming key={:?}", key),
			VerifyError::ExistingValue(value) =>
				write!(f, "trie contains a value for given key value={:?}", value),
			VerifyError::ValueMismatch(key) =>
				write!(f, "Expected value was not found in the trie: key={:?}", key),
			VerifyError::IncompleteProof => write!(f, "Proof is incomplete -- expected more nodes"),
			VerifyError::HashMismatch(hash) => write!(f, "hash mismatch found: hash={:?}", hash),
			VerifyError::DecodeError(err) => write!(f, "Unable to decode proof node: {}", err),
			VerifyError::HashDecodeError(plain_hash) =>
				write!(f, "Unable to decode hash value plain_hash: {:?}", plain_hash),
		}
	}
}

#[cfg(feature = "std")]
impl<'a, HO: std::fmt::Debug, CE: std::error::Error + 'static> std::error::Error
	for VerifyError<'a, HO, CE>
{
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			VerifyError::DecodeError(err) => Some(err),
			_ => None,
		}
	}
}

/// Verify a compact proof for key-value pairs in a trie given a root hash.
pub fn verify_proof<'a, L>(
	root: &<L::Hash as Hasher>::Out,
	proof: &'a [Vec<u8>],
	raw_key: &'a [u8],
	expected_value: Option<&[u8]>,
) -> Result<(), VerifyError<'a, TrieHash<L>, CError<L>>>
where
	L: TrieLayout,
{
	crate::proof::verify_proof::<L, _, _, _>(root, proof, vec![&(raw_key, expected_value)]).unwrap();
	Ok(())
}
