use solana_program::{
    account_info::AccountInfo, program_error::ProgramError, pubkey::Pubkey,
    sysvar::instructions::load_instruction_at_checked,
};

use crate::error::MachineWalletError;

/// Secp256r1 precompile program ID.
/// Base58: `Secp256r1SigVerify1111111111111111111111111`
pub const SECP256R1_PROGRAM_ID: Pubkey =
    solana_program::pubkey!("Secp256r1SigVerify1111111111111111111111111");

/// Size of the SignatureOffsets struct in the precompile instruction data.
/// Shared with ed25519.rs (identical layout across precompiles).
pub(crate) const SIGNATURE_OFFSETS_SIZE: usize = 14;

/// Precompile header size: 1 byte signature_count + 1 byte padding.
/// Shared with ed25519.rs (identical layout across precompiles).
pub(crate) const HEADER_SIZE: usize = 2;

/// Compressed P-256 public key is 33 bytes.
const PUBKEY_SIZE: usize = 33;

/// Parsed signature offsets from precompile instruction data.
/// Shared by secp256r1 and ed25519 precompiles (identical wire layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SignatureOffsets {
    pub signature_offset: u16,
    pub signature_instruction_index: u16,
    pub public_key_offset: u16,
    pub public_key_instruction_index: u16,
    pub message_data_offset: u16,
    pub message_data_size: u16,
    pub message_instruction_index: u16,
}

impl SignatureOffsets {
    /// Parse SignatureOffsets from a 14-byte slice.
    pub fn parse(data: &[u8]) -> Result<Self, ProgramError> {
        if data.len() < SIGNATURE_OFFSETS_SIZE {
            return Err(MachineWalletError::InvalidSignatureOffsets.into());
        }

        Ok(Self {
            signature_offset: u16::from_le_bytes([data[0], data[1]]),
            signature_instruction_index: u16::from_le_bytes([data[2], data[3]]),
            public_key_offset: u16::from_le_bytes([data[4], data[5]]),
            public_key_instruction_index: u16::from_le_bytes([data[6], data[7]]),
            message_data_offset: u16::from_le_bytes([data[8], data[9]]),
            message_data_size: u16::from_le_bytes([data[10], data[11]]),
            message_instruction_index: u16::from_le_bytes([data[12], data[13]]),
        })
    }
}

/// Expected size of the signed message (keccak256 hash).
const MESSAGE_SIZE: usize = 32;

/// Result of verifying a secp256r1 precompile instruction.
#[derive(Debug)]
pub struct PrecompileVerifyResult {
    /// The compressed P-256 public key (33 bytes).
    pub pubkey: [u8; PUBKEY_SIZE],
    /// The signed message (keccak256 hash, exactly 32 bytes).
    pub message: [u8; MESSAGE_SIZE],
}

/// Shared header / offsets / pubkey validation used by both sized and
/// variable-length precompile parsers. Returns the parsed offsets and pubkey
/// but does NOT slice the message — message slicing depends on whether the
/// caller enforces a fixed size (see `parse_precompile_data_sized`) or accepts
/// any declared size (see `parse_precompile_data_any_len`).
fn parse_offsets_and_pubkey(
    data: &[u8],
) -> Result<(SignatureOffsets, [u8; PUBKEY_SIZE]), ProgramError> {
    if data.len() < HEADER_SIZE {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }
    if data[0] != 1 {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }
    if data.len() < HEADER_SIZE + SIGNATURE_OFFSETS_SIZE {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }
    let offsets =
        SignatureOffsets::parse(&data[HEADER_SIZE..HEADER_SIZE + SIGNATURE_OFFSETS_SIZE])?;

    if offsets.signature_instruction_index != 0xFFFF
        || offsets.public_key_instruction_index != 0xFFFF
        || offsets.message_instruction_index != 0xFFFF
    {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }

    let pk_start = offsets.public_key_offset as usize;
    let pk_end = pk_start + PUBKEY_SIZE;
    if pk_end > data.len() {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }
    let mut pubkey = [0u8; PUBKEY_SIZE];
    pubkey.copy_from_slice(&data[pk_start..pk_end]);
    if pubkey[0] != 0x02 && pubkey[0] != 0x03 {
        return Err(MachineWalletError::PublicKeyMismatch.into());
    }

    Ok((offsets, pubkey))
}

/// Parse a secp256r1 precompile instruction, enforcing a caller-specified message length.
///
/// Shared primitive for both fixed-32-byte (execute hash) and variable-length
/// (WebAuthn `authenticatorData || SHA256(clientDataJSON)`) signed messages.
///
/// Validates:
/// 1. Exactly 1 signature (Phase 0)
/// 2. All instruction_index fields == 0xFFFF (data in same instruction)
/// 3. Valid compressed pubkey prefix (0x02 or 0x03)
/// 4. `message_data_size == expected_msg_len`
/// 5. Offsets in-bounds
///
/// Returns `(pubkey, message_slice)`. The message slice borrows from `data`.
pub fn parse_precompile_data_sized<'a>(
    data: &'a [u8],
    expected_msg_len: usize,
) -> Result<([u8; PUBKEY_SIZE], &'a [u8]), ProgramError> {
    let (offsets, pubkey) = parse_offsets_and_pubkey(data)?;

    if offsets.message_data_size as usize != expected_msg_len {
        return Err(MachineWalletError::MessageMismatch.into());
    }
    let msg_start = offsets.message_data_offset as usize;
    let msg_end = msg_start + expected_msg_len;
    if msg_end > data.len() {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }

    Ok((pubkey, &data[msg_start..msg_end]))
}

/// Parse a secp256r1 precompile instruction without asserting a specific
/// message length. Same validation as `parse_precompile_data_sized` except
/// step 4 is skipped. Used for the variable-length WebAuthn match path.
pub fn parse_precompile_data_any_len<'a>(
    data: &'a [u8],
) -> Result<([u8; PUBKEY_SIZE], &'a [u8]), ProgramError> {
    let (offsets, pubkey) = parse_offsets_and_pubkey(data)?;

    let msg_start = offsets.message_data_offset as usize;
    let msg_end = msg_start
        .checked_add(offsets.message_data_size as usize)
        .ok_or(MachineWalletError::InvalidSignatureOffsets)?;
    if msg_end > data.len() {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }

    Ok((pubkey, &data[msg_start..msg_end]))
}

/// Parse a secp256r1 precompile signing a fixed 32-byte message (keccak256 hash).
pub fn parse_precompile_data(data: &[u8]) -> Result<PrecompileVerifyResult, ProgramError> {
    let (pubkey, msg_slice) = parse_precompile_data_sized(data, MESSAGE_SIZE)?;
    let mut message = [0u8; MESSAGE_SIZE];
    message.copy_from_slice(msg_slice);
    Ok(PrecompileVerifyResult { pubkey, message })
}

/// Verify the secp256r1 precompile instruction at the given index.
///
/// The precompile cannot be called via CPI — it must be a separate instruction in the
/// transaction. If our program is executing, the precompile instruction already succeeded
/// (Solana runtime verifies all precompile instructions before executing any program).
///
/// We load the instruction via sysvar introspection and verify:
/// 1. Correct program ID (secp256r1 precompile)
/// 2. Exactly 1 signature (Phase 0)
/// 3. All data references point to the same instruction (instruction_index = 0xFFFF)
/// 4. Valid compressed pubkey prefix (0x02 or 0x03)
///
/// Returns the extracted pubkey and signed message.
pub fn verify_precompile_instruction(
    instructions_sysvar: &AccountInfo,
    precompile_ix_index: u8,
) -> Result<PrecompileVerifyResult, ProgramError> {
    // Load the instruction at the specified index from the sysvar
    let ix = load_instruction_at_checked(precompile_ix_index as usize, instructions_sysvar)
        .map_err(|_| MachineWalletError::InstructionMissing)?;

    // Verify it's the secp256r1 precompile
    if ix.program_id != SECP256R1_PROGRAM_ID {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }

    parse_precompile_data(&ix.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_offsets_valid() {
        // Build 14 bytes of offsets
        let mut data = Vec::new();
        data.extend_from_slice(&100u16.to_le_bytes()); // signature_offset
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // signature_instruction_index
        data.extend_from_slice(&200u16.to_le_bytes()); // public_key_offset
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // public_key_instruction_index
        data.extend_from_slice(&300u16.to_le_bytes()); // message_data_offset
        data.extend_from_slice(&64u16.to_le_bytes()); // message_data_size
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // message_instruction_index

        let offsets = SignatureOffsets::parse(&data).unwrap();
        assert_eq!(offsets.signature_offset, 100);
        assert_eq!(offsets.signature_instruction_index, 0xFFFF);
        assert_eq!(offsets.public_key_offset, 200);
        assert_eq!(offsets.public_key_instruction_index, 0xFFFF);
        assert_eq!(offsets.message_data_offset, 300);
        assert_eq!(offsets.message_data_size, 64);
        assert_eq!(offsets.message_instruction_index, 0xFFFF);
    }

    #[test]
    fn test_parse_offsets_too_short() {
        let data = [0u8; 10]; // less than 14 bytes
        let result = SignatureOffsets::parse(&data);
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidSignatureOffsets as u32)
        );
    }

    #[test]
    fn test_parse_offsets_exact_size() {
        let data = [0u8; 14];
        let offsets = SignatureOffsets::parse(&data).unwrap();
        assert_eq!(offsets.signature_offset, 0);
        assert_eq!(offsets.message_data_size, 0);
    }

    #[test]
    fn test_parse_offsets_with_extra_data() {
        let data = [0u8; 20]; // more than 14 bytes — should still work
        let offsets = SignatureOffsets::parse(&data).unwrap();
        assert_eq!(offsets.signature_offset, 0);
    }

    #[test]
    fn test_secp256r1_program_id_bytes() {
        // Verify the compile-time program ID matches the expected base58 address
        assert_eq!(
            SECP256R1_PROGRAM_ID.to_string(),
            "Secp256r1SigVerify1111111111111111111111111"
        );
    }

    // --- parse_precompile_data ---

    /// Build a minimal valid secp256r1 precompile instruction payload.
    ///
    /// Layout:
    ///   [0]       signature_count = 1
    ///   [1]       padding = 0
    ///   [2..16]   SignatureOffsets (14 bytes)
    ///   [16..80]  signature (64 bytes, all zeros for test)
    ///   [80..113] pubkey (33 bytes, compressed P-256)
    ///   [113..145] message (32 bytes)
    fn build_valid_payload(pubkey: [u8; 33], message: [u8; 32]) -> Vec<u8> {
        let sig_offset: u16 = 16;
        let pk_offset: u16 = 80;
        let msg_offset: u16 = 113;

        let mut data = Vec::with_capacity(145);
        data.push(1u8); // signature_count
        data.push(0u8); // padding

        // SignatureOffsets
        data.extend_from_slice(&sig_offset.to_le_bytes());
        data.extend_from_slice(&0xFFFFu16.to_le_bytes());
        data.extend_from_slice(&pk_offset.to_le_bytes());
        data.extend_from_slice(&0xFFFFu16.to_le_bytes());
        data.extend_from_slice(&msg_offset.to_le_bytes());
        data.extend_from_slice(&32u16.to_le_bytes());
        data.extend_from_slice(&0xFFFFu16.to_le_bytes());

        assert_eq!(data.len(), 16); // HEADER_SIZE + SIGNATURE_OFFSETS_SIZE

        data.extend_from_slice(&[0u8; 64]); // signature placeholder → ends at 80
        data.extend_from_slice(&pubkey); // pubkey 33 bytes → ends at 113
        data.extend_from_slice(&message); // message 32 bytes → ends at 145

        data
    }

    #[test]
    fn test_parse_precompile_data_valid() {
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02; // valid compressed prefix
        let message = [0x99u8; 32];
        let data = build_valid_payload(pubkey, message);

        let result = parse_precompile_data(&data).unwrap();
        assert_eq!(result.pubkey, pubkey);
        assert_eq!(result.message, message);
    }

    #[test]
    fn test_parse_precompile_data_rejects_zero_signatures() {
        let mut pubkey = [0u8; 33];
        pubkey[0] = 0x02;
        let mut data = build_valid_payload(pubkey, [0u8; 32]);
        data[0] = 0; // signature_count = 0
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidPrecompileInstruction as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_two_signatures() {
        let mut pubkey = [0u8; 33];
        pubkey[0] = 0x02;
        let mut data = build_valid_payload(pubkey, [0u8; 32]);
        data[0] = 2; // signature_count = 2
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidPrecompileInstruction as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_wrong_message_size() {
        let mut pubkey = [0u8; 33];
        pubkey[0] = 0x02;
        let mut data = build_valid_payload(pubkey, [0u8; 32]);
        // message_data_size is at offset 10..12 within SignatureOffsets → data[12..14]
        let wrong_size: u16 = 64;
        data[12] = wrong_size.to_le_bytes()[0];
        data[13] = wrong_size.to_le_bytes()[1];
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::MessageMismatch as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_invalid_prefix() {
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x04; // uncompressed prefix — invalid for compressed
        let data = build_valid_payload(pubkey, [0u8; 32]);
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::PublicKeyMismatch as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_non_self_instruction_index() {
        let mut pubkey = [0u8; 33];
        pubkey[0] = 0x02;
        let mut data = build_valid_payload(pubkey, [0u8; 32]);
        // signature_instruction_index at data[4..6]
        data[4] = 0x00;
        data[5] = 0x00; // instruction_index = 0 instead of 0xFFFF
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidPrecompileInstruction as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_too_short() {
        let data = [1u8; 5]; // shorter than HEADER_SIZE + SIGNATURE_OFFSETS_SIZE
        assert!(parse_precompile_data(&data).is_err());
    }
}
