use solana_program::{
    account_info::AccountInfo, program_error::ProgramError, pubkey::Pubkey,
    sysvar::instructions::load_instruction_at_checked,
};

use crate::error::MachineWalletError;
use crate::secp256r1::{SignatureOffsets, HEADER_SIZE, SIGNATURE_OFFSETS_SIZE};

/// Ed25519 precompile program ID.
/// Base58: `Ed25519SigVerify111111111111111111111111111`
pub const ED25519_PROGRAM_ID: Pubkey =
    solana_program::pubkey!("Ed25519SigVerify111111111111111111111111111");

/// Ed25519 public key is 32 bytes.
const PUBKEY_SIZE: usize = 32;

/// Expected size of the signed message (hash).
const MESSAGE_SIZE: usize = 32;

/// Result of verifying an Ed25519 precompile instruction.
#[derive(Debug)]
pub struct Ed25519VerifyResult {
    /// The Ed25519 public key (32 bytes).
    pub pubkey: [u8; PUBKEY_SIZE],
    /// The signed message (exactly 32 bytes).
    pub message: [u8; MESSAGE_SIZE],
}

/// Parse the Ed25519 precompile instruction data, extracting the pubkey and message.
///
/// This function operates on raw instruction data bytes and does **not** check the
/// program ID. It is intended for use by `threshold.rs` when scanning multiple
/// instructions in the transaction, as well as internally by
/// `verify_precompile_instruction`.
///
/// Validates:
/// 1. Exactly 1 signature (Phase 0)
/// 2. All instruction_index fields == 0xFFFF (data in same instruction)
/// 3. `message_data_size == 32`
/// 4. Offsets are in-bounds
pub fn parse_precompile_data(data: &[u8]) -> Result<Ed25519VerifyResult, ProgramError> {
    // Parse header: [0] = signature_count, [1] = padding
    if data.len() < HEADER_SIZE {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }

    let signature_count = data[0];
    if signature_count != 1 {
        // Phase 0: exactly one signature
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }

    // Parse signature offsets (starts after header)
    if data.len() < HEADER_SIZE + SIGNATURE_OFFSETS_SIZE {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }

    let offsets =
        SignatureOffsets::parse(&data[HEADER_SIZE..HEADER_SIZE + SIGNATURE_OFFSETS_SIZE])?;

    // All instruction_index fields must be 0xFFFF (data in same instruction)
    if offsets.signature_instruction_index != 0xFFFF
        || offsets.public_key_instruction_index != 0xFFFF
        || offsets.message_instruction_index != 0xFFFF
    {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }

    // Extract Ed25519 pubkey (32 bytes, no prefix validation needed)
    let pk_start = offsets.public_key_offset as usize;
    let pk_end = pk_start + PUBKEY_SIZE;
    if pk_end > data.len() {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }

    let mut pubkey = [0u8; PUBKEY_SIZE];
    pubkey.copy_from_slice(&data[pk_start..pk_end]);

    // Extract message (must be exactly 32 bytes)
    if offsets.message_data_size as usize != MESSAGE_SIZE {
        return Err(MachineWalletError::MessageMismatch.into());
    }
    let msg_start = offsets.message_data_offset as usize;
    let msg_end = msg_start + MESSAGE_SIZE;
    if msg_end > data.len() {
        return Err(MachineWalletError::InvalidSignatureOffsets.into());
    }

    let mut message = [0u8; MESSAGE_SIZE];
    message.copy_from_slice(&data[msg_start..msg_end]);

    Ok(Ed25519VerifyResult { pubkey, message })
}

/// Verify the Ed25519 precompile instruction at the given index.
///
/// The precompile cannot be called via CPI — it must be a separate instruction in the
/// transaction. If our program is executing, the precompile instruction already succeeded
/// (Solana runtime verifies all precompile instructions before executing any program).
///
/// We load the instruction via sysvar introspection and verify:
/// 1. Correct program ID (Ed25519 precompile)
/// 2. Exactly 1 signature (Phase 0)
/// 3. All data references point to the same instruction (instruction_index = 0xFFFF)
/// 4. `message_data_size == 32`
///
/// Returns the extracted pubkey and signed message.
pub fn verify_precompile_instruction(
    instructions_sysvar: &AccountInfo,
    precompile_ix_index: u8,
) -> Result<Ed25519VerifyResult, ProgramError> {
    // Load the instruction at the specified index from the sysvar
    let ix = load_instruction_at_checked(precompile_ix_index as usize, instructions_sysvar)
        .map_err(|_| MachineWalletError::InstructionMissing)?;

    // Verify it's the Ed25519 precompile
    if ix.program_id != ED25519_PROGRAM_ID {
        return Err(MachineWalletError::InvalidPrecompileInstruction.into());
    }

    parse_precompile_data(&ix.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid Ed25519 precompile instruction payload.
    ///
    /// Layout:
    ///   [0]       signature_count = 1
    ///   [1]       padding = 0
    ///   [2..16]   SignatureOffsets (14 bytes)
    ///   [16..80]  signature (64 bytes, all zeros for test)
    ///   [80..112] pubkey (32 bytes)
    ///   [112..144] message (32 bytes)
    fn build_valid_payload(pubkey: [u8; 32], message: [u8; 32]) -> Vec<u8> {
        // Fixed offsets layout — signature at 16, pubkey at 80, message at 112
        let sig_offset: u16 = 16;
        let pk_offset: u16 = 80;
        let msg_offset: u16 = 112;

        let mut data = Vec::with_capacity(144);
        data.push(1u8); // signature_count
        data.push(0u8); // padding

        // SignatureOffsets
        data.extend_from_slice(&sig_offset.to_le_bytes()); // signature_offset
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // signature_instruction_index
        data.extend_from_slice(&pk_offset.to_le_bytes()); // public_key_offset
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // public_key_instruction_index
        data.extend_from_slice(&msg_offset.to_le_bytes()); // message_data_offset
        data.extend_from_slice(&32u16.to_le_bytes()); // message_data_size
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // message_instruction_index

        assert_eq!(data.len(), 16); // HEADER_SIZE + SIGNATURE_OFFSETS_SIZE

        data.extend_from_slice(&[0u8; 64]); // signature placeholder (64 bytes) → ends at 80
        data.extend_from_slice(&pubkey); // pubkey 32 bytes → ends at 112
        data.extend_from_slice(&message); // message 32 bytes → ends at 144

        data
    }

    // --- SignatureOffsets parsing ---

    #[test]
    fn test_parse_offsets_valid() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u16.to_le_bytes()); // signature_offset
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // signature_instruction_index
        data.extend_from_slice(&200u16.to_le_bytes()); // public_key_offset
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // public_key_instruction_index
        data.extend_from_slice(&300u16.to_le_bytes()); // message_data_offset
        data.extend_from_slice(&32u16.to_le_bytes()); // message_data_size
        data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // message_instruction_index

        let offsets = SignatureOffsets::parse(&data).unwrap();
        assert_eq!(offsets.signature_offset, 100);
        assert_eq!(offsets.signature_instruction_index, 0xFFFF);
        assert_eq!(offsets.public_key_offset, 200);
        assert_eq!(offsets.public_key_instruction_index, 0xFFFF);
        assert_eq!(offsets.message_data_offset, 300);
        assert_eq!(offsets.message_data_size, 32);
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

    // --- Program ID constant ---

    #[test]
    fn test_ed25519_program_id_string() {
        assert_eq!(
            ED25519_PROGRAM_ID.to_string(),
            "Ed25519SigVerify111111111111111111111111111"
        );
    }

    // --- parse_precompile_data ---

    #[test]
    fn test_parse_precompile_data_valid() {
        let pubkey = [0x42u8; 32];
        let message = [0x99u8; 32];
        let data = build_valid_payload(pubkey, message);

        let result = parse_precompile_data(&data).unwrap();
        assert_eq!(result.pubkey, pubkey);
        assert_eq!(result.message, message);
    }

    #[test]
    fn test_parse_precompile_data_rejects_zero_signatures() {
        let mut data = build_valid_payload([0u8; 32], [0u8; 32]);
        data[0] = 0; // signature_count = 0
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidPrecompileInstruction as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_two_signatures() {
        let mut data = build_valid_payload([0u8; 32], [0u8; 32]);
        data[0] = 2; // signature_count = 2
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidPrecompileInstruction as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_wrong_message_size() {
        let mut data = build_valid_payload([0u8; 32], [0u8; 32]);
        // SignatureOffsets starts at data[2] (after 2-byte header).
        // message_data_size is at offset 10..12 within SignatureOffsets → data[12..14].
        let wrong_size: u16 = 64;
        data[12] = wrong_size.to_le_bytes()[0];
        data[13] = wrong_size.to_le_bytes()[1];
        assert_eq!(
            parse_precompile_data(&data).unwrap_err(),
            ProgramError::Custom(MachineWalletError::MessageMismatch as u32)
        );
    }

    #[test]
    fn test_parse_precompile_data_rejects_non_self_instruction_index() {
        let mut data = build_valid_payload([0u8; 32], [0u8; 32]);
        // signature_instruction_index is at data[4..6] (bytes 2..4 of SignatureOffsets = data[4..6])
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
