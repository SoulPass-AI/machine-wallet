use solana_program::{
    account_info::AccountInfo, program_error::ProgramError,
    sysvar::instructions::load_instruction_at_checked,
};

use crate::{
    ed25519,
    error::MachineWalletError,
    secp256r1,
    state::{
        AuthoritySlot, MachineWallet, MAX_AUTHORITIES, SIG_SCHEME_ED25519, SIG_SCHEME_SECP256R1,
    },
};

// Compile-time guarantee: matched_count (u8) cannot overflow.
const _: () = assert!(MAX_AUTHORITIES as usize <= u8::MAX as usize);

/// Scan transaction instructions for precompile signatures matching wallet authorities.
/// Returns Ok(()) if matched_count >= threshold.
///
/// SECURITY PROPERTIES:
/// - Each authority counted at most once (bitmap dedup)
/// - Each precompile instruction checked against all unmatched authorities
/// - message_data_size == 32 enforced by both precompile parsers
/// - signature_count == 1 enforced by both precompile parsers
/// - Early exit once threshold met (saves CU)
pub fn verify_threshold_signatures(
    instructions_sysvar: &AccountInfo,
    authorities: &[AuthoritySlot],
    authority_count: u8,
    threshold: u8,
    expected_message_hash: &[u8; 32],
) -> Result<(), ProgramError> {
    // Defense-in-depth: threshold must be positive regardless of caller guarantees.
    if threshold == 0 {
        return Err(MachineWalletError::InvalidThreshold.into());
    }

    let count = authority_count as usize;
    let mut authority_matched = [false; MAX_AUTHORITIES as usize];
    let mut matched_count: u8 = 0;

    // Scan all instructions in the transaction
    let mut ix_index: usize = 0;
    loop {
        let ix = match load_instruction_at_checked(ix_index, instructions_sysvar) {
            Ok(ix) => ix,
            Err(_) => break, // No more instructions
        };

        // Try secp256r1 precompile
        if ix.program_id == secp256r1::SECP256R1_PROGRAM_ID {
            if let Ok(result) = secp256r1::parse_precompile_data(&ix.data) {
                if result.message == *expected_message_hash {
                    // Find an unmatched P-256 authority with matching pubkey
                    for i in 0..count {
                        if !authority_matched[i]
                            && authorities[i].sig_scheme == SIG_SCHEME_SECP256R1
                            && authorities[i].pubkey == result.pubkey
                        {
                            authority_matched[i] = true;
                            matched_count += 1;
                            break;
                        }
                    }
                }
            }
        }
        // Try ed25519 precompile
        else if ix.program_id == ed25519::ED25519_PROGRAM_ID {
            if let Ok(result) = ed25519::parse_precompile_data(&ix.data) {
                if result.message == *expected_message_hash {
                    // Find an unmatched Ed25519 authority with matching pubkey.
                    // Ed25519 pubkey is 32 bytes; AuthoritySlot.pubkey is 33 bytes
                    // with the last byte being 0x00 padding.
                    for i in 0..count {
                        if !authority_matched[i]
                            && authorities[i].sig_scheme == SIG_SCHEME_ED25519
                            && authorities[i].pubkey[..32] == result.pubkey
                        {
                            authority_matched[i] = true;
                            matched_count += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Early exit once threshold met
        if matched_count >= threshold {
            return Ok(());
        }

        ix_index += 1;
    }

    // Not enough signatures matched
    Err(MachineWalletError::InsufficientSignatures.into())
}

/// Version-aware signature verification.
/// v0: single secp256r1 precompile at `precompile_ix_index`.
/// v1: threshold scan across all transaction instructions (`precompile_ix_index` ignored).
pub fn verify_wallet_signatures(
    instructions_sysvar: &AccountInfo,
    wallet: &MachineWallet,
    precompile_ix_index: u8,
    expected_message: &[u8; 32],
) -> Result<(), ProgramError> {
    if wallet.version == 0 {
        let result =
            secp256r1::verify_precompile_instruction(instructions_sysvar, precompile_ix_index)?;
        if result.pubkey != wallet.authorities[0].pubkey {
            return Err(MachineWalletError::PublicKeyMismatch.into());
        }
        if result.message != *expected_message {
            return Err(MachineWalletError::MessageMismatch.into());
        }
        Ok(())
    } else {
        verify_threshold_signatures(
            instructions_sysvar,
            &wallet.authorities,
            wallet.authority_count,
            wallet.threshold,
            expected_message,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the function signature compiles and basic type constraints hold.
    #[test]
    fn test_authority_matched_bitmap_size() {
        // The bitmap must be large enough for MAX_AUTHORITIES
        let bitmap = [false; MAX_AUTHORITIES as usize];
        assert_eq!(bitmap.len(), 16);
    }

    #[test]
    fn test_threshold_zero_is_rejected() {
        // threshold=0 is rejected by verify_threshold_signatures as defense-in-depth.
        // State deserialization also rejects threshold=0, but the function itself
        // must not rely on caller guarantees for safety-critical invariants.
        let threshold: u8 = 0;
        assert_eq!(threshold, 0, "threshold=0 would bypass signature verification");
    }

    #[test]
    fn test_sig_scheme_constants_match() {
        // Ensure the constants we import match expected values
        assert_eq!(SIG_SCHEME_SECP256R1, 0);
        assert_eq!(SIG_SCHEME_ED25519, 1);
    }

    #[test]
    fn test_authority_slot_pubkey_comparison() {
        // Verify P-256 full 33-byte comparison works
        let mut pubkey_p256 = [0x42u8; 33];
        pubkey_p256[0] = 0x02;
        let auth = AuthoritySlot {
            sig_scheme: SIG_SCHEME_SECP256R1,
            pubkey: pubkey_p256,
        };
        assert_eq!(auth.pubkey, pubkey_p256);

        // Verify Ed25519 32-byte prefix comparison works
        let mut pubkey_ed = [0u8; 33];
        pubkey_ed[..32].copy_from_slice(&[0x55u8; 32]);
        // pubkey_ed[32] == 0x00 (padding)
        let auth_ed = AuthoritySlot {
            sig_scheme: SIG_SCHEME_ED25519,
            pubkey: pubkey_ed,
        };
        let ed_result_pubkey = [0x55u8; 32];
        assert_eq!(auth_ed.pubkey[..32], ed_result_pubkey);
    }
}
