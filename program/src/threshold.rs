use solana_program::{
    account_info::AccountInfo, keccak, program_error::ProgramError, pubkey::Pubkey,
    sysvar::instructions::load_instruction_at_checked,
};

use crate::{
    ed25519,
    error::MachineWalletError,
    instruction::{parse_webauthn_evidence_payload, PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR},
    secp256r1,
    state::{
        AuthoritySlot, MachineWallet, MAX_AUTHORITIES, SIG_SCHEME_ED25519, SIG_SCHEME_SECP256R1,
        SIG_SCHEME_WEBAUTHN,
    },
    webauthn::{self, MAX_AUTH_DATA_SIZE, MIN_AUTH_DATA_SIZE},
};

// Compile-time guarantee: matched_count (u8) cannot overflow.
const _: () = assert!(MAX_AUTHORITIES as usize <= u8::MAX as usize);

// Compile-time guarantee: a WebAuthn signed message (auth_data ‖ SHA256(cd))
// can never be exactly 32 bytes, so the Pass-2 `else if` routing is sound:
// `parse_precompile_data_sized(..., 32)` succeeds ⇒ SECP256R1 slot; failure
// ⇒ try WebAuthn. If this ever becomes false, WebAuthn messages could be
// matched against SECP256R1 authorities and vice versa.
const _: () = assert!(
    MAX_AUTH_DATA_SIZE + 32 > 32 && webauthn::MIN_AUTH_DATA_SIZE > 0,
    "WebAuthn msg size must differ from the 32-byte execute hash"
);

/// Maximum number of ProvideWebAuthnEvidence sidecar instructions honoured in
/// one transaction. A sidecar beyond this cap means the transaction is paying
/// for more WebAuthn parsing than any legitimate wallet could consume — fail
/// closed to bound Pass-1 CU cost.
///
/// Sized to match MAX_AUTHORITIES: even a fully passkey-backed N-of-N wallet
/// only needs N sidecars, and no wallet can have more authorities than this.
pub const MAX_WEBAUTHN_EVIDENCE: usize = MAX_AUTHORITIES as usize;

/// Scan a transaction for authority contributions matching a specific operation.
///
/// # Design: Evidence Sidecar Model
///
/// The wallet's authority list is scheme-tagged. Each scheme supplies its
/// signature evidence as an *independent unit* in the transaction's instruction
/// list, which the scanner below enumerates via the instructions sysvar:
///
/// - **SECP256R1 / ED25519**: the scheme's built-in Solana precompile — signs
///   the 32-byte `expected_execute_hash` directly.
/// - **WEBAUTHN**: a secp256r1 precompile signs `auth_data ‖ SHA256(cd)` while a
///   separate `ProvideWebAuthnEvidence` instruction in the same tx carries
///   `auth_data` and `clientDataJSON`, letting the scanner reconstruct the
///   precompile's signed bytes.
///
/// This means an M-of-N wallet can combine signers of different schemes (and
/// multiple signers of the same scheme, including multiple passkeys) in a
/// single transaction. Every authority's evidence is self-contained.
///
/// # Algorithm
///
/// **Pass 1** – sidecar collection. For each `ProvideWebAuthnEvidence`
/// instruction, try to build a WebAuthn signed message bound to the *current*
/// operation's `expected_execute_hash`:
/// - if validation succeeds (rpIdHash, UP/UV flags, type, challenge bound to
///   this op), store `keccak256(webauthn_msg)` in a scratch array;
/// - if validation fails (most commonly: challenge targets a *different*
///   operation in the same tx), silently skip — that sidecar legitimately
///   belongs to a different wallet instruction.
///
/// **Pass 2** – precompile matching.
/// - `secp256r1` + 32-byte message == `expected_execute_hash` → SECP256R1 slot.
/// - `secp256r1` + message of some valid WebAuthn size and `keccak256(msg)` in
///   the sidecar-hash set → WEBAUTHN slot.
/// - `ed25519` + 32-byte message == `expected_execute_hash` → ED25519 slot.
///
/// # Security properties
/// - Each authority is counted at most once (bitmap dedup).
/// - Scheme strictly corresponds to message format — no cross-scheme matching.
/// - `signature_count == 1` enforced by the precompile parsers.
/// - Early exit once threshold is met (saves CU).
/// - MAX_WEBAUTHN_EVIDENCE caps CU cost of Pass 1.
/// - Fail-open on per-sidecar validation: composable with multiple wallet
///   instructions in one tx, while a *truly* malformed sidecar contributes
///   nothing to any op (so it can only waste its own paid fee).
pub fn verify_threshold_signatures(
    instructions_sysvar: &AccountInfo,
    program_id: &Pubkey,
    authorities: &[AuthoritySlot],
    authority_count: u8,
    threshold: u8,
    expected_execute_hash: &[u8; 32],
) -> Result<(), ProgramError> {
    // Defense-in-depth: threshold must be positive regardless of caller guarantees.
    if threshold == 0 {
        return Err(MachineWalletError::InvalidThreshold.into());
    }

    let count = authority_count as usize;
    let has_webauthn_authority = authorities[..count]
        .iter()
        .any(|a| a.sig_scheme == SIG_SCHEME_WEBAUTHN);

    // Dispatch to a specialized variant. Non-WEBAUTHN wallets (the common
    // case — pure SECP256R1/ED25519) skip the 512B sidecar_hashes stack
    // allocation, Pass 1 scan, and Pass 2 variable-length match branch
    // entirely. The two variants share the same `match_authority` primitive,
    // so wallet-level security invariants (at-most-once counting, scheme-
    // strict matching, early-exit on threshold met) are identical.
    if has_webauthn_authority {
        verify_threshold_with_webauthn(
            instructions_sysvar,
            program_id,
            authorities,
            count,
            threshold,
            expected_execute_hash,
        )
    } else {
        verify_threshold_simple(
            instructions_sysvar,
            authorities,
            count,
            threshold,
            expected_execute_hash,
        )
    }
}

/// Threshold verification for wallets with at least one WEBAUTHN authority.
///
/// Two passes: (1) scan for ProvideWebAuthnEvidence sidecars to build
/// keccak256(auth_data ‖ SHA256(cd)) set; (2) match each precompile ix
/// against the authority list, including variable-length secp256r1 msgs
/// against the sidecar set.
#[inline(never)]
fn verify_threshold_with_webauthn(
    instructions_sysvar: &AccountInfo,
    program_id: &Pubkey,
    authorities: &[AuthoritySlot],
    count: usize,
    threshold: u8,
    expected_execute_hash: &[u8; 32],
) -> Result<(), ProgramError> {
    // ── Pass 1: collect WebAuthn evidence hashes ──────────────────────────────
    // 32B per sidecar (vs up to 544B of raw bytes) — keeps scratch footprint small.
    let mut sidecar_hashes: [[u8; 32]; MAX_WEBAUTHN_EVIDENCE] =
        [[0u8; 32]; MAX_WEBAUTHN_EVIDENCE];
    let mut sidecar_count: usize = 0;

    let expected_challenge_b64 = webauthn::encode_challenge(expected_execute_hash);
    let mut webauthn_buf = [0u8; MAX_AUTH_DATA_SIZE + 32];
    let mut ix_index: usize = 0;
    loop {
        let ix = match load_instruction_at_checked(ix_index, instructions_sysvar) {
            Ok(ix) => ix,
            Err(_) => break,
        };
        ix_index += 1;

        // Identify sidecars by (our program_id, discriminator = 14).
        if ix.program_id != *program_id {
            continue;
        }
        let Some((&disc, rest)) = ix.data.split_first() else {
            continue;
        };
        if disc != PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR {
            continue;
        }

        // Hard cap on sidecar count — bounds Pass-1 CU against adversarial stuffing.
        if sidecar_count >= MAX_WEBAUTHN_EVIDENCE {
            return Err(MachineWalletError::TooManyWebAuthnEvidence.into());
        }

        let Ok((auth_data, client_data_json)) = parse_webauthn_evidence_payload(rest) else {
            continue;
        };

        // Failure mode split:
        //  - `WebAuthnChallengeMismatch` is op-specific — the sidecar
        //    legitimately targets a DIFFERENT wallet op in this tx. Skip,
        //    preserving composability.
        //  - All other failures are op-invariant. Fail closed.
        match webauthn::build_webauthn_message(
            auth_data,
            client_data_json,
            &expected_challenge_b64,
            &mut webauthn_buf,
        ) {
            Ok(len) => {
                sidecar_hashes[sidecar_count] = keccak::hash(&webauthn_buf[..len]).to_bytes();
                sidecar_count += 1;
            }
            Err(e)
                if e == ProgramError::Custom(
                    MachineWalletError::WebAuthnChallengeMismatch as u32,
                ) =>
            {
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    // ── Pass 2: precompile matching ───────────────────────────────────────────
    let mut authority_matched = [false; MAX_AUTHORITIES as usize];
    let mut matched_count: u8 = 0;

    let mut ix_index: usize = 0;
    loop {
        let ix = match load_instruction_at_checked(ix_index, instructions_sysvar) {
            Ok(ix) => ix,
            Err(_) => break,
        };
        ix_index += 1;

        if ix.program_id == secp256r1::SECP256R1_PROGRAM_ID {
            // Fast path: precompile signing exactly 32 bytes → try SECP256R1.
            if let Ok((pubkey, msg)) = secp256r1::parse_precompile_data_sized(&ix.data, 32) {
                if msg == expected_execute_hash {
                    match_authority(
                        authorities,
                        count,
                        &mut authority_matched,
                        &mut matched_count,
                        SIG_SCHEME_SECP256R1,
                        &pubkey,
                    );
                }
            } else if sidecar_count > 0 {
                // Variable-length message → try matching against each sidecar.
                // Skip the keccak syscall when the message length is outside
                // any valid WebAuthn range (auth_data ‖ SHA256(cd)).
                if let Ok((pubkey, msg_bytes)) = secp256r1::parse_precompile_data_any_len(&ix.data)
                {
                    const MIN_WEBAUTHN_MSG: usize = MIN_AUTH_DATA_SIZE + 32;
                    const MAX_WEBAUTHN_MSG: usize = MAX_AUTH_DATA_SIZE + 32;
                    if (MIN_WEBAUTHN_MSG..=MAX_WEBAUTHN_MSG).contains(&msg_bytes.len()) {
                        let msg_hash = keccak::hash(msg_bytes).to_bytes();
                        if (0..sidecar_count).any(|s| sidecar_hashes[s] == msg_hash) {
                            match_authority(
                                authorities,
                                count,
                                &mut authority_matched,
                                &mut matched_count,
                                SIG_SCHEME_WEBAUTHN,
                                &pubkey,
                            );
                        }
                    }
                }
            }
        } else if ix.program_id == ed25519::ED25519_PROGRAM_ID {
            if let Ok(result) = ed25519::parse_precompile_data(&ix.data) {
                if result.message == *expected_execute_hash {
                    let mut padded_pubkey = [0u8; 33];
                    padded_pubkey[..32].copy_from_slice(&result.pubkey);
                    match_authority(
                        authorities,
                        count,
                        &mut authority_matched,
                        &mut matched_count,
                        SIG_SCHEME_ED25519,
                        &padded_pubkey,
                    );
                }
            }
        }

        if matched_count >= threshold {
            return Ok(());
        }
    }

    Err(MachineWalletError::InsufficientSignatures.into())
}

/// Threshold verification for wallets with ONLY SECP256R1 / ED25519
/// authorities. Skips the 512B sidecar scratch buffer, Pass 1 scan, and the
/// Pass 2 variable-length match branch entirely.
///
/// Security: the skipped branch can only match WEBAUTHN-scheme authorities,
/// and this function is called exclusively when no such authority exists —
/// so the branch would contribute 0 to `matched_count` anyway. Omitting it
/// changes CU cost, not behavior.
#[inline(never)]
fn verify_threshold_simple(
    instructions_sysvar: &AccountInfo,
    authorities: &[AuthoritySlot],
    count: usize,
    threshold: u8,
    expected_execute_hash: &[u8; 32],
) -> Result<(), ProgramError> {
    let mut authority_matched = [false; MAX_AUTHORITIES as usize];
    let mut matched_count: u8 = 0;

    let mut ix_index: usize = 0;
    loop {
        let ix = match load_instruction_at_checked(ix_index, instructions_sysvar) {
            Ok(ix) => ix,
            Err(_) => break,
        };
        ix_index += 1;

        if ix.program_id == secp256r1::SECP256R1_PROGRAM_ID {
            if let Ok((pubkey, msg)) = secp256r1::parse_precompile_data_sized(&ix.data, 32) {
                if msg == expected_execute_hash {
                    match_authority(
                        authorities,
                        count,
                        &mut authority_matched,
                        &mut matched_count,
                        SIG_SCHEME_SECP256R1,
                        &pubkey,
                    );
                }
            }
        } else if ix.program_id == ed25519::ED25519_PROGRAM_ID {
            if let Ok(result) = ed25519::parse_precompile_data(&ix.data) {
                if result.message == *expected_execute_hash {
                    let mut padded_pubkey = [0u8; 33];
                    padded_pubkey[..32].copy_from_slice(&result.pubkey);
                    match_authority(
                        authorities,
                        count,
                        &mut authority_matched,
                        &mut matched_count,
                        SIG_SCHEME_ED25519,
                        &padded_pubkey,
                    );
                }
            }
        }

        if matched_count >= threshold {
            return Ok(());
        }
    }

    Err(MachineWalletError::InsufficientSignatures.into())
}

/// Try to mark the first unmatched authority that matches (scheme, pubkey).
/// For ED25519, caller passes a zero-padded 33-byte pubkey so the comparison
/// is uniform.
fn match_authority(
    authorities: &[AuthoritySlot],
    count: usize,
    authority_matched: &mut [bool; MAX_AUTHORITIES as usize],
    matched_count: &mut u8,
    scheme: u8,
    pubkey: &[u8; 33],
) {
    for i in 0..count {
        if !authority_matched[i]
            && authorities[i].sig_scheme == scheme
            && authorities[i].pubkey == *pubkey
        {
            authority_matched[i] = true;
            *matched_count += 1;
            return;
        }
    }
}

/// Unified wallet signature verification.
///
/// v0 wallets (single SECP256R1 authority, legacy): verifies one secp256r1
/// precompile instruction at `precompile_ix_index` against the stored authority.
///
/// v1 wallets: scans the transaction for every scheme-appropriate evidence
/// contribution — SECP256R1/ED25519 precompile instructions directly, and
/// WEBAUTHN via paired precompile + ProvideWebAuthnEvidence sidecar
/// (see `verify_threshold_signatures` for the algorithm). `precompile_ix_index`
/// is ignored on v1.
///
/// This is the single entry point for every wallet-authorized operation. Future
/// signature schemes (post-quantum, etc.) extend this by:
///   1. adding a new SIG_SCHEME_* constant in `state.rs`,
///   2. adding a Pass-2 branch inside `verify_threshold_signatures` for the
///      new precompile program,
///   3. (if the scheme needs auxiliary data) adding a new sidecar discriminator
///      and Pass-1 collection branch.
/// Wallet instruction wire formats are not affected.
pub fn verify_wallet_signatures(
    instructions_sysvar: &AccountInfo,
    program_id: &Pubkey,
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
        return Ok(());
    }
    verify_threshold_signatures(
        instructions_sysvar,
        program_id,
        &wallet.authorities,
        wallet.authority_count,
        wallet.threshold,
        expected_message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authority_matched_bitmap_size() {
        let bitmap = [false; MAX_AUTHORITIES as usize];
        assert_eq!(bitmap.len(), 16);
    }

    #[test]
    fn test_max_webauthn_evidence_matches_authorities() {
        assert_eq!(MAX_WEBAUTHN_EVIDENCE, MAX_AUTHORITIES as usize);
    }

    #[test]
    fn test_sig_scheme_constants() {
        assert_eq!(SIG_SCHEME_SECP256R1, 0);
        assert_eq!(SIG_SCHEME_ED25519, 1);
        assert_eq!(SIG_SCHEME_WEBAUTHN, 2);
    }

    #[test]
    fn test_match_authority_marks_first_free_slot() {
        let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        let mut pubkey = [0x02u8; 33];
        pubkey[1] = 0xAA;
        authorities[0] = AuthoritySlot {
            sig_scheme: SIG_SCHEME_SECP256R1,
            pubkey,
        };
        let mut matched = [false; MAX_AUTHORITIES as usize];
        let mut count: u8 = 0;
        match_authority(
            &authorities,
            1,
            &mut matched,
            &mut count,
            SIG_SCHEME_SECP256R1,
            &pubkey,
        );
        assert!(matched[0]);
        assert_eq!(count, 1);

        // Second call with same pubkey must NOT double-count.
        match_authority(
            &authorities,
            1,
            &mut matched,
            &mut count,
            SIG_SCHEME_SECP256R1,
            &pubkey,
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn test_match_authority_rejects_wrong_scheme() {
        let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        let mut pubkey = [0x02u8; 33];
        pubkey[1] = 0xAA;
        authorities[0] = AuthoritySlot {
            sig_scheme: SIG_SCHEME_SECP256R1,
            pubkey,
        };
        let mut matched = [false; MAX_AUTHORITIES as usize];
        let mut count: u8 = 0;
        // Same pubkey but request WEBAUTHN match — must not hit the SECP256R1 slot.
        match_authority(
            &authorities,
            1,
            &mut matched,
            &mut count,
            SIG_SCHEME_WEBAUTHN,
            &pubkey,
        );
        assert!(!matched[0]);
        assert_eq!(count, 0);
    }

}
