use solana_program::program_error::ProgramError;

use crate::error::MachineWalletError;
use crate::state::MAX_ALLOWED_PROGRAMS;
use crate::webauthn::{MAX_AUTH_DATA_SIZE, MAX_CLIENT_DATA_JSON_SIZE, MIN_AUTH_DATA_SIZE};

/// Maximum number of inner instructions in a single Execute call.
/// Prevents OOM from malicious count values and keeps compute budget reasonable.
pub const MAX_INNER_INSTRUCTIONS: usize = 64;

/// Parse the payload of a ProvideWebAuthnEvidence instruction (i.e. the bytes
/// *after* the discriminator).
///
/// Wire format:
///   auth_data_len        : u16 LE
///   auth_data            : auth_data_len bytes
///   client_data_json_len : u16 LE
///   client_data_json     : client_data_json_len bytes
///
/// Structural invariants enforced here — the downstream semantic validator
/// (`webauthn::build_webauthn_message`) never has to reject inputs for being
/// the wrong size or empty; it only rejects for cryptographic / policy reasons
/// (wrong rpIdHash, UP/UV cleared, challenge mismatch, etc.). Keeping these
/// layers separate is first-principles defense-in-depth.
pub(crate) fn parse_webauthn_evidence_payload(
    data: &[u8],
) -> Result<(&[u8], &[u8]), ProgramError> {
    if data.len() < 4 {
        return Err(ProgramError::InvalidInstructionData);
    }
    let auth_data_len = u16::from_le_bytes([data[0], data[1]]) as usize;
    if auth_data_len < MIN_AUTH_DATA_SIZE || auth_data_len > MAX_AUTH_DATA_SIZE {
        return Err(ProgramError::InvalidInstructionData);
    }
    let cd_len_off = 2usize
        .checked_add(auth_data_len)
        .ok_or(ProgramError::InvalidInstructionData)?;
    let cd_len_end = cd_len_off
        .checked_add(2)
        .ok_or(ProgramError::InvalidInstructionData)?;
    if cd_len_end > data.len() {
        return Err(ProgramError::InvalidInstructionData);
    }
    let auth_data = &data[2..cd_len_off];
    let client_data_json_len =
        u16::from_le_bytes([data[cd_len_off], data[cd_len_off + 1]]) as usize;
    if client_data_json_len == 0 || client_data_json_len > MAX_CLIENT_DATA_JSON_SIZE {
        return Err(ProgramError::InvalidInstructionData);
    }
    let cd_start = cd_len_end;
    let cd_end = cd_start
        .checked_add(client_data_json_len)
        .ok_or(ProgramError::InvalidInstructionData)?;
    if cd_end != data.len() {
        return Err(ProgramError::InvalidInstructionData);
    }
    Ok((auth_data, &data[cd_start..cd_end]))
}

/// A single account entry for CPI: index into remaining_accounts + permission flags.
/// The authority signs these flags, preventing relay manipulation of is_writable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccountEntry {
    pub index: u8,
    /// Bit 0: is_writable
    pub flags: u8,
}

impl AccountEntry {
    pub const FLAG_WRITABLE: u8 = 0x01;

    pub fn is_writable(&self) -> bool {
        self.flags & Self::FLAG_WRITABLE != 0
    }
}

/// A CPI instruction to be executed by the wallet vault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnerInstruction {
    pub program_id: [u8; 32],
    pub accounts: Vec<AccountEntry>,
    pub data: Vec<u8>,
}

/// Borrowed inner instruction view used by the on-chain execute hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnerInstructionRef<'a> {
    pub program_id: [u8; 32],
    account_bytes: &'a [u8],
    pub data: &'a [u8],
}

impl InnerInstructionRef<'_> {
    pub fn new<'a>(
        program_id: [u8; 32],
        account_bytes: &'a [u8],
        data: &'a [u8],
    ) -> InnerInstructionRef<'a> {
        InnerInstructionRef {
            program_id,
            account_bytes,
            data,
        }
    }

    pub fn account_count(&self) -> usize {
        self.account_bytes.len() / 2
    }

    pub fn account_bytes(&self) -> &[u8] {
        self.account_bytes
    }

    pub fn accounts(&self) -> impl ExactSizeIterator<Item = AccountEntry> + '_ {
        self.account_bytes
            .chunks_exact(2)
            .map(|chunk| AccountEntry {
                index: chunk[0],
                flags: chunk[1],
            })
    }
}

/// Instruction variants for the MachineWallet program.
///
/// The wallet uses the **Evidence Sidecar** model for multi-scheme signing.
/// Wallet instructions (Execute, management ops) carry only the operation's
/// parameters; signature evidence lives in **independent sidecar instructions**
/// within the same transaction:
///
/// - **SECP256R1 / ED25519**: the scheme's precompile instruction (signing the
///   32-byte operation hash) is its own evidence — standard Solana flow.
/// - **WEBAUTHN**: one secp256r1 precompile (signing `auth_data ‖ SHA256(cd)`)
///   *plus* one `ProvideWebAuthnEvidence` instruction carrying `auth_data` and
///   `clientDataJSON` so the on-chain scanner can reconstruct the signed bytes.
///
/// This symmetry means an M-of-N wallet can combine arbitrarily many signers
/// of arbitrary schemes in one transaction — each signer contributes an
/// independent set of ixs, and the threshold scanner in `threshold.rs` counts
/// all valid contributions.
///
/// Wire format:
/// - CreateWallet:    [0] + authority (33 bytes) OR [0] + sig_scheme (u8) + authority (33 bytes)
/// - Execute:         [1] + secp256r1_ix_index (u8) + max_slot (u64 LE) + inner_instruction_count (u32 LE) + inner instructions
/// - CloseWallet:     [2] + secp256r1_ix_index (u8) + max_slot (u64 LE) + destination (32 bytes)
/// - AdvanceNonce:    [3] + secp256r1_ix_index (u8) + max_slot (u64 LE)
/// - CreateSession:   [4] + secp256r1_ix_index (u8) + max_slot (u64 LE) + session_authority (32) + expiry_slot (u64 LE) + max_lamports_per_call (u64 LE) + max_total_spent_lamports (u64 LE) + allowed_programs_count (u8) + allowed_programs (count*32)
/// - SessionExecute:  [5] + inner_instruction_count (u32 LE) + inner instructions
/// - RevokeSession:   [6] + secp256r1_ix_index (u8) + max_slot (u64 LE) + session_authority (32)
/// - SelfRevokeSession: [7]
/// - CloseSession:    [8]  — accounts: [session_account, authority (signer), destination]
/// - AddAuthority:    [9] + precompile_ix_index (u8) + new_sig_scheme (u8) + new_pubkey (33) + new_threshold (u8) + max_slot (u64 LE)
/// - RemoveAuthority: [10] + precompile_ix_index (u8) + remove_sig_scheme (u8) + remove_pubkey (33) + new_threshold (u8) + max_slot (u64 LE)
/// - SetThreshold:    [11] + precompile_ix_index (u8) + new_threshold (u8) + max_slot (u64 LE)
/// - OwnerCloseSession: [12] + precompile_ix_index (u8) + max_slot (u64 LE) + session_authority (32) + destination (32)
/// - ProvideWebAuthnEvidence: [14] + auth_data_len (u16 LE) + auth_data + client_data_json_len (u16 LE) + client_data_json
///   Sidecar instruction — no state change. Its presence in the tx makes the
///   carried auth_data / clientDataJSON available to every wallet instruction
///   in the same tx for WEBAUTHN-scheme threshold contribution.
///
/// Discriminator 13 is reserved.
///
/// Inner instruction wire format:
///   program_id (32) + accounts_len (u16 LE) + data_len (u16 LE) + accounts (accounts_len * 2 bytes: index + flags) + data
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // Box would heap-allocate on SBF — unacceptable
pub enum MachineWalletInstruction<'a> {
    /// Initialize a new wallet with the given authority.
    /// sig_scheme=0 means Secp256r1 (backward-compatible default).
    CreateWallet { sig_scheme: u8, authority: [u8; 33] },

    /// Execute one or more inner instructions, verified by secp256r1 precompile.
    Execute {
        secp256r1_ix_index: u8,
        max_slot: u64,
        inner_instructions: Vec<InnerInstructionRef<'a>>,
    },

    /// Close the wallet and reclaim rent, verified by secp256r1 precompile.
    /// destination is included in the signed message to prevent relay redirection.
    CloseWallet {
        secp256r1_ix_index: u8,
        max_slot: u64,
        destination: [u8; 32],
    },

    /// Advance nonce without executing anything (cancel a pending signed operation).
    AdvanceNonce {
        secp256r1_ix_index: u8,
        max_slot: u64,
    },

    /// Create a session key, verified by wallet signatures.
    CreateSession {
        secp256r1_ix_index: u8,
        max_slot: u64,
        session_authority: [u8; 32],
        expiry_slot: u64,
        /// Net vault outflow cap per SessionExecute call (0 = unlimited).
        max_lamports_per_call: u64,
        /// Lifetime cumulative outflow cap across all SessionExecute calls under
        /// this session (0 = unlimited). When > 0, bounds the blast radius of a
        /// compromised session key independent of how many transactions it submits.
        max_total_spent_lamports: u64,
        allowed_programs_count: u8,
        allowed_programs: [[u8; 32]; MAX_ALLOWED_PROGRAMS],
    },

    /// Execute inner instructions using a session key (Ed25519 signer, no precompile).
    SessionExecute {
        inner_instructions: Vec<InnerInstructionRef<'a>>,
    },

    /// Revoke a session (owner path, requires secp256r1 precompile).
    RevokeSession {
        secp256r1_ix_index: u8,
        max_slot: u64,
        session_authority: [u8; 32],
    },

    /// Self-revoke a session (session authority signs directly, no precompile).
    SelfRevokeSession,

    /// Close an expired or revoked session to reclaim rent lamports.
    /// Session authority signs the transaction (Ed25519).
    CloseSession,

    /// Add a new authority to the wallet. Reallocs account.
    AddAuthority {
        precompile_ix_index: u8,
        new_sig_scheme: u8,
        new_pubkey: [u8; 33],
        new_threshold: u8, // 0 = keep current
        max_slot: u64,
    },

    /// Remove an authority from the wallet. Reallocs account smaller.
    RemoveAuthority {
        precompile_ix_index: u8,
        remove_sig_scheme: u8,
        remove_pubkey: [u8; 33],
        new_threshold: u8, // 0 = keep current
        max_slot: u64,
    },

    /// Change the threshold without adding/removing authorities.
    SetThreshold {
        precompile_ix_index: u8,
        new_threshold: u8,
        max_slot: u64,
    },

    /// Wallet-owner path to close a revoked or expired session and reclaim rent.
    OwnerCloseSession {
        precompile_ix_index: u8,
        max_slot: u64,
        session_authority: [u8; 32],
        destination: [u8; 32],
    },

    /// Sidecar instruction carrying a WebAuthn passkey assertion (authenticatorData
    /// + clientDataJSON) so that the tx-wide threshold scanner can reconstruct
    /// the signed message for a paired secp256r1 precompile.
    ///
    /// This instruction does NOT mutate any state. Its only purpose is to live in
    /// the tx's instruction list so that `threshold::verify_threshold_signatures`
    /// can see it via the instructions sysvar. One sidecar contributes at most
    /// one WEBAUTHN-scheme authority toward a wallet instruction's threshold.
    ProvideWebAuthnEvidence {
        auth_data: &'a [u8],
        client_data_json: &'a [u8],
    },
}

/// Parse inner instructions from a byte slice starting at the count field.
/// Shared by Execute and SessionExecute parsers.
fn parse_inner_instructions(data: &[u8]) -> Result<Vec<InnerInstructionRef<'_>>, ProgramError> {
    if data.len() < 4 {
        return Err(ProgramError::InvalidInstructionData);
    }
    let count = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;

    if count > MAX_INNER_INSTRUCTIONS {
        return Err(MachineWalletError::TooManyInnerInstructions.into());
    }

    let mut offset = 4;
    let mut inner_instructions = Vec::with_capacity(count);
    for _ in 0..count {
        if offset + 32 > data.len() {
            return Err(ProgramError::InvalidInstructionData);
        }
        let mut program_id = [0u8; 32];
        program_id.copy_from_slice(&data[offset..offset + 32]);
        offset += 32;

        if offset + 4 > data.len() {
            return Err(ProgramError::InvalidInstructionData);
        }
        let accounts_len =
            u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
        let data_len =
            u16::from_le_bytes(data[offset + 2..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let accounts_bytes = accounts_len
            .checked_mul(2)
            .ok_or(ProgramError::InvalidInstructionData)?;
        let accounts_end = offset
            .checked_add(accounts_bytes)
            .ok_or(ProgramError::InvalidInstructionData)?;
        if accounts_end > data.len() {
            return Err(ProgramError::InvalidInstructionData);
        }
        let account_bytes = &data[offset..accounts_end];
        for chunk in account_bytes.chunks_exact(2) {
            if chunk[1] & !AccountEntry::FLAG_WRITABLE != 0 {
                return Err(ProgramError::InvalidInstructionData);
            }
        }
        offset = accounts_end;

        let data_end = offset
            .checked_add(data_len)
            .ok_or(ProgramError::InvalidInstructionData)?;
        if data_end > data.len() {
            return Err(ProgramError::InvalidInstructionData);
        }
        let ix_data = &data[offset..data_end];
        offset = data_end;

        inner_instructions.push(InnerInstructionRef {
            program_id,
            account_bytes,
            data: ix_data,
        });
    }

    if offset != data.len() {
        return Err(ProgramError::InvalidInstructionData);
    }

    Ok(inner_instructions)
}

impl<'a> MachineWalletInstruction<'a> {
    /// Deserialize instruction data.
    pub fn unpack(data: &'a [u8]) -> Result<Self, ProgramError> {
        let (&discriminator, rest) = data
            .split_first()
            .ok_or(ProgramError::InvalidInstructionData)?;

        match discriminator {
            // CreateWallet
            0 => {
                match rest.len() {
                    33 => {
                        // Old format: authority only, assume Secp256r1
                        let mut authority = [0u8; 33];
                        authority.copy_from_slice(rest);
                        Ok(Self::CreateWallet {
                            sig_scheme: 0,
                            authority,
                        }) // 0 = SIG_SCHEME_SECP256R1
                    }
                    34 => {
                        // New format: sig_scheme + authority
                        let sig_scheme = rest[0];
                        let mut authority = [0u8; 33];
                        authority.copy_from_slice(&rest[1..34]);
                        Ok(Self::CreateWallet {
                            sig_scheme,
                            authority,
                        })
                    }
                    _ => Err(ProgramError::InvalidInstructionData),
                }
            }

            // Execute
            1 => {
                if rest.len() < 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let secp256r1_ix_index = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                let inner_instructions = parse_inner_instructions(&rest[9..])?;

                Ok(Self::Execute {
                    secp256r1_ix_index,
                    max_slot,
                    inner_instructions,
                })
            }

            // CloseWallet
            2 => {
                if rest.len() != 1 + 8 + 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let secp256r1_ix_index = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                let mut destination = [0u8; 32];
                destination.copy_from_slice(&rest[9..41]);
                Ok(Self::CloseWallet {
                    secp256r1_ix_index,
                    max_slot,
                    destination,
                })
            }

            // AdvanceNonce
            3 => {
                if rest.len() != 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let secp256r1_ix_index = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                Ok(Self::AdvanceNonce {
                    secp256r1_ix_index,
                    max_slot,
                })
            }

            // CreateSession
            4 => {
                // Minimum: secp256r1_ix_index(1) + max_slot(8) + session_authority(32)
                //        + expiry_slot(8) + max_lamports_per_call(8)
                //        + max_total_spent_lamports(8) + allowed_programs_count(1) = 66
                const BASE_LEN: usize = 1 + 8 + 32 + 8 + 8 + 8 + 1;
                if rest.len() < BASE_LEN {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let secp256r1_ix_index = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                let mut session_authority = [0u8; 32];
                session_authority.copy_from_slice(&rest[9..41]);
                let expiry_slot = u64::from_le_bytes(rest[41..49].try_into().unwrap());
                let max_lamports_per_call = u64::from_le_bytes(rest[49..57].try_into().unwrap());
                let max_total_spent_lamports =
                    u64::from_le_bytes(rest[57..65].try_into().unwrap());
                let allowed_programs_count = rest[65];

                // Reject zero (state deserialization also rejects it; keep both
                // layers in sync so a malformed session cannot be created and
                // then bricked on first SessionExecute).
                if allowed_programs_count == 0 {
                    return Err(MachineWalletError::TooManyAllowedPrograms.into());
                }
                if allowed_programs_count as usize > MAX_ALLOWED_PROGRAMS {
                    return Err(MachineWalletError::TooManyAllowedPrograms.into());
                }

                let count = allowed_programs_count as usize;
                if rest.len() != BASE_LEN + count * 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }

                let mut allowed_programs = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
                for (i, prog) in allowed_programs.iter_mut().take(count).enumerate() {
                    let start = BASE_LEN + i * 32;
                    prog.copy_from_slice(&rest[start..start + 32]);
                }

                Ok(Self::CreateSession {
                    secp256r1_ix_index,
                    max_slot,
                    session_authority,
                    expiry_slot,
                    max_lamports_per_call,
                    max_total_spent_lamports,
                    allowed_programs_count,
                    allowed_programs,
                })
            }

            // SessionExecute
            5 => {
                let inner_instructions = parse_inner_instructions(rest)?;
                Ok(Self::SessionExecute { inner_instructions })
            }

            // RevokeSession
            6 => {
                // Exact: secp256r1_ix_index(1) + max_slot(8) + session_authority(32) = 41
                if rest.len() != 1 + 8 + 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let secp256r1_ix_index = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                let mut session_authority = [0u8; 32];
                session_authority.copy_from_slice(&rest[9..41]);
                Ok(Self::RevokeSession {
                    secp256r1_ix_index,
                    max_slot,
                    session_authority,
                })
            }

            // SelfRevokeSession
            7 => {
                if !rest.is_empty() {
                    return Err(ProgramError::InvalidInstructionData);
                }
                Ok(Self::SelfRevokeSession)
            }

            // CloseSession
            8 => {
                if !rest.is_empty() {
                    return Err(ProgramError::InvalidInstructionData);
                }
                Ok(Self::CloseSession)
            }

            // AddAuthority: disc(1) + precompile_ix_index(1) + new_sig_scheme(1) + new_pubkey(33) + new_threshold(1) + max_slot(8) = 44 bytes rest
            9 => {
                if rest.len() != 1 + 1 + 33 + 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let precompile_ix_index = rest[0];
                let new_sig_scheme = rest[1];
                let mut new_pubkey = [0u8; 33];
                new_pubkey.copy_from_slice(&rest[2..35]);
                let new_threshold = rest[35];
                let max_slot = u64::from_le_bytes(rest[36..44].try_into().unwrap());
                Ok(Self::AddAuthority {
                    precompile_ix_index,
                    new_sig_scheme,
                    new_pubkey,
                    new_threshold,
                    max_slot,
                })
            }

            // RemoveAuthority: same layout as AddAuthority
            10 => {
                if rest.len() != 1 + 1 + 33 + 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let precompile_ix_index = rest[0];
                let remove_sig_scheme = rest[1];
                let mut remove_pubkey = [0u8; 33];
                remove_pubkey.copy_from_slice(&rest[2..35]);
                let new_threshold = rest[35];
                let max_slot = u64::from_le_bytes(rest[36..44].try_into().unwrap());
                Ok(Self::RemoveAuthority {
                    precompile_ix_index,
                    remove_sig_scheme,
                    remove_pubkey,
                    new_threshold,
                    max_slot,
                })
            }

            // SetThreshold: disc(1) + precompile_ix_index(1) + new_threshold(1) + max_slot(8) = 10 bytes rest
            11 => {
                if rest.len() != 1 + 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let precompile_ix_index = rest[0];
                let new_threshold = rest[1];
                let max_slot = u64::from_le_bytes(rest[2..10].try_into().unwrap());
                Ok(Self::SetThreshold {
                    precompile_ix_index,
                    new_threshold,
                    max_slot,
                })
            }

            // OwnerCloseSession: disc(1) + precompile_ix_index(1) + max_slot(8) + session_authority(32) + destination(32) = 73 bytes rest
            12 => {
                if rest.len() != 1 + 8 + 32 + 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let precompile_ix_index = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                let mut session_authority = [0u8; 32];
                session_authority.copy_from_slice(&rest[9..41]);
                let mut destination = [0u8; 32];
                destination.copy_from_slice(&rest[41..73]);
                Ok(Self::OwnerCloseSession {
                    precompile_ix_index,
                    max_slot,
                    session_authority,
                    destination,
                })
            }

            // ProvideWebAuthnEvidence (sidecar). No state change; delegates
            // structural validation to `parse_webauthn_evidence_payload`.
            14 => {
                let (auth_data, client_data_json) = parse_webauthn_evidence_payload(rest)?;
                Ok(Self::ProvideWebAuthnEvidence {
                    auth_data,
                    client_data_json,
                })
            }

            _ => Err(ProgramError::InvalidInstructionData),
        }
    }
}

/// Discriminator for `ProvideWebAuthnEvidence`. Exposed so the threshold
/// scanner can recognize sidecar instructions without depending on the
/// full enum.
pub const PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR: u8 = 14;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_create_wallet() {
        let mut data = vec![0u8]; // discriminator
        let mut authority = [0xAAu8; 33];
        authority[0] = 0x02;
        data.extend_from_slice(&authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CreateWallet {
                sig_scheme,
                authority: a,
            } => {
                assert_eq!(sig_scheme, 0); // old format defaults to Secp256r1
                assert_eq!(a[0], 0x02);
                assert_eq!(a[1], 0xAA);
                assert_eq!(a.len(), 33);
            }
            _ => panic!("Expected CreateWallet"),
        }
    }

    #[test]
    fn test_parse_execute_with_one_inner() {
        let mut data = vec![1u8]; // discriminator
        data.push(3); // secp256r1_ix_index
        data.extend_from_slice(&999u64.to_le_bytes()); // max_slot

        // inner_instruction_count = 1
        data.extend_from_slice(&1u32.to_le_bytes());

        // inner instruction: program_id (32 bytes)
        let program_id = [0xBBu8; 32];
        data.extend_from_slice(&program_id);

        // accounts_len = 2, data_len = 4
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());

        // account entries: (index, flags) pairs
        data.extend_from_slice(&[0, 0x01]); // index=0, writable
        data.extend_from_slice(&[1, 0x00]); // index=1, readonly

        // data
        data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::Execute {
                secp256r1_ix_index,
                max_slot,
                inner_instructions,
            } => {
                assert_eq!(secp256r1_ix_index, 3);
                assert_eq!(max_slot, 999);
                assert_eq!(inner_instructions.len(), 1);
                assert_eq!(inner_instructions[0].program_id, program_id);
                assert_eq!(inner_instructions[0].account_count(), 2);
                let accounts: Vec<_> = inner_instructions[0].accounts().collect();
                assert_eq!(accounts[0].index, 0);
                assert!(accounts[0].is_writable());
                assert_eq!(accounts[1].index, 1);
                assert!(!accounts[1].is_writable());
                assert_eq!(inner_instructions[0].data, [0xDE, 0xAD, 0xBE, 0xEF]);
            }
            _ => panic!("Expected Execute"),
        }
    }

    #[test]
    fn test_parse_execute_empty() {
        let mut data = vec![1u8]; // discriminator
        data.push(0); // secp256r1_ix_index
        data.extend_from_slice(&500u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&0u32.to_le_bytes()); // 0 inner instructions

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::Execute {
                secp256r1_ix_index,
                max_slot,
                inner_instructions,
            } => {
                assert_eq!(secp256r1_ix_index, 0);
                assert_eq!(max_slot, 500);
                assert!(inner_instructions.is_empty());
            }
            _ => panic!("Expected Execute"),
        }
    }

    #[test]
    fn test_parse_execute_multiple_inner() {
        let mut data = vec![1u8]; // discriminator
        data.push(5); // secp256r1_ix_index
        data.extend_from_slice(&1000u64.to_le_bytes()); // max_slot

        // 2 inner instructions
        data.extend_from_slice(&2u32.to_le_bytes());

        // First inner instruction
        data.extend_from_slice(&[0x11u8; 32]); // program_id
        data.extend_from_slice(&1u16.to_le_bytes()); // accounts_len
        data.extend_from_slice(&2u16.to_le_bytes()); // data_len
        data.extend_from_slice(&[0, 0x01]); // account entry: index=0, writable
        data.extend_from_slice(&[0xAA, 0xBB]); // data

        // Second inner instruction
        data.extend_from_slice(&[0x22u8; 32]); // program_id
        data.extend_from_slice(&0u16.to_le_bytes()); // accounts_len = 0
        data.extend_from_slice(&0u16.to_le_bytes()); // data_len = 0

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::Execute {
                inner_instructions, ..
            } => {
                assert_eq!(inner_instructions.len(), 2);
                assert_eq!(inner_instructions[0].program_id, [0x11u8; 32]);
                assert_eq!(inner_instructions[0].account_count(), 1);
                let accounts: Vec<_> = inner_instructions[0].accounts().collect();
                assert_eq!(accounts[0].index, 0);
                assert!(accounts[0].is_writable());
                assert_eq!(inner_instructions[0].data, [0xAA, 0xBB]);
                assert_eq!(inner_instructions[1].program_id, [0x22u8; 32]);
                assert_eq!(inner_instructions[1].account_count(), 0);
                assert!(inner_instructions[1].data.is_empty());
            }
            _ => panic!("Expected Execute"),
        }
    }

    #[test]
    fn test_parse_close_wallet() {
        let mut data = vec![2u8, 7]; // discriminator + secp256r1_ix_index
        data.extend_from_slice(&42u64.to_le_bytes()); // max_slot
        let dest = [0xCCu8; 32];
        data.extend_from_slice(&dest); // destination

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CloseWallet {
                secp256r1_ix_index,
                max_slot,
                destination,
            } => {
                assert_eq!(secp256r1_ix_index, 7);
                assert_eq!(max_slot, 42);
                assert_eq!(destination, dest);
            }
            _ => panic!("Expected CloseWallet"),
        }
    }

    #[test]
    fn test_parse_advance_nonce() {
        let mut data = vec![3u8, 2]; // discriminator + secp256r1_ix_index
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::AdvanceNonce {
                secp256r1_ix_index,
                max_slot,
            } => {
                assert_eq!(secp256r1_ix_index, 2);
                assert_eq!(max_slot, 100);
            }
            _ => panic!("Expected AdvanceNonce"),
        }
    }

    #[test]
    fn test_unknown_discriminator() {
        let data = vec![255u8, 0, 0];
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<u8> = vec![];
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_create_wallet() {
        // Only 10 bytes of authority instead of 33
        let mut data = vec![0u8];
        data.extend_from_slice(&[0xAAu8; 10]);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_create_wallet_rejects_trailing_bytes() {
        // 35 bytes (neither 33 nor 34) — must be rejected
        let mut data = vec![0u8];
        data.extend_from_slice(&[0xAAu8; 33]);
        data.extend_from_slice(&[0xFF, 0xFF]); // 2 extra bytes → 35 total → error
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_no_max_slot() {
        // Discriminator + ix_index, but no max_slot
        let data = vec![1u8, 0];
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_no_count() {
        // Discriminator + ix_index + max_slot, but no count
        let mut data = vec![1u8, 0];
        data.extend_from_slice(&100u64.to_le_bytes());
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_inner_missing_program_id() {
        let mut data = vec![1u8, 0]; // discriminator + ix_index
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        data.extend_from_slice(&[0xBBu8; 16]); // only 16 bytes of program_id
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_inner_missing_data() {
        let mut data = vec![1u8, 0];
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&[0xBBu8; 32]); // program_id
        data.extend_from_slice(&0u16.to_le_bytes()); // accounts_len = 0
        data.extend_from_slice(&10u16.to_le_bytes()); // data_len = 10
                                                      // but no data follows
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_execute_rejects_trailing_bytes() {
        let mut data = vec![1u8, 0];
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_close_wallet() {
        let data = vec![2u8]; // discriminator only
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_close_wallet_no_destination() {
        let mut data = vec![2u8, 0]; // discriminator + ix_index
        data.extend_from_slice(&42u64.to_le_bytes()); // max_slot but no destination
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_close_wallet_rejects_trailing_bytes() {
        let mut data = vec![2u8, 0];
        data.extend_from_slice(&42u64.to_le_bytes());
        data.extend_from_slice(&[0xCCu8; 32]);
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_advance_nonce_rejects_trailing_bytes() {
        let mut data = vec![3u8, 0];
        data.extend_from_slice(&42u64.to_le_bytes());
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_invalid_account_flags_rejected() {
        let mut data = vec![1u8, 0]; // Execute discriminator
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        data.extend_from_slice(&[0xAAu8; 32]); // program_id
        data.extend_from_slice(&1u16.to_le_bytes()); // accounts_len = 1
        data.extend_from_slice(&0u16.to_le_bytes()); // data_len = 0
        data.extend_from_slice(&[0, 0x02]); // index=0, flags=0x02 (unknown bit)
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_valid_account_flags_accepted() {
        let mut data = vec![1u8, 0];
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 32]);
        data.extend_from_slice(&2u16.to_le_bytes()); // accounts_len = 2
        data.extend_from_slice(&0u16.to_le_bytes()); // data_len = 0
        data.extend_from_slice(&[0, 0x01]); // writable
        data.extend_from_slice(&[1, 0x00]); // readonly
        let result = MachineWalletInstruction::unpack(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_too_many_inner_instructions() {
        let mut data = vec![1u8, 0];
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&(MAX_INNER_INSTRUCTIONS as u32 + 1).to_le_bytes());
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(MachineWalletError::TooManyInnerInstructions as u32)
        );
    }

    #[test]
    fn test_max_inner_instructions_accepted() {
        // count = MAX but no actual data — will fail on parsing first inner ix, not on limit
        let mut data = vec![1u8, 0];
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&(MAX_INNER_INSTRUCTIONS as u32).to_le_bytes());
        let result = MachineWalletInstruction::unpack(&data);
        // Should fail on missing program_id, not on count
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_create_session() {
        let mut data = vec![4u8]; // discriminator
        data.push(2); // secp256r1_ix_index
        data.extend_from_slice(&12345u64.to_le_bytes()); // max_slot
        let session_authority = [0xAAu8; 32];
        data.extend_from_slice(&session_authority); // session_authority
        data.extend_from_slice(&99999u64.to_le_bytes()); // expiry_slot
        data.extend_from_slice(&5_000_000u64.to_le_bytes()); // max_lamports_per_call
        data.extend_from_slice(&100_000_000u64.to_le_bytes()); // max_total_spent_lamports
        data.push(2); // allowed_programs_count
        let prog0 = [0x11u8; 32];
        let prog1 = [0x22u8; 32];
        data.extend_from_slice(&prog0);
        data.extend_from_slice(&prog1);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CreateSession {
                secp256r1_ix_index,
                max_slot,
                session_authority: sa,
                expiry_slot,
                max_lamports_per_call,
                max_total_spent_lamports,
                allowed_programs_count,
                allowed_programs,
            } => {
                assert_eq!(secp256r1_ix_index, 2);
                assert_eq!(max_slot, 12345);
                assert_eq!(sa, session_authority);
                assert_eq!(expiry_slot, 99999);
                assert_eq!(max_lamports_per_call, 5_000_000);
                assert_eq!(max_total_spent_lamports, 100_000_000);
                assert_eq!(allowed_programs_count, 2);
                assert_eq!(allowed_programs[0], prog0);
                assert_eq!(allowed_programs[1], prog1);
                assert_eq!(allowed_programs[2], [0u8; 32]);
            }
            _ => panic!("Expected CreateSession"),
        }
    }

    #[test]
    fn test_parse_create_session_zero_programs_rejected() {
        let mut data = vec![4u8]; // discriminator
        data.push(0); // secp256r1_ix_index
        data.extend_from_slice(&0u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&[0u8; 32]); // session_authority
        data.extend_from_slice(&0u64.to_le_bytes()); // expiry_slot
        data.extend_from_slice(&0u64.to_le_bytes()); // max_lamports_per_call
        data.extend_from_slice(&0u64.to_le_bytes()); // max_total_spent_lamports
        data.push(0); // allowed_programs_count = 0 — must be rejected (creates a brick session)

        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(crate::error::MachineWalletError::TooManyAllowedPrograms as u32)
        );
    }

    #[test]
    fn test_parse_create_session_too_many_programs() {
        let mut data = vec![4u8]; // discriminator
        data.push(0); // secp256r1_ix_index
        data.extend_from_slice(&0u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&[0u8; 32]); // session_authority
        data.extend_from_slice(&0u64.to_le_bytes()); // expiry_slot
        data.extend_from_slice(&0u64.to_le_bytes()); // max_lamports_per_call
        data.extend_from_slice(&0u64.to_le_bytes()); // max_total_spent_lamports
        data.push(9); // allowed_programs_count = 9 (> MAX_ALLOWED_PROGRAMS = 8)
                      // Add 9 * 32 bytes of program data
        data.extend_from_slice(&[0u8; 9 * 32]);

        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(crate::error::MachineWalletError::TooManyAllowedPrograms as u32)
        );
    }

    #[test]
    fn test_parse_session_execute() {
        let mut data = vec![5u8]; // discriminator

        // inner_instruction_count = 1
        data.extend_from_slice(&1u32.to_le_bytes());

        // inner instruction: program_id (32 bytes)
        let program_id = [0xCCu8; 32];
        data.extend_from_slice(&program_id);

        // accounts_len = 1, data_len = 2
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());

        // account entry
        data.extend_from_slice(&[0, 0x01]); // index=0, writable

        // data
        data.extend_from_slice(&[0xAB, 0xCD]);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::SessionExecute { inner_instructions } => {
                assert_eq!(inner_instructions.len(), 1);
                assert_eq!(inner_instructions[0].program_id, program_id);
                assert_eq!(inner_instructions[0].account_count(), 1);
                let accounts: Vec<_> = inner_instructions[0].accounts().collect();
                assert!(accounts[0].is_writable());
                assert_eq!(inner_instructions[0].data, [0xAB, 0xCD]);
            }
            _ => panic!("Expected SessionExecute"),
        }
    }

    #[test]
    fn test_parse_revoke_session() {
        let mut data = vec![6u8]; // discriminator
        data.push(1); // secp256r1_ix_index
        data.extend_from_slice(&777u64.to_le_bytes()); // max_slot
        let session_authority = [0xBBu8; 32];
        data.extend_from_slice(&session_authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::RevokeSession {
                secp256r1_ix_index,
                max_slot,
                session_authority: sa,
            } => {
                assert_eq!(secp256r1_ix_index, 1);
                assert_eq!(max_slot, 777);
                assert_eq!(sa, session_authority);
            }
            _ => panic!("Expected RevokeSession"),
        }
    }

    #[test]
    fn test_parse_self_revoke_session() {
        let data = vec![7u8]; // discriminator only

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::SelfRevokeSession => {}
            _ => panic!("Expected SelfRevokeSession"),
        }
    }

    #[test]
    fn test_self_revoke_rejects_trailing() {
        let data = vec![7u8, 0xFF]; // trailing byte should be rejected
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_close_session() {
        let data = vec![8u8]; // discriminator only
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CloseSession => {}
            _ => panic!("Expected CloseSession"),
        }
    }

    #[test]
    fn test_close_session_rejects_trailing() {
        let data = vec![8u8, 0xFF]; // trailing byte should be rejected
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    // ─── AddAuthority tests ───────────────────────────────────────────────────

    #[test]
    fn test_parse_add_authority() {
        let mut data = vec![9u8]; // discriminator
        data.push(0x03); // precompile_ix_index = 3
        data.push(0x01); // new_sig_scheme = 1
        let pubkey = [0xABu8; 33];
        data.extend_from_slice(&pubkey); // new_pubkey
        data.push(2); // new_threshold
        data.extend_from_slice(&12345u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::AddAuthority {
                precompile_ix_index,
                new_sig_scheme,
                new_pubkey,
                new_threshold,
                max_slot,
            } => {
                assert_eq!(precompile_ix_index, 3);
                assert_eq!(new_sig_scheme, 1);
                assert_eq!(new_pubkey, pubkey);
                assert_eq!(new_threshold, 2);
                assert_eq!(max_slot, 12345);
            }
            _ => panic!("Expected AddAuthority"),
        }
    }

    #[test]
    fn test_parse_add_authority_truncated() {
        let mut data = vec![9u8];
        data.push(0); // precompile_ix_index
        data.push(0); // new_sig_scheme
        data.extend_from_slice(&[0u8; 20]); // incomplete (need 33 + 1 + 8 more)
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_add_authority_trailing() {
        let mut data = vec![9u8];
        data.push(0); // precompile_ix_index
        data.push(0); // new_sig_scheme
        data.extend_from_slice(&[0u8; 33]); // new_pubkey
        data.push(1); // new_threshold
        data.extend_from_slice(&999u64.to_le_bytes()); // max_slot
        data.push(0xFF); // trailing byte — should be rejected
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    // ─── RemoveAuthority tests ────────────────────────────────────────────────

    #[test]
    fn test_parse_remove_authority() {
        let mut data = vec![10u8]; // discriminator
        data.push(0x02); // precompile_ix_index = 2
        data.push(0x00); // remove_sig_scheme = 0
        let pubkey = [0xCDu8; 33];
        data.extend_from_slice(&pubkey); // remove_pubkey
        data.push(0); // new_threshold = 0 (keep current)
        data.extend_from_slice(&99999u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::RemoveAuthority {
                precompile_ix_index,
                remove_sig_scheme,
                remove_pubkey,
                new_threshold,
                max_slot,
            } => {
                assert_eq!(precompile_ix_index, 2);
                assert_eq!(remove_sig_scheme, 0);
                assert_eq!(remove_pubkey, pubkey);
                assert_eq!(new_threshold, 0);
                assert_eq!(max_slot, 99999);
            }
            _ => panic!("Expected RemoveAuthority"),
        }
    }

    #[test]
    fn test_parse_remove_authority_truncated() {
        let mut data = vec![10u8];
        data.push(0); // precompile_ix_index
        data.push(0); // remove_sig_scheme
        data.extend_from_slice(&[0u8; 10]); // incomplete
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    // ─── SetThreshold tests ───────────────────────────────────────────────────

    #[test]
    fn test_parse_set_threshold() {
        let mut data = vec![11u8]; // discriminator
        data.push(0x01); // precompile_ix_index = 1
        data.push(3); // new_threshold
        data.extend_from_slice(&777u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::SetThreshold {
                precompile_ix_index,
                new_threshold,
                max_slot,
            } => {
                assert_eq!(precompile_ix_index, 1);
                assert_eq!(new_threshold, 3);
                assert_eq!(max_slot, 777);
            }
            _ => panic!("Expected SetThreshold"),
        }
    }

    #[test]
    fn test_parse_set_threshold_truncated() {
        let mut data = vec![11u8];
        data.push(0); // precompile_ix_index
        data.push(1); // new_threshold only — missing max_slot
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_set_threshold_trailing() {
        let mut data = vec![11u8];
        data.push(0); // precompile_ix_index
        data.push(2); // new_threshold
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.push(0xFF); // trailing byte — should be rejected
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_owner_close_session() {
        let mut data = vec![12u8];
        data.push(4);
        data.extend_from_slice(&777u64.to_le_bytes());
        let session_authority = [0xABu8; 32];
        let destination = [0xCDu8; 32];
        data.extend_from_slice(&session_authority);
        data.extend_from_slice(&destination);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::OwnerCloseSession {
                precompile_ix_index,
                max_slot,
                session_authority: sa,
                destination: dest,
            } => {
                assert_eq!(precompile_ix_index, 4);
                assert_eq!(max_slot, 777);
                assert_eq!(sa, session_authority);
                assert_eq!(dest, destination);
            }
            _ => panic!("Expected OwnerCloseSession"),
        }
    }

    #[test]
    fn test_parse_owner_close_session_truncated() {
        let mut data = vec![12u8];
        data.push(1);
        data.extend_from_slice(&999u64.to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 31]);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_owner_close_session_trailing() {
        let mut data = vec![12u8];
        data.push(1);
        data.extend_from_slice(&999u64.to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 32]);
        data.extend_from_slice(&[0xBBu8; 32]);
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    // ─── CreateWallet backward-compat tests ───────────────────────────────────

    #[test]
    fn test_parse_create_wallet_old_format() {
        // 33 bytes: old format, sig_scheme should default to 0 (Secp256r1)
        let mut data = vec![0u8]; // discriminator
        let authority = [0x02u8; 33];
        data.extend_from_slice(&authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CreateWallet {
                sig_scheme,
                authority: a,
            } => {
                assert_eq!(sig_scheme, 0); // SIG_SCHEME_SECP256R1
                assert_eq!(a, authority);
            }
            _ => panic!("Expected CreateWallet"),
        }
    }

    #[test]
    fn test_parse_create_wallet_new_format() {
        // 34 bytes: new format with explicit sig_scheme
        let mut data = vec![0u8]; // discriminator
        data.push(0x01); // sig_scheme = 1
        let authority = [0x03u8; 33];
        data.extend_from_slice(&authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CreateWallet {
                sig_scheme,
                authority: a,
            } => {
                assert_eq!(sig_scheme, 1);
                assert_eq!(a, authority);
            }
            _ => panic!("Expected CreateWallet"),
        }
    }

    // ─── ProvideWebAuthnEvidence sidecar parsing ─────────────────────────────

    fn build_sidecar_data(auth_data: &[u8], client_data_json: &[u8]) -> Vec<u8> {
        let mut data = vec![14u8]; // discriminator
        data.extend_from_slice(&(auth_data.len() as u16).to_le_bytes());
        data.extend_from_slice(auth_data);
        data.extend_from_slice(&(client_data_json.len() as u16).to_le_bytes());
        data.extend_from_slice(client_data_json);
        data
    }

    #[test]
    fn test_parse_provide_webauthn_evidence_valid() {
        let auth_data = [0xAAu8; MIN_AUTH_DATA_SIZE];
        let cd = br#"{"type":"webauthn.get","challenge":"abc"}"#;
        let data = build_sidecar_data(&auth_data, cd);
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::ProvideWebAuthnEvidence {
                auth_data: ad,
                client_data_json: cdj,
            } => {
                assert_eq!(ad, &auth_data[..]);
                assert_eq!(cdj, &cd[..]);
            }
            _ => panic!("Expected ProvideWebAuthnEvidence"),
        }
    }

    #[test]
    fn test_parse_sidecar_empty_payload() {
        let data = vec![14u8]; // discriminator only — no length bytes
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_sidecar_auth_data_below_min() {
        for short_len in [0usize, 1, 10, MIN_AUTH_DATA_SIZE - 1] {
            let mut data = vec![14u8];
            data.extend_from_slice(&(short_len as u16).to_le_bytes());
            data.extend_from_slice(&vec![0u8; short_len]);
            data.extend_from_slice(&3u16.to_le_bytes());
            data.extend_from_slice(b"abc");
            let err = MachineWalletInstruction::unpack(&data).unwrap_err();
            assert_eq!(
                err,
                ProgramError::InvalidInstructionData,
                "auth_data_len={} must be rejected",
                short_len
            );
        }
    }

    #[test]
    fn test_parse_sidecar_auth_data_above_max() {
        let mut data = vec![14u8];
        data.extend_from_slice(&((MAX_AUTH_DATA_SIZE as u16 + 1).to_le_bytes()));
        data.extend_from_slice(&vec![0u8; MAX_AUTH_DATA_SIZE + 1]);
        data.extend_from_slice(&3u16.to_le_bytes());
        data.extend_from_slice(b"abc");
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_sidecar_cd_empty_rejected() {
        let auth_data = [0u8; MIN_AUTH_DATA_SIZE];
        let mut data = vec![14u8];
        data.extend_from_slice(&(MIN_AUTH_DATA_SIZE as u16).to_le_bytes());
        data.extend_from_slice(&auth_data);
        data.extend_from_slice(&0u16.to_le_bytes()); // cd_len = 0
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_sidecar_cd_above_max() {
        let auth_data = [0u8; MIN_AUTH_DATA_SIZE];
        let mut data = vec![14u8];
        data.extend_from_slice(&(MIN_AUTH_DATA_SIZE as u16).to_le_bytes());
        data.extend_from_slice(&auth_data);
        data.extend_from_slice(&((MAX_CLIENT_DATA_JSON_SIZE as u16 + 1).to_le_bytes()));
        data.extend_from_slice(&vec![0u8; MAX_CLIENT_DATA_JSON_SIZE + 1]);
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_sidecar_rejects_trailing_bytes() {
        let auth_data = [0u8; MIN_AUTH_DATA_SIZE];
        let cd = b"{}";
        let mut data = build_sidecar_data(&auth_data, cd);
        data.push(0xFF); // trailing garbage
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_sidecar_auth_at_min_boundary() {
        let auth_data = [0u8; MIN_AUTH_DATA_SIZE];
        let cd = b"{}";
        let data = build_sidecar_data(&auth_data, cd);
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        matches!(ix, MachineWalletInstruction::ProvideWebAuthnEvidence { .. });
    }

    #[test]
    fn test_parse_sidecar_auth_at_max_boundary() {
        let auth_data = vec![0u8; MAX_AUTH_DATA_SIZE];
        let cd = b"{}";
        let data = build_sidecar_data(&auth_data, cd);
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        matches!(ix, MachineWalletInstruction::ProvideWebAuthnEvidence { .. });
    }

    /// The previously exposed `PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR` must
    /// match the wire discriminator we accept.
    #[test]
    fn test_sidecar_discriminator_constant() {
        assert_eq!(PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR, 14);
    }

    /// Discriminator 13 (previously `ExecuteWebAuthn`) must now be rejected —
    /// clients relying on it must migrate to `Execute` + sidecar.
    #[test]
    fn test_disc_13_is_reserved_and_rejected() {
        let data = vec![13u8];
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }
}
