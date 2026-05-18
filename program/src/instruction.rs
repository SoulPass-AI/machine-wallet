use solana_program::program_error::ProgramError;

use crate::error::MachineWalletError;
use crate::state::{MAX_ALLOWED_PROGRAMS, MAX_EPHEMERAL_SIGNERS};
use crate::webauthn::MAX_CLIENT_DATA_JSON_SIZE;

/// Maximum number of inner instructions in a single Execute call.
/// Prevents OOM from malicious count values and keeps compute budget reasonable.
pub const MAX_INNER_INSTRUCTIONS: usize = 64;

/// A single account entry for CPI: index into remaining_accounts + permission flags.
/// The authority signs these flags (they participate in the inner-ix hash that
/// feeds the threshold signature), preventing relay manipulation of is_writable
/// or is_signer between sign-time and execute-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccountEntry {
    pub index: u8,
    /// Bit 0: is_writable
    /// Bit 1: is_ephemeral_signer — account is one of the per-Execute ephemeral
    ///        signer PDAs. Signer privilege is provided by `invoke_signed` using
    ///        seeds derived from `wallet.nonce`, not by an outer-tx signature.
    pub flags: u8,
}

impl AccountEntry {
    pub const FLAG_WRITABLE: u8 = 0x01;
    pub const FLAG_EPHEMERAL_SIGNER: u8 = 0x02;
    /// Mask of all currently-defined flag bits — reserved bits MUST be zero so
    /// future flag additions can extend the schema without breaking the
    /// "signed flags are stable" invariant.
    pub const FLAGS_VALID_MASK: u8 = Self::FLAG_WRITABLE | Self::FLAG_EPHEMERAL_SIGNER;

    pub fn is_writable(&self) -> bool {
        self.flags & Self::FLAG_WRITABLE != 0
    }

    pub fn is_ephemeral_signer(&self) -> bool {
        self.flags & Self::FLAG_EPHEMERAL_SIGNER != 0
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
///   *plus* one `ProvideWebAuthnEvidence` instruction carrying `clientDataJSON`.
///   The on-chain scanner parses challenge/type, hashes the JSON, then verifies
///   auth_data directly from the precompile message head.
///
/// This symmetry means an M-of-N wallet can combine arbitrarily many signers
/// of arbitrary schemes in one transaction — each signer contributes an
/// independent set of ixs, and the threshold scanner in `threshold.rs` counts
/// all valid contributions.
///
/// Wire format:
/// - CreateWallet:    [0] + max_slot (u64 LE) + sig_scheme (u8) + authority (33 bytes)
/// - Execute:         [1] + max_slot (u64 LE) + inner_instruction_count (u32 LE) + inner instructions
/// - CloseWallet:     [2] + max_slot (u64 LE) + destination (32 bytes)
/// - AdvanceNonce:    [3] + max_slot (u64 LE)
/// - CreateSession:   [4] + max_slot (u64 LE) + session_authority (32) + expiry_slot (u64 LE) + max_lamports_per_call (u64 LE) + max_total_spent_lamports (u64 LE) + allowed_programs_count (u8) + allowed_programs (count*32)
/// - SessionExecute:  [5] + inner_instruction_count (u32 LE) + inner instructions
/// - RevokeSession:   [6] + max_slot (u64 LE) + session_authority (32)
/// - SelfRevokeSession: [7]
/// - CloseSession:    [8]  — accounts: [session_account, authority (signer), destination]
/// - AddAuthority:    [9] + new_sig_scheme (u8) + new_pubkey (33) + new_threshold (u8) + max_slot (u64 LE)
/// - RemoveAuthority: [10] + remove_sig_scheme (u8) + remove_pubkey (33) + new_threshold (u8) + max_slot (u64 LE)
/// - SetThreshold:    [11] + new_threshold (u8) + max_slot (u64 LE)
/// - OwnerCloseSession: [12] + max_slot (u64 LE) + session_authority (32) + destination (32)
/// - ProvideWebAuthnEvidence: [15] + client_data_json_len (u16 LE) + client_data_json
///   Sidecar instruction — no state change. Its presence in the tx makes the
///   clientDataJSON available to every wallet instruction in the same tx for
///   WEBAUTHN-scheme threshold contribution. auth_data is read directly from
///   the paired secp256r1 precompile message by the threshold scanner.
/// - ExecuteWithEphemeralSigners: [16] + max_slot (u64 LE)
///                                + num_ephemeral (u8) + ephemeral_signer_bumps (num_ephemeral bytes)
///                                + inner_instruction_count (u32 LE) + inner instructions
///
/// Inner instruction wire format:
///   program_id (32) + accounts_len (u16 LE) + data_len (u16 LE) + accounts (accounts_len * 2 bytes: index + flags) + data
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // Box would heap-allocate on SBF — unacceptable
pub enum MachineWalletInstruction<'a> {
    /// Initialize a new wallet with a scheme-specific authority proof.
    /// sig_scheme=0 means Secp256r1.
    CreateWallet {
        max_slot: u64,
        sig_scheme: u8,
        authority: [u8; 33],
    },

    /// Execute one or more inner instructions, verified by wallet signatures.
    Execute {
        max_slot: u64,
        inner_instructions: Vec<InnerInstructionRef<'a>>,
    },

    /// Close the wallet and reclaim rent, verified by wallet signatures.
    /// destination is included in the signed message to prevent relay redirection.
    CloseWallet {
        max_slot: u64,
        destination: [u8; 32],
    },

    /// Advance nonce without executing anything (cancel a pending signed operation).
    AdvanceNonce { max_slot: u64 },

    /// Create a session key, verified by wallet signatures.
    CreateSession {
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

    /// Revoke a session (owner path, verified by wallet signatures).
    RevokeSession {
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
        new_sig_scheme: u8,
        new_pubkey: [u8; 33],
        new_threshold: u8, // 0 = keep current
        max_slot: u64,
    },

    /// Remove an authority from the wallet. Reallocs account smaller.
    RemoveAuthority {
        remove_sig_scheme: u8,
        remove_pubkey: [u8; 33],
        new_threshold: u8, // 0 = keep current
        max_slot: u64,
    },

    /// Change the threshold without adding/removing authorities.
    SetThreshold { new_threshold: u8, max_slot: u64 },

    /// Wallet-owner path to close a revoked or expired session and reclaim rent.
    OwnerCloseSession {
        max_slot: u64,
        session_authority: [u8; 32],
        destination: [u8; 32],
    },

    /// Sidecar instruction carrying WebAuthn clientDataJSON.
    ///
    /// This instruction does NOT mutate any state. Its only purpose is to live in
    /// the tx's instruction list so that `threshold::verify_threshold_signatures`
    /// can see it via the instructions sysvar. One sidecar contributes at most
    /// one WEBAUTHN-scheme authority toward a wallet instruction's threshold.
    ///
    /// `auth_data` is not carried here — the scanner reads it from the paired
    /// secp256r1 precompile message and verifies rpIdHash + UP/UV directly.
    /// The JSON is re-hashed on-chain and compared to the precompile tail, so
    /// the parsed challenge is bound to the exact bytes signed by WebAuthn.
    ProvideWebAuthnEvidence { client_data_json: &'a [u8] },

    /// Execute (disc=16) — same as `Execute` but with N per-call ephemeral
    /// signer PDAs whose signer privilege is supplied by `invoke_signed` (no
    /// outer-tx signature needed). Models Squads v4's ephemeral signers:
    /// a dApp tells the wallet "I need K ephemeral signers", the wallet
    /// derives K PDAs scoped to `(wallet, current_nonce, slot_index)`, and
    /// inner instructions referencing those PDAs in slots flagged
    /// `FLAG_EPHEMERAL_SIGNER` get the signer flag at CPI time.
    ///
    /// Unblocks integrations with programs that require external keypair
    /// signers on inner instructions (Switchboard `Randomness.create`,
    /// `SystemProgram.createAccount`, token-metadata `MasterEdition`, …)
    /// — none of which would otherwise be reachable from a passkey wallet.
    ExecuteWithEphemeralSigners {
        max_slot: u64,
        /// Number of valid entries in `ephemeral_signer_bumps`. `1..=MAX_EPHEMERAL_SIGNERS`.
        num_ephemeral_signers: u8,
        /// Caller-supplied bumps for `Pubkey::create_program_address`. Fixed-size
        /// array (not Vec) to keep the parser on `&[u8]` view with no heap.
        ephemeral_signer_bumps: [u8; MAX_EPHEMERAL_SIGNERS],
        inner_instructions: Vec<InnerInstructionRef<'a>>,
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
            if chunk[1] & !AccountEntry::FLAGS_VALID_MASK != 0 {
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
                if rest.len() != 8 + 1 + 33 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let sig_scheme = rest[8];
                let mut authority = [0u8; 33];
                authority.copy_from_slice(&rest[9..42]);
                Ok(Self::CreateWallet {
                    max_slot,
                    sig_scheme,
                    authority,
                })
            }

            // Execute
            1 => {
                if rest.len() < 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let inner_instructions = parse_inner_instructions(&rest[8..])?;

                Ok(Self::Execute {
                    max_slot,
                    inner_instructions,
                })
            }

            // CloseWallet
            2 => {
                if rest.len() != 8 + 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let mut destination = [0u8; 32];
                destination.copy_from_slice(&rest[8..40]);
                Ok(Self::CloseWallet {
                    max_slot,
                    destination,
                })
            }

            // AdvanceNonce
            3 => {
                if rest.len() != 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                Ok(Self::AdvanceNonce { max_slot })
            }

            // CreateSession
            4 => {
                // Minimum: max_slot(8) + session_authority(32) + expiry_slot(8)
                //        + max_lamports_per_call(8) + max_total_spent_lamports(8)
                //        + allowed_programs_count(1) = 65
                const BASE_LEN: usize = 8 + 32 + 8 + 8 + 8 + 1;
                if rest.len() < BASE_LEN {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let mut session_authority = [0u8; 32];
                session_authority.copy_from_slice(&rest[8..40]);
                let expiry_slot = u64::from_le_bytes(rest[40..48].try_into().unwrap());
                let max_lamports_per_call = u64::from_le_bytes(rest[48..56].try_into().unwrap());
                let max_total_spent_lamports = u64::from_le_bytes(rest[56..64].try_into().unwrap());
                let allowed_programs_count = rest[64];

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
                // Exact: max_slot(8) + session_authority(32) = 40
                if rest.len() != 8 + 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let mut session_authority = [0u8; 32];
                session_authority.copy_from_slice(&rest[8..40]);
                Ok(Self::RevokeSession {
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

            // AddAuthority: disc(1) + new_sig_scheme(1) + new_pubkey(33) + new_threshold(1) + max_slot(8) = 43 bytes rest
            9 => {
                if rest.len() != 1 + 33 + 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let new_sig_scheme = rest[0];
                let mut new_pubkey = [0u8; 33];
                new_pubkey.copy_from_slice(&rest[1..34]);
                let new_threshold = rest[34];
                let max_slot = u64::from_le_bytes(rest[35..43].try_into().unwrap());
                Ok(Self::AddAuthority {
                    new_sig_scheme,
                    new_pubkey,
                    new_threshold,
                    max_slot,
                })
            }

            // RemoveAuthority: same layout as AddAuthority
            10 => {
                if rest.len() != 1 + 33 + 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let remove_sig_scheme = rest[0];
                let mut remove_pubkey = [0u8; 33];
                remove_pubkey.copy_from_slice(&rest[1..34]);
                let new_threshold = rest[34];
                let max_slot = u64::from_le_bytes(rest[35..43].try_into().unwrap());
                Ok(Self::RemoveAuthority {
                    remove_sig_scheme,
                    remove_pubkey,
                    new_threshold,
                    max_slot,
                })
            }

            // SetThreshold: disc(1) + new_threshold(1) + max_slot(8) = 9 bytes rest
            11 => {
                if rest.len() != 1 + 8 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let new_threshold = rest[0];
                let max_slot = u64::from_le_bytes(rest[1..9].try_into().unwrap());
                Ok(Self::SetThreshold {
                    new_threshold,
                    max_slot,
                })
            }

            // OwnerCloseSession: disc(1) + max_slot(8) + session_authority(32) + destination(32) = 72 bytes rest
            12 => {
                if rest.len() != 8 + 32 + 32 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let mut session_authority = [0u8; 32];
                session_authority.copy_from_slice(&rest[8..40]);
                let mut destination = [0u8; 32];
                destination.copy_from_slice(&rest[40..72]);
                Ok(Self::OwnerCloseSession {
                    max_slot,
                    session_authority,
                    destination,
                })
            }

            15 => {
                let client_data_json = parse_webauthn_evidence_payload(rest)?;
                Ok(Self::ProvideWebAuthnEvidence { client_data_json })
            }

            // num_ephemeral=0 is rejected so the disc=1/disc=16 split stays
            // semantic ("v1 always carries at least one signer") instead of
            // letting disc=16 silently behave like disc=1.
            16 => {
                if rest.len() < 8 + 1 {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let max_slot = u64::from_le_bytes(rest[0..8].try_into().unwrap());
                let num_ephemeral = rest[8] as usize;
                if num_ephemeral == 0 || num_ephemeral > MAX_EPHEMERAL_SIGNERS {
                    return Err(MachineWalletError::TooManyEphemeralSigners.into());
                }
                let bumps_end = 9usize
                    .checked_add(num_ephemeral)
                    .ok_or(ProgramError::InvalidInstructionData)?;
                if bumps_end > rest.len() {
                    return Err(ProgramError::InvalidInstructionData);
                }
                let mut ephemeral_signer_bumps = [0u8; MAX_EPHEMERAL_SIGNERS];
                ephemeral_signer_bumps[..num_ephemeral].copy_from_slice(&rest[9..bumps_end]);
                let inner_instructions = parse_inner_instructions(&rest[bumps_end..])?;
                Ok(Self::ExecuteWithEphemeralSigners {
                    max_slot,
                    num_ephemeral_signers: num_ephemeral as u8,
                    ephemeral_signer_bumps,
                    inner_instructions,
                })
            }

            _ => Err(ProgramError::InvalidInstructionData),
        }
    }
}

/// Discriminator for `ProvideWebAuthnEvidence`. Exposed so the threshold
/// scanner can recognize sidecar instructions without depending on the full enum.
pub const PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR: u8 = 15;

/// Parse a ProvideWebAuthnEvidence payload (disc=15).
///
/// Wire format:
///   client_data_json_len : u16 LE
///   client_data_json     : client_data_json_len bytes
pub(crate) fn parse_webauthn_evidence_payload(data: &[u8]) -> Result<&[u8], ProgramError> {
    if data.len() < 2 {
        return Err(ProgramError::InvalidInstructionData);
    }
    let len = u16::from_le_bytes([data[0], data[1]]) as usize;
    if len == 0 || len > MAX_CLIENT_DATA_JSON_SIZE {
        return Err(ProgramError::InvalidInstructionData);
    }
    let end = 2usize
        .checked_add(len)
        .ok_or(ProgramError::InvalidInstructionData)?;
    if end != data.len() {
        return Err(ProgramError::InvalidInstructionData);
    }
    Ok(&data[2..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_create_wallet() {
        let mut data = vec![0u8]; // discriminator
        data.extend_from_slice(&999u64.to_le_bytes());
        data.push(0);
        let mut authority = [0xAAu8; 33];
        authority[0] = 0x02;
        data.extend_from_slice(&authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CreateWallet {
                max_slot,
                sig_scheme,
                authority: a,
            } => {
                assert_eq!(max_slot, 999);
                assert_eq!(sig_scheme, 0);
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
                max_slot,
                inner_instructions,
            } => {
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
        data.extend_from_slice(&500u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&0u32.to_le_bytes()); // 0 inner instructions

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::Execute {
                max_slot,
                inner_instructions,
            } => {
                assert_eq!(max_slot, 500);
                assert!(inner_instructions.is_empty());
            }
            _ => panic!("Expected Execute"),
        }
    }

    #[test]
    fn test_parse_execute_multiple_inner() {
        let mut data = vec![1u8]; // discriminator
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
        let mut data = vec![2u8]; // discriminator
        data.extend_from_slice(&42u64.to_le_bytes()); // max_slot
        let dest = [0xCCu8; 32];
        data.extend_from_slice(&dest); // destination

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CloseWallet {
                max_slot,
                destination,
            } => {
                assert_eq!(max_slot, 42);
                assert_eq!(destination, dest);
            }
            _ => panic!("Expected CloseWallet"),
        }
    }

    #[test]
    fn test_parse_advance_nonce() {
        let mut data = vec![3u8]; // discriminator
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::AdvanceNonce { max_slot } => {
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
        // CreateWallet must be exactly max_slot(8) + sig_scheme(1) + authority(33).
        let mut data = vec![0u8];
        data.extend_from_slice(&999u64.to_le_bytes());
        data.push(0);
        data.extend_from_slice(&[0xAAu8; 33]);
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_no_max_slot() {
        // Discriminator only, no max_slot.
        let data = vec![1u8];
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_no_count() {
        // Discriminator + max_slot, but no inner-instruction count.
        let mut data = vec![1u8];
        data.extend_from_slice(&100u64.to_le_bytes());
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_inner_missing_program_id() {
        let mut data = vec![1u8];
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        data.extend_from_slice(&[0xBBu8; 16]); // only 16 bytes of program_id
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_truncated_execute_inner_missing_data() {
        let mut data = vec![1u8];
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
        let mut data = vec![1u8];
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
        let mut data = vec![2u8];
        data.extend_from_slice(&42u64.to_le_bytes()); // max_slot but no destination
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_close_wallet_rejects_trailing_bytes() {
        let mut data = vec![2u8];
        data.extend_from_slice(&42u64.to_le_bytes());
        data.extend_from_slice(&[0xCCu8; 32]);
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_advance_nonce_rejects_trailing_bytes() {
        let mut data = vec![3u8];
        data.extend_from_slice(&42u64.to_le_bytes());
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_invalid_account_flags_rejected() {
        let mut data = vec![1u8]; // Execute discriminator
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        data.extend_from_slice(&[0xAAu8; 32]); // program_id
        data.extend_from_slice(&1u16.to_le_bytes()); // accounts_len = 1
        data.extend_from_slice(&0u16.to_le_bytes()); // data_len = 0
        // Pick the lowest reserved bit (above FLAGS_VALID_MASK) so this test
        // remains a "any future flag added breaks me" canary — the failing
        // assertion forces an explicit decision about the new bit's semantics.
        let reserved_bit = !AccountEntry::FLAGS_VALID_MASK & 0x04;
        data.extend_from_slice(&[0, reserved_bit]);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_ephemeral_signer_flag_accepted() {
        // FLAG_EPHEMERAL_SIGNER alone is a legal AccountEntry.flags value —
        // the corresponding processor enforces "pubkey must match a derived
        // PDA", not the parser. Parser only refuses *unknown* bits.
        let mut data = vec![1u8];
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 32]);
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&[0, AccountEntry::FLAG_EPHEMERAL_SIGNER]);
        let result = MachineWalletInstruction::unpack(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_valid_account_flags_accepted() {
        let mut data = vec![1u8];
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
        let mut data = vec![1u8];
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
        let mut data = vec![1u8];
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&(MAX_INNER_INSTRUCTIONS as u32).to_le_bytes());
        let result = MachineWalletInstruction::unpack(&data);
        // Should fail on missing program_id, not on count
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_create_session() {
        let mut data = vec![4u8]; // discriminator
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
                max_slot,
                session_authority: sa,
                expiry_slot,
                max_lamports_per_call,
                max_total_spent_lamports,
                allowed_programs_count,
                allowed_programs,
            } => {
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
        data.extend_from_slice(&777u64.to_le_bytes()); // max_slot
        let session_authority = [0xBBu8; 32];
        data.extend_from_slice(&session_authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::RevokeSession {
                max_slot,
                session_authority: sa,
            } => {
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
        data.push(0x01); // new_sig_scheme = 1
        let pubkey = [0xABu8; 33];
        data.extend_from_slice(&pubkey); // new_pubkey
        data.push(2); // new_threshold
        data.extend_from_slice(&12345u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::AddAuthority {
                new_sig_scheme,
                new_pubkey,
                new_threshold,
                max_slot,
            } => {
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
        data.push(0); // new_sig_scheme
        data.extend_from_slice(&[0u8; 20]); // incomplete (need 33 + 1 + 8 more)
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_add_authority_trailing() {
        let mut data = vec![9u8];
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
        data.push(0x00); // remove_sig_scheme = 0
        let pubkey = [0xCDu8; 33];
        data.extend_from_slice(&pubkey); // remove_pubkey
        data.push(0); // new_threshold = 0 (keep current)
        data.extend_from_slice(&99999u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::RemoveAuthority {
                remove_sig_scheme,
                remove_pubkey,
                new_threshold,
                max_slot,
            } => {
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
        data.push(0); // remove_sig_scheme
        data.extend_from_slice(&[0u8; 10]); // incomplete
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    // ─── SetThreshold tests ───────────────────────────────────────────────────

    #[test]
    fn test_parse_set_threshold() {
        let mut data = vec![11u8]; // discriminator
        data.push(3); // new_threshold
        data.extend_from_slice(&777u64.to_le_bytes()); // max_slot

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::SetThreshold {
                new_threshold,
                max_slot,
            } => {
                assert_eq!(new_threshold, 3);
                assert_eq!(max_slot, 777);
            }
            _ => panic!("Expected SetThreshold"),
        }
    }

    #[test]
    fn test_parse_set_threshold_truncated() {
        let mut data = vec![11u8];
        data.push(1); // new_threshold only — missing max_slot
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_set_threshold_trailing() {
        let mut data = vec![11u8];
        data.push(2); // new_threshold
        data.extend_from_slice(&100u64.to_le_bytes()); // max_slot
        data.push(0xFF); // trailing byte — should be rejected
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_owner_close_session() {
        let mut data = vec![12u8];
        data.extend_from_slice(&777u64.to_le_bytes());
        let session_authority = [0xABu8; 32];
        let destination = [0xCDu8; 32];
        data.extend_from_slice(&session_authority);
        data.extend_from_slice(&destination);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::OwnerCloseSession {
                max_slot,
                session_authority: sa,
                destination: dest,
            } => {
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
        data.extend_from_slice(&999u64.to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 31]);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_owner_close_session_trailing() {
        let mut data = vec![12u8];
        data.extend_from_slice(&999u64.to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 32]);
        data.extend_from_slice(&[0xBBu8; 32]);
        data.push(0xFF);
        let result = MachineWalletInstruction::unpack(&data);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_parse_create_wallet_new_format() {
        let mut data = vec![0u8]; // discriminator
        data.extend_from_slice(&123u64.to_le_bytes());
        data.push(0x01); // sig_scheme = 1
        let authority = [0x03u8; 33];
        data.extend_from_slice(&authority);

        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::CreateWallet {
                max_slot,
                sig_scheme,
                authority: a,
            } => {
                assert_eq!(max_slot, 123);
                assert_eq!(sig_scheme, 1);
                assert_eq!(a, authority);
            }
            _ => panic!("Expected CreateWallet"),
        }
    }

    #[test]
    fn test_parse_create_wallet_old_format_rejected() {
        let mut data = vec![0u8]; // discriminator
        data.extend_from_slice(&[0x02u8; 33]);
        assert_eq!(
            MachineWalletInstruction::unpack(&data).unwrap_err(),
            ProgramError::InvalidInstructionData
        );
    }

    /// Discriminator 13 is unknown and must be rejected. Disc=14 was the
    /// legacy full `authData + clientDataJSON` slot — removed in favor of
    /// disc=15, which carries only clientDataJSON.
    #[test]
    fn test_disc_13_and_14_rejected() {
        for disc in [13u8, 14u8] {
            let data = vec![disc];
            let err = MachineWalletInstruction::unpack(&data).unwrap_err();
            assert_eq!(err, ProgramError::InvalidInstructionData);
        }
    }

    // ─── ProvideWebAuthnEvidence (disc=15) parsing ───────────────────────────

    fn build_sidecar_data(client_data_json: &[u8]) -> Vec<u8> {
        let mut data = vec![PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR];
        data.extend_from_slice(&(client_data_json.len() as u16).to_le_bytes());
        data.extend_from_slice(client_data_json);
        data
    }

    #[test]
    fn test_parse_webauthn_sidecar_valid() {
        let client_data_json = br#"{"type":"webauthn.get","challenge":"abc"}"#;
        let data = build_sidecar_data(client_data_json);
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::ProvideWebAuthnEvidence {
                client_data_json: c,
            } => {
                assert_eq!(c, client_data_json);
            }
            _ => panic!("Expected ProvideWebAuthnEvidence"),
        }
    }

    #[test]
    fn test_webauthn_sidecar_discriminator_constant() {
        assert_eq!(PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR, 15);
    }

    #[test]
    fn test_webauthn_sidecar_empty_payload_rejected() {
        let data = vec![15u8]; // discriminator only — no payload
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_webauthn_sidecar_declared_len_missing_bytes_rejected() {
        let mut data = vec![PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR];
        data.extend_from_slice(&5u16.to_le_bytes());
        data.extend_from_slice(b"abc");
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_webauthn_sidecar_trailing_byte_rejected() {
        let mut data = build_sidecar_data(b"{}");
        data.push(0xFF);
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_webauthn_sidecar_zero_len_rejected() {
        let mut data = vec![PROVIDE_WEBAUTHN_EVIDENCE_DISCRIMINATOR];
        data.extend_from_slice(&0u16.to_le_bytes());
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }

    #[test]
    fn test_webauthn_sidecar_max_len_accepted() {
        let client_data_json = vec![b'a'; MAX_CLIENT_DATA_JSON_SIZE];
        let data = build_sidecar_data(&client_data_json);
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        assert!(matches!(
            ix,
            MachineWalletInstruction::ProvideWebAuthnEvidence { .. }
        ));
    }

    // ─── ExecuteWithEphemeralSigners (disc=16) parser ──────────────────────

    fn build_exec_v1(num_eph: u8, bumps: &[u8]) -> Vec<u8> {
        let mut data = vec![16u8];
        data.extend_from_slice(&500u64.to_le_bytes()); // max_slot
        data.push(num_eph);
        data.extend_from_slice(bumps);
        data.extend_from_slice(&1u32.to_le_bytes()); // inner_count = 1
        data.extend_from_slice(&[0xCCu8; 32]); // program_id
        data.extend_from_slice(&0u16.to_le_bytes()); // accounts_len = 0
        data.extend_from_slice(&0u16.to_le_bytes()); // data_len = 0
        data
    }

    #[test]
    fn test_parse_execute_v1_happy() {
        let data = build_exec_v1(2, &[255, 254]);
        let ix = MachineWalletInstruction::unpack(&data).unwrap();
        match ix {
            MachineWalletInstruction::ExecuteWithEphemeralSigners {
                max_slot,
                num_ephemeral_signers,
                ephemeral_signer_bumps,
                inner_instructions,
            } => {
                assert_eq!(max_slot, 500);
                assert_eq!(num_ephemeral_signers, 2);
                assert_eq!(&ephemeral_signer_bumps[..2], &[255, 254]);
                // Tail of the array stays zero — only first `num` are meaningful.
                assert!(ephemeral_signer_bumps[2..].iter().all(|b| *b == 0));
                assert_eq!(inner_instructions.len(), 1);
            }
            _ => panic!("expected ExecuteWithEphemeralSigners"),
        }
    }

    #[test]
    fn test_parse_execute_v1_zero_signers_rejected() {
        // disc=16 with num_ephemeral=0 is illegal — callers must use disc=1
        // instead. Catching this in the parser keeps the disc=1/disc=16 split
        // semantic ("v1 always carries at least one signer") instead of letting
        // disc=16 silently behave like disc=1.
        let data = build_exec_v1(0, &[]);
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(
            err,
            MachineWalletError::TooManyEphemeralSigners.into(),
        );
    }

    #[test]
    fn test_parse_execute_v1_over_cap_rejected() {
        let bumps = vec![1u8; MAX_EPHEMERAL_SIGNERS + 1];
        let data = build_exec_v1((MAX_EPHEMERAL_SIGNERS + 1) as u8, &bumps);
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(
            err,
            MachineWalletError::TooManyEphemeralSigners.into(),
        );
    }

    #[test]
    fn test_parse_execute_v1_truncated_bumps() {
        // Declare 3 ephemeral signers but only supply 2 bumps → must reject
        // before reaching parse_inner_instructions (which would otherwise read
        // partial bump bytes as inner_count and produce confusing errors).
        let mut data = vec![16u8];
        data.extend_from_slice(&500u64.to_le_bytes());
        data.push(3);
        data.extend_from_slice(&[1, 2]); // only 2 bumps
        // No inner_count follows — payload too short.
        let err = MachineWalletInstruction::unpack(&data).unwrap_err();
        assert_eq!(err, ProgramError::InvalidInstructionData);
    }
}
