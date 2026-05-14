pub mod add_authority;
pub mod advance_nonce;
pub mod close_session;
pub mod close_wallet;
pub mod create_session;
pub mod create_wallet;
pub mod execute;
pub(crate) mod init_pda_account;
pub mod owner_close_session;
pub mod provide_webauthn_evidence;
pub mod remove_authority;
pub mod revoke_session;
pub mod self_revoke_session;
pub mod session_execute;
pub mod set_threshold;

use solana_program::{
    account_info::AccountInfo, entrypoint::ProgramResult, program_error::ProgramError,
    pubkey::Pubkey,
};

use crate::instruction::MachineWalletInstruction;

/// Reject any account that isn't the instructions sysvar.
///
/// Every state-mutating processor must read the sysvar to scan for the
/// secp256r1/ed25519 precompile evidence — a wrong account here would silently
/// disable signature checks, so a typo at any callsite must fail closed.
#[inline(always)]
pub(crate) fn require_instructions_sysvar(account: &AccountInfo) -> ProgramResult {
    if *account.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }
    Ok(())
}

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = MachineWalletInstruction::unpack(instruction_data)?;

    match instruction {
        MachineWalletInstruction::CreateWallet {
            max_slot,
            sig_scheme,
            authority,
        } => create_wallet::process(program_id, accounts, max_slot, sig_scheme, authority),
        MachineWalletInstruction::Execute {
            secp256r1_ix_index,
            max_slot,
            inner_instructions,
        } => execute::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            inner_instructions,
        ),
        MachineWalletInstruction::CloseWallet {
            secp256r1_ix_index,
            max_slot,
            destination,
        } => close_wallet::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            destination,
        ),
        MachineWalletInstruction::AdvanceNonce {
            secp256r1_ix_index,
            max_slot,
        } => advance_nonce::process(program_id, accounts, secp256r1_ix_index, max_slot),
        MachineWalletInstruction::CreateSession {
            secp256r1_ix_index,
            max_slot,
            session_authority,
            expiry_slot,
            max_lamports_per_call,
            max_total_spent_lamports,
            allowed_programs_count,
            allowed_programs,
        } => create_session::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            session_authority,
            expiry_slot,
            max_lamports_per_call,
            max_total_spent_lamports,
            allowed_programs_count,
            allowed_programs,
        ),
        MachineWalletInstruction::SessionExecute { inner_instructions } => {
            session_execute::process(program_id, accounts, inner_instructions)
        }
        MachineWalletInstruction::RevokeSession {
            secp256r1_ix_index,
            max_slot,
            session_authority,
        } => revoke_session::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            session_authority,
        ),
        MachineWalletInstruction::SelfRevokeSession => {
            self_revoke_session::process(program_id, accounts)
        }
        MachineWalletInstruction::CloseSession => close_session::process(program_id, accounts),
        MachineWalletInstruction::AddAuthority {
            precompile_ix_index,
            new_sig_scheme,
            new_pubkey,
            new_threshold,
            max_slot,
        } => add_authority::process(
            program_id,
            accounts,
            precompile_ix_index,
            new_sig_scheme,
            new_pubkey,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::RemoveAuthority {
            precompile_ix_index,
            remove_sig_scheme,
            remove_pubkey,
            new_threshold,
            max_slot,
        } => remove_authority::process(
            program_id,
            accounts,
            precompile_ix_index,
            remove_sig_scheme,
            remove_pubkey,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::SetThreshold {
            precompile_ix_index,
            new_threshold,
            max_slot,
        } => set_threshold::process(
            program_id,
            accounts,
            precompile_ix_index,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::OwnerCloseSession {
            precompile_ix_index,
            max_slot,
            session_authority,
            destination,
        } => owner_close_session::process(
            program_id,
            accounts,
            precompile_ix_index,
            max_slot,
            session_authority,
            destination,
        ),
        MachineWalletInstruction::ProvideWebAuthnEvidence {
            auth_data,
            client_data_json,
        } => provide_webauthn_evidence::process(program_id, accounts, auth_data, client_data_json),
    }
}
