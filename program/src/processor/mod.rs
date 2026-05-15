pub mod add_authority;
pub mod advance_nonce;
pub mod close_session;
pub mod close_wallet;
pub mod create_session;
pub mod create_wallet;
pub mod execute;
pub(crate) mod init_pda_account;
pub mod owner_close_session;
pub mod remove_authority;
pub mod revoke_session;
pub mod self_revoke_session;
pub mod session_execute;
pub mod set_threshold;

use solana_program::{
    account_info::AccountInfo,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    program_error::ProgramError,
    pubkey::Pubkey,
};

use crate::error::MachineWalletError;
use crate::instruction::MachineWalletInstruction;
use crate::state::MachineWallet;

/// Reject CPI invocation. The stateless `ProvideWebAuthnEvidenceCompact`
/// sidecar must run at top level so the precompile scanner observes it via
/// the instructions sysvar; a CPI-invoked sidecar would not be visible to
/// the consuming wallet instruction.
#[inline(always)]
pub(crate) fn reject_cpi_reentry() -> ProgramResult {
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }
    Ok(())
}

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

/// Re-derive the wallet PDA from the cached `(id, bump)` and verify it
/// matches `wallet_account.key`. Centralizes the check so every signed
/// operation refuses to trust a forged wallet account.
#[inline(always)]
pub(crate) fn verify_wallet_pda(
    wallet_account: &AccountInfo,
    wallet: &MachineWallet,
    program_id: &Pubkey,
) -> ProgramResult {
    let id = wallet.id();
    let expected = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected {
        return Err(MachineWalletError::InvalidWalletPDA.into());
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
            max_slot,
            inner_instructions,
        } => execute::process(program_id, accounts, max_slot, inner_instructions),
        MachineWalletInstruction::CloseWallet {
            max_slot,
            destination,
        } => close_wallet::process(program_id, accounts, max_slot, destination),
        MachineWalletInstruction::AdvanceNonce { max_slot } => {
            advance_nonce::process(program_id, accounts, max_slot)
        }
        MachineWalletInstruction::CreateSession {
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
            max_slot,
            session_authority,
        } => revoke_session::process(program_id, accounts, max_slot, session_authority),
        MachineWalletInstruction::SelfRevokeSession => {
            self_revoke_session::process(program_id, accounts)
        }
        MachineWalletInstruction::CloseSession => close_session::process(program_id, accounts),
        MachineWalletInstruction::AddAuthority {
            new_sig_scheme,
            new_pubkey,
            new_threshold,
            max_slot,
        } => add_authority::process(
            program_id,
            accounts,
            new_sig_scheme,
            new_pubkey,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::RemoveAuthority {
            remove_sig_scheme,
            remove_pubkey,
            new_threshold,
            max_slot,
        } => remove_authority::process(
            program_id,
            accounts,
            remove_sig_scheme,
            remove_pubkey,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::SetThreshold {
            new_threshold,
            max_slot,
        } => set_threshold::process(program_id, accounts, new_threshold, max_slot),
        MachineWalletInstruction::OwnerCloseSession {
            max_slot,
            session_authority,
            destination,
        } => owner_close_session::process(
            program_id,
            accounts,
            max_slot,
            session_authority,
            destination,
        ),
        MachineWalletInstruction::ProvideWebAuthnEvidenceCompact { .. } => reject_cpi_reentry(),
    }
}
