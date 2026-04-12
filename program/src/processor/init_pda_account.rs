use solana_program::{
    account_info::AccountInfo,
    entrypoint::ProgramResult,
    program::{invoke, invoke_signed},
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    sysvar::Sysvar,
};
use solana_system_interface::instruction as system_instruction;

use crate::{error::MachineWalletError, state::SYSTEM_PROGRAM_ID};

/// Initialize a PDA in a dust-resistant way.
///
/// Attackers can pre-fund a PDA with lamports, making `create_account` unusable forever.
/// This helper accepts either:
/// - a fresh empty system account (lamports may be 0), or
/// - a dusted empty system account (lamports > 0, data len = 0),
/// and upgrades it into a program-owned PDA via transfer + allocate + assign.
pub(crate) fn init_pda_account<'a>(
    payer: &AccountInfo<'a>,
    pda_account: &AccountInfo<'a>,
    system_program: &AccountInfo<'a>,
    owner: &Pubkey,
    space: usize,
    signer_seeds: &[&[u8]],
    already_initialized_error: MachineWalletError,
) -> ProgramResult {
    if *system_program.key != SYSTEM_PROGRAM_ID {
        return Err(ProgramError::IncorrectProgramId);
    }

    // Program-owned means the PDA was already initialized by us — this
    // covers both live accounts AND tombstoned ones (closed wallets retain
    // program ownership + rent-exempt lamports so the runtime never
    // zero-lamport-GCs them, which is the mechanism that prevents
    // close-then-recreate replay of pre-signed Executes; see
    // `processor::close_wallet` for the full invariant).
    if *pda_account.owner == *owner {
        return Err(already_initialized_error.into());
    }

    // Only an empty system-owned account may be initialized.
    if *pda_account.owner != SYSTEM_PROGRAM_ID
        || !pda_account.data_is_empty()
        || pda_account.executable
    {
        return Err(already_initialized_error.into());
    }

    let rent = Rent::get()?;
    let required_lamports = rent.minimum_balance(space);
    let current_lamports = pda_account.lamports();
    if current_lamports < required_lamports {
        let deficit = required_lamports - current_lamports;
        invoke(
            &system_instruction::transfer(payer.key, pda_account.key, deficit),
            &[payer.clone(), pda_account.clone(), system_program.clone()],
        )?;
    }

    invoke_signed(
        &system_instruction::allocate(pda_account.key, space as u64),
        &[pda_account.clone()],
        &[signer_seeds],
    )?;
    invoke_signed(
        &system_instruction::assign(pda_account.key, owner),
        &[pda_account.clone()],
        &[signer_seeds],
    )?;

    Ok(())
}
