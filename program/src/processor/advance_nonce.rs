use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    keccak,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::Sysvar,
};

use crate::{error::MachineWalletError, state::MachineWallet, threshold};

/// Domain separator for AdvanceNonce messages.
const ADVANCE_NONCE_TAG: &[u8] = b"machine_wallet_advance_nonce_v0";

/// Compute advance-nonce message: keccak256(ADVANCE_NONCE_TAG || wallet || creation_slot || nonce || max_slot)
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_advance_nonce_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        ADVANCE_NONCE_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
    ])
    .to_bytes()
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    secp256r1_ix_index: u8,
    max_slot: u64,
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;

    // 1. Cheap field checks first — fail fast before any syscall
    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // 2. Validate instructions sysvar
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    // 3. Anti-reentry: stack_height == TRANSACTION_LEVEL_STACK_HEIGHT guarantees we are
    //    a top-level instruction, not reached via any CPI chain.
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 4. Signature expiry
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 5. Load and deserialize MachineWallet state
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 6. Verify wallet PDA using cached bump (id computed from authority, ~100 CU syscall)
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 7. Compute expected message (zero-alloc via hashv)
    let expected_message = compute_advance_nonce_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
    );

    // 8. Signature verification.
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        program_id,
        &wallet,
        secp256r1_ix_index,
        &expected_message,
    )?;

    // 9. Increment nonce (invalidates any pending signed operations with the old nonce)
    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;

    let mut data = wallet_account.try_borrow_mut_data()?;
    let nonce_off = wallet.nonce_offset();
    data[nonce_off..nonce_off + 8].copy_from_slice(&new_nonce.to_le_bytes());

    Ok(())
}
