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

/// Domain separator for SetThreshold messages.
const SET_THRESHOLD_TAG: &[u8] = b"machine_wallet_set_threshold_v0";

/// Compute the signed message hash for SetThreshold.
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_set_threshold_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    new_threshold: u8,
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        SET_THRESHOLD_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        &[new_threshold],
    ])
    .to_bytes()
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    new_threshold: u8,
    max_slot: u64,
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;

    // 1. Anti-reentry: must be top-level instruction
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 2. Account validation — fail fast before any syscall
    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }
    super::require_instructions_sysvar(instructions_sysvar)?;

    // 3. Signature expiry
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 4. Load wallet state.
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 5. Verify wallet PDA.
    super::verify_wallet_pda(wallet_account, &wallet, program_id)?;

    // 6. Validate new_threshold: must be >= 1 and <= authority_count
    if new_threshold < 1 || new_threshold > wallet.authority_count {
        return Err(MachineWalletError::InvalidThreshold.into());
    }

    // 7. Compute message hash (domain separated)
    let expected_message = compute_set_threshold_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        new_threshold,
    );

    // 8. Signature verification.
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        program_id,
        &wallet,
        &expected_message,
    )?;

    // 9. Update threshold byte and bump nonce.
    let mut data = wallet_account.try_borrow_mut_data()?;
    data[MachineWallet::THRESHOLD_OFFSET] = new_threshold;
    wallet.write_incremented_nonce(&mut data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_threshold_message_deterministic() {
        let wallet = Pubkey::new_unique();

        let msg1 = compute_set_threshold_message(&wallet, 100, 5, 200, 2);
        let msg2 = compute_set_threshold_message(&wallet, 100, 5, 200, 2);
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_set_threshold_message_domain_separation() {
        let wallet = Pubkey::new_unique();

        let msg_st = compute_set_threshold_message(&wallet, 100, 5, 200, 2);
        // Different nonce should produce different hash
        let msg_diff = compute_set_threshold_message(&wallet, 100, 6, 200, 2);
        assert_ne!(msg_st, msg_diff);
    }

    #[test]
    fn test_set_threshold_message_threshold_sensitivity() {
        let wallet = Pubkey::new_unique();

        let msg_t1 = compute_set_threshold_message(&wallet, 100, 5, 200, 1);
        let msg_t2 = compute_set_threshold_message(&wallet, 100, 5, 200, 2);
        assert_ne!(msg_t1, msg_t2);
    }

    #[test]
    fn test_set_threshold_message_wallet_sensitivity() {
        let wallet1 = Pubkey::new_unique();
        let wallet2 = Pubkey::new_unique();

        let msg1 = compute_set_threshold_message(&wallet1, 100, 5, 200, 2);
        let msg2 = compute_set_threshold_message(&wallet2, 100, 5, 200, 2);
        assert_ne!(msg1, msg2);
    }
}
