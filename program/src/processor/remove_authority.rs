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

use crate::{
    error::MachineWalletError,
    state::{MachineWallet, AUTHORITY_SLOT_SIZE, SYSTEM_PROGRAM_ID, V1_HEADER_SIZE},
    threshold,
};

/// Domain separator for RemoveAuthority messages.
const REMOVE_AUTHORITY_TAG: &[u8] = b"machine_wallet_remove_authority_v0";

/// Compute the signed message hash for RemoveAuthority.
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_remove_authority_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    remove_sig_scheme: u8,
    remove_pubkey: &[u8; 33],
    new_threshold: u8,
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        REMOVE_AUTHORITY_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        &[remove_sig_scheme],
        remove_pubkey,
        &[new_threshold],
    ])
    .to_bytes()
}

#[allow(clippy::too_many_arguments)]
pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    precompile_ix_index: u8,
    remove_sig_scheme: u8,
    remove_pubkey: [u8; 33],
    new_threshold: u8,
    max_slot: u64,
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;
    let system_program = next_account_info(account_iter)?;

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
    if *system_program.key != SYSTEM_PROGRAM_ID {
        return Err(ProgramError::IncorrectProgramId);
    }
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    // 3. Signature expiry
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 4. Load wallet state (works for both v0 and v1)
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 5. Verify wallet PDA
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 6. Constraints

    // 6a. Cannot remove last authority — would permanently lock the wallet
    // NOTE: v0 wallets naturally hit this constraint (authority_count == 1)
    if wallet.authority_count <= 1 {
        return Err(MachineWalletError::CannotRemoveLastAuthority.into());
    }

    let count = wallet.authority_count as usize;

    // 6b. Find the authority to remove by (sig_scheme, pubkey) pair
    let remove_index = (0..count)
        .find(|&i| {
            wallet.authorities[i].sig_scheme == remove_sig_scheme
                && wallet.authorities[i].pubkey == remove_pubkey
        })
        .ok_or(MachineWalletError::AuthorityNotFound)?;

    // 6c. Threshold validation after removal (new count = count - 1)
    let new_count = wallet.authority_count - 1; // >= 1, guaranteed by 6a
    if new_threshold == 0 {
        // Keep current threshold — must still be valid after removal
        if wallet.threshold > new_count {
            return Err(MachineWalletError::InvalidThreshold.into());
        }
    } else {
        // Explicit threshold update — must be in valid range [1, new_count]
        if new_threshold < 1 || new_threshold > new_count {
            return Err(MachineWalletError::InvalidThreshold.into());
        }
    }

    // 7. Compute message hash (domain separated)
    let expected_message = compute_remove_authority_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        remove_sig_scheme,
        &remove_pubkey,
        new_threshold,
    );

    // 8. Signature verification. v0 wallets cannot reach here (blocked by 6a),
    //    but we use the unified path for consistency and future-proofing.
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        program_id,
        &wallet,
        precompile_ix_index,
        &expected_message,
    )?;

    // 9. State mutation
    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;

    // Wallet must be v1 (v0 blocked by CannotRemoveLastAuthority above)
    // Swap-remove: copy last slot over the removed slot, then realloc shrink
    let new_size = MachineWallet::v1_account_size(new_count);

    {
        let mut data = wallet_account.try_borrow_mut_data()?;

        // Swap-remove: overwrite removed slot with last slot (if not already last)
        let last_index = count - 1;
        if remove_index != last_index {
            let removed_offset = V1_HEADER_SIZE + remove_index * AUTHORITY_SLOT_SIZE;
            let last_offset = V1_HEADER_SIZE + last_index * AUTHORITY_SLOT_SIZE;
            // Copy last slot bytes into removed slot position
            // We need to do this without overlapping borrows — use a temp buffer
            let mut last_slot_bytes = [0u8; AUTHORITY_SLOT_SIZE];
            last_slot_bytes.copy_from_slice(&data[last_offset..last_offset + AUTHORITY_SLOT_SIZE]);
            data[removed_offset..removed_offset + AUTHORITY_SLOT_SIZE]
                .copy_from_slice(&last_slot_bytes);
        }
        // (if remove_index == last_index, the slot is already at the tail; realloc will drop it)

        // Update authority_count
        data[35] = new_count;

        // Update threshold if requested
        if new_threshold != 0 {
            data[34] = new_threshold;
        }

        // Increment nonce (v1 offset)
        data[MachineWallet::V1_NONCE_OFFSET..MachineWallet::V1_NONCE_OFFSET + 8]
            .copy_from_slice(&new_nonce.to_le_bytes());
    }

    // Resize account to shrink (drops the last authority slot)
    wallet_account.resize(new_size)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_authority_message_deterministic() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg1 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 1);
        let msg2 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 1);
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_remove_authority_message_domain_separation() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg_remove = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 1);
        // Different nonce should produce different hash
        let msg_diff = compute_remove_authority_message(&wallet, 100, 6, 200, 0, &pubkey, 1);
        assert_ne!(msg_remove, msg_diff);
    }

    #[test]
    fn test_remove_authority_message_scheme_sensitivity() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg_p256 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 1);
        let msg_ed = compute_remove_authority_message(&wallet, 100, 5, 200, 1, &pubkey, 1);
        assert_ne!(msg_p256, msg_ed);
    }

    #[test]
    fn test_remove_authority_message_threshold_sensitivity() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg_t0 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 0);
        let msg_t1 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 1);
        assert_ne!(msg_t0, msg_t1);
    }

    #[test]
    fn test_remove_authority_message_pubkey_sensitivity() {
        let wallet = Pubkey::new_unique();
        let mut pubkey1 = [0x42u8; 33];
        pubkey1[0] = 0x02;
        let mut pubkey2 = [0x43u8; 33];
        pubkey2[0] = 0x03;

        let msg1 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey1, 1);
        let msg2 = compute_remove_authority_message(&wallet, 100, 5, 200, 0, &pubkey2, 1);
        assert_ne!(msg1, msg2);
    }
}
