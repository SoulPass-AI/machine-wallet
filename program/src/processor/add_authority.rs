use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    keccak,
    program::invoke,
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
    sysvar::Sysvar,
};
use solana_system_interface::instruction as system_instruction;

use crate::{
    error::MachineWalletError,
    state::{
        AuthoritySlot, MachineWallet, AUTHORITY_SLOT_SIZE, SIG_SCHEME_ED25519,
        SIG_SCHEME_SECP256R1, SYSTEM_PROGRAM_ID, V1_HEADER_SIZE,
    },
    threshold,
};

/// Domain separator for AddAuthority messages.
const ADD_AUTHORITY_TAG: &[u8] = b"machine_wallet_add_authority_v0";

/// Compute the signed message hash for AddAuthority.
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_add_authority_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    new_sig_scheme: u8,
    new_pubkey: &[u8; 33],
    new_threshold: u8,
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        ADD_AUTHORITY_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        &[new_sig_scheme],
        new_pubkey,
        &[new_threshold],
    ])
    .to_bytes()
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    precompile_ix_index: u8,
    new_sig_scheme: u8,
    new_pubkey: [u8; 33],
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
    // 6a. Authority count limit
    if wallet.authority_count >= crate::state::MAX_AUTHORITIES {
        return Err(MachineWalletError::AuthorityLimitExceeded.into());
    }

    // 6b. Validate the new authority slot
    let new_slot = AuthoritySlot {
        sig_scheme: new_sig_scheme,
        pubkey: new_pubkey,
    };
    if !new_slot.is_valid() {
        return match new_sig_scheme {
            SIG_SCHEME_ED25519 => Err(MachineWalletError::InvalidEd25519Pubkey.into()),
            SIG_SCHEME_SECP256R1 => Err(ProgramError::InvalidAccountData),
            _ => Err(ProgramError::InvalidAccountData),
        };
    }

    // 6c. No duplicate: (sig_scheme, pubkey) must not already exist
    let count = wallet.authority_count as usize;
    for i in 0..count {
        if wallet.authorities[i].sig_scheme == new_sig_scheme
            && wallet.authorities[i].pubkey == new_pubkey
        {
            return Err(MachineWalletError::DuplicateAuthority.into());
        }
    }

    // 6d. Threshold validation
    let new_count = wallet
        .authority_count
        .checked_add(1)
        .ok_or(MachineWalletError::AuthorityLimitExceeded)?;
    if new_threshold != 0 && new_threshold > new_count {
        return Err(MachineWalletError::InvalidThreshold.into());
    }

    // 7. Compute message hash (domain separated)
    let expected_message = compute_add_authority_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        new_sig_scheme,
        &new_pubkey,
        new_threshold,
    );

    // 8. Signature verification (v0: single precompile at index, v1: threshold scan)
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        &wallet,
        precompile_ix_index,
        &expected_message,
    )?;

    // 9. State mutation
    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;

    if wallet.version == 0 {
        // ── v0 → v1 migration ──
        // All v0 fields are already in `wallet` from deserialize_runtime above.
        // Defense-in-depth: v0 must have exactly 1 authority
        if wallet.authority_count != 1 {
            return Err(ProgramError::InvalidAccountData);
        }

        // Realloc to v1 size for 2 authorities
        let new_size = MachineWallet::v1_account_size(2);
        realloc_with_rent(wallet_account, fee_payer, system_program, new_size)?;

        // Write v1 layout
        let mut data = wallet_account.try_borrow_mut_data()?;
        data[0] = 1; // version = 1
                     // [1..34] unchanged (bump, wallet_id)

        let effective_threshold = if new_threshold != 0 {
            new_threshold
        } else {
            wallet.threshold
        };
        data[34] = effective_threshold;
        data[35] = 2; // authority_count = 2

        // Write nonce (incremented)
        data[MachineWallet::V1_NONCE_OFFSET..MachineWallet::V1_NONCE_OFFSET + 8]
            .copy_from_slice(&new_nonce.to_le_bytes());
        // Write creation_slot (preserved from v0)
        data[MachineWallet::V1_CREATION_SLOT_OFFSET..MachineWallet::V1_CREATION_SLOT_OFFSET + 8]
            .copy_from_slice(&wallet.creation_slot.to_le_bytes());
        // Write vault_bump
        data[MachineWallet::V1_VAULT_BUMP_OFFSET] = wallet.vault_bump;

        // Authority slot 0: original authority
        let slot0_offset = V1_HEADER_SIZE;
        data[slot0_offset] = wallet.authorities[0].sig_scheme;
        data[slot0_offset + 1..slot0_offset + 1 + 33]
            .copy_from_slice(&wallet.authorities[0].pubkey);

        // Authority slot 1: new authority
        let slot1_offset = V1_HEADER_SIZE + AUTHORITY_SLOT_SIZE;
        data[slot1_offset] = new_sig_scheme;
        data[slot1_offset + 1..slot1_offset + 1 + 33].copy_from_slice(&new_pubkey);
    } else {
        // ── v1 append ──
        let new_size = MachineWallet::v1_account_size(new_count);
        realloc_with_rent(wallet_account, fee_payer, system_program, new_size)?;

        let mut data = wallet_account.try_borrow_mut_data()?;

        // Write new authority slot at end
        let new_slot_offset = V1_HEADER_SIZE + count * AUTHORITY_SLOT_SIZE;
        data[new_slot_offset] = new_sig_scheme;
        data[new_slot_offset + 1..new_slot_offset + 1 + 33].copy_from_slice(&new_pubkey);

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

    Ok(())
}

/// Realloc a program-owned account and transfer rent deficit from fee_payer.
fn realloc_with_rent<'a>(
    account: &AccountInfo<'a>,
    fee_payer: &AccountInfo<'a>,
    system_program: &AccountInfo<'a>,
    new_size: usize,
) -> ProgramResult {
    let rent = Rent::get()?;
    let new_min_balance = rent.minimum_balance(new_size);
    let current_balance = account.lamports();
    if current_balance < new_min_balance {
        let deficit = new_min_balance - current_balance;
        invoke(
            &system_instruction::transfer(fee_payer.key, account.key, deficit),
            &[fee_payer.clone(), account.clone(), system_program.clone()],
        )?;
    }
    account.resize(new_size)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_authority_message_deterministic() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg1 = compute_add_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 2);
        let msg2 = compute_add_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 2);
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_add_authority_message_domain_separation() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg_add = compute_add_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 2);
        // Different nonce should produce different hash
        let msg_diff = compute_add_authority_message(&wallet, 100, 6, 200, 0, &pubkey, 2);
        assert_ne!(msg_add, msg_diff);
    }

    #[test]
    fn test_add_authority_message_threshold_sensitivity() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg_t0 = compute_add_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 0);
        let msg_t2 = compute_add_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 2);
        assert_ne!(msg_t0, msg_t2);
    }

    #[test]
    fn test_add_authority_message_scheme_sensitivity() {
        let wallet = Pubkey::new_unique();
        let mut pubkey = [0x42u8; 33];
        pubkey[0] = 0x02;

        let msg_p256 = compute_add_authority_message(&wallet, 100, 5, 200, 0, &pubkey, 2);
        let msg_ed = compute_add_authority_message(&wallet, 100, 5, 200, 1, &pubkey, 2);
        assert_ne!(msg_p256, msg_ed);
    }
}
