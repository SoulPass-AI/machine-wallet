use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    keccak,
    program::invoke_signed,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::Sysvar,
};
use solana_system_interface::instruction as system_instruction;

use crate::{
    error::MachineWalletError,
    state::{MachineWallet, SYSTEM_PROGRAM_ID},
    threshold,
};

/// Domain separator for Close messages, prevents cross-protocol and cross-instruction collision.
const CLOSE_TAG: &[u8] = b"machine_wallet_close_v0";

fn validate_destination(
    wallet_pubkey: &Pubkey,
    vault_pubkey: &Pubkey,
    destination_pubkey: &[u8; 32],
    destination_account: &Pubkey,
) -> Result<(), ProgramError> {
    if destination_account.as_ref() != destination_pubkey {
        return Err(MachineWalletError::InvalidDestination.into());
    }
    if destination_account == wallet_pubkey || destination_account == vault_pubkey {
        return Err(MachineWalletError::InvalidDestination.into());
    }
    Ok(())
}

/// Compute close message: keccak256(CLOSE_TAG || wallet || creation_slot || nonce || max_slot || destination)
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_close_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    destination: &[u8; 32],
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        CLOSE_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        destination,
    ])
    .to_bytes()
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    secp256r1_ix_index: u8,
    max_slot: u64,
    destination_pubkey: [u8; 32],
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;
    let vault_account = next_account_info(account_iter)?;
    let destination = next_account_info(account_iter)?;
    let system_program = next_account_info(account_iter)?;

    // 1. Cheap field checks first — fail fast before any syscall
    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable || !vault_account.is_writable || !destination.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // 2. Validate destination matches the authority-signed destination pubkey
    //    This prevents relay/fee_payer from redirecting funds to an arbitrary address.
    validate_destination(
        wallet_account.key,
        vault_account.key,
        &destination_pubkey,
        destination.key,
    )?;

    // 3. Validate instructions sysvar
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    // 4. Anti-reentry: stack_height == TRANSACTION_LEVEL_STACK_HEIGHT guarantees we are
    //    a top-level instruction, not reached via any CPI chain.
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 5. Signature expiry: current slot must not exceed max_slot
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 6. Load and deserialize MachineWallet state
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 7. Verify wallet PDA using cached bump (id computed from authority, ~100 CU syscall)
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 8. Verify vault PDA using cached vault_bump
    let expected_vault_pda = Pubkey::create_program_address(
        &[
            MachineWallet::VAULT_SEED_PREFIX,
            wallet_account.key.as_ref(),
            &[wallet.vault_bump],
        ],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidVaultPDA)?;
    if *vault_account.key != expected_vault_pda {
        return Err(MachineWalletError::InvalidVaultPDA.into());
    }

    // 9. Verify vault is system-owned (vault stays system-owned; SOL is reclaimed via CPI).
    if *vault_account.owner != SYSTEM_PROGRAM_ID {
        return Err(MachineWalletError::InvalidVaultOwner.into());
    }

    // 9a. Validate system program
    if *system_program.key != SYSTEM_PROGRAM_ID {
        return Err(ProgramError::IncorrectProgramId);
    }

    // 10. Compute expected close message (zero-alloc via hashv)
    let expected_message = compute_close_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        &destination_pubkey,
    );

    // 11. Signature verification (v0: single precompile, v1: threshold scan)
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        &wallet,
        secp256r1_ix_index,
        &expected_message,
    )?;

    // 14. Transfer vault SOL to destination via system_program::transfer CPI.
    //     Vault is system-owned, so we use invoke_signed with vault PDA seeds.
    let vault_lamports = vault_account.lamports();
    if vault_lamports > 0 {
        let vault_signer_seeds: &[&[u8]] = &[
            MachineWallet::VAULT_SEED_PREFIX,
            wallet_account.key.as_ref(),
            &[wallet.vault_bump],
        ];
        invoke_signed(
            &system_instruction::transfer(vault_account.key, destination.key, vault_lamports),
            &[
                vault_account.clone(),
                destination.clone(),
                system_program.clone(),
            ],
            &[vault_signer_seeds],
        )?;
    }

    // 15. Transfer wallet SOL to destination (wallet is program-owned, direct lamport manipulation).
    let wallet_lamports = wallet_account.lamports();
    **wallet_account.try_borrow_mut_lamports()? = 0;
    **destination.try_borrow_mut_lamports()? = destination
        .lamports()
        .checked_add(wallet_lamports)
        .ok_or(ProgramError::ArithmeticOverflow)?;

    // 16. Zero wallet account data.
    wallet_account.try_borrow_mut_data()?.fill(0);

    // SAFETY INVARIANT — Same-transaction close+recreate prevention:
    // After zeroing lamports and data above, the wallet account still has data_len == 87
    // (fill(0) zeroes content, not length). Solana's system_instruction::create_account
    // requires data_len == 0, so CreateWallet will fail within the same transaction.
    // The runtime GCs zero-lamport accounts at transaction end (zeroing data_len + reassigning
    // to SystemProgram), enabling recreation only in a subsequent transaction — at which point
    // creation_slot will differ, invalidating all prior signatures.
    // If Solana changes GC timing or create_account preconditions, re-audit this assumption.

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_destination_rejects_wallet_alias() {
        let wallet = Pubkey::new_unique();
        let vault = Pubkey::new_unique();
        let result = validate_destination(&wallet, &vault, &wallet.to_bytes(), &wallet);
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidDestination as u32)
        );
    }

    #[test]
    fn test_validate_destination_rejects_vault_alias() {
        let wallet = Pubkey::new_unique();
        let vault = Pubkey::new_unique();
        let result = validate_destination(&wallet, &vault, &vault.to_bytes(), &vault);
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidDestination as u32)
        );
    }

    #[test]
    fn test_validate_destination_rejects_mismatch() {
        let wallet = Pubkey::new_unique();
        let vault = Pubkey::new_unique();
        let destination = Pubkey::new_unique();
        let signed_destination = Pubkey::new_unique();
        let result = validate_destination(
            &wallet,
            &vault,
            &signed_destination.to_bytes(),
            &destination,
        );
        assert_eq!(
            result.unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidDestination as u32)
        );
    }

    #[test]
    fn test_validate_destination_accepts_external_account() {
        let wallet = Pubkey::new_unique();
        let vault = Pubkey::new_unique();
        let destination = Pubkey::new_unique();
        validate_destination(&wallet, &vault, &destination.to_bytes(), &destination).unwrap();
    }
}
