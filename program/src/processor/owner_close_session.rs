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
    state::{MachineWallet, SessionState, SESSION_SEED_PREFIX},
    threshold,
};

const OWNER_CLOSE_SESSION_TAG: &[u8] = b"machine_wallet_owner_close_session_v0";

fn validate_destination(
    session_account: &Pubkey,
    destination_pubkey: &[u8; 32],
    destination_account: &AccountInfo,
) -> ProgramResult {
    if !destination_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if destination_account.key.as_ref() != destination_pubkey {
        return Err(MachineWalletError::InvalidDestination.into());
    }
    if destination_account.key == session_account {
        return Err(MachineWalletError::InvalidDestination.into());
    }
    Ok(())
}

pub fn compute_owner_close_session_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    session_authority: &[u8; 32],
    destination: &[u8; 32],
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        OWNER_CLOSE_SESSION_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        session_authority,
        destination,
    ])
    .to_bytes()
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    precompile_ix_index: u8,
    max_slot: u64,
    session_authority: [u8; 32],
    destination_pubkey: [u8; 32],
) -> ProgramResult {
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    let account_iter = &mut accounts.iter();
    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;
    let session_account = next_account_info(account_iter)?;
    let destination = next_account_info(account_iter)?;

    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable || !session_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id || session_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    validate_destination(session_account.key, &destination_pubkey, destination)?;

    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    let wallet_data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&wallet_data)?;
    drop(wallet_data);

    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    let session_data = session_account.try_borrow_data()?;
    let session = SessionState::deserialize_runtime(&session_data)?;
    drop(session_data);

    // Validate session PDA before trusting any session content.
    let expected_session_pda = Pubkey::create_program_address(
        &[
            SESSION_SEED_PREFIX,
            wallet_account.key.as_ref(),
            &session_authority,
            &[session.bump],
        ],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidSessionPDA)?;
    if *session_account.key != expected_session_pda {
        return Err(MachineWalletError::InvalidSessionPDA.into());
    }

    if session.wallet != wallet_account.key.to_bytes()
        || session.wallet_creation_slot != wallet.creation_slot
    {
        return Err(MachineWalletError::SessionWalletMismatch.into());
    }
    if session.authority != session_authority {
        return Err(MachineWalletError::SessionAuthorityMismatch.into());
    }

    // Only revoked or expired sessions can be closed by the owner.
    if !session.revoked && clock.slot <= session.expiry_slot {
        return Err(MachineWalletError::SessionStillActive.into());
    }

    let expected_message = compute_owner_close_session_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        &session_authority,
        &destination_pubkey,
    );
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        &wallet,
        precompile_ix_index,
        &expected_message,
    )?;

    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;
    {
        let mut data = wallet_account.try_borrow_mut_data()?;
        let nonce_off = wallet.nonce_offset();
        data[nonce_off..nonce_off + 8].copy_from_slice(&new_nonce.to_le_bytes());
    }

    session_account.try_borrow_mut_data()?.fill(0);
    let session_lamports = session_account.lamports();
    **session_account.try_borrow_mut_lamports()? = 0;
    **destination.try_borrow_mut_lamports()? = destination
        .lamports()
        .checked_add(session_lamports)
        .ok_or(ProgramError::ArithmeticOverflow)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owner_close_session_message_deterministic() {
        let wallet = Pubkey::new_unique();
        let session_authority = [0xAAu8; 32];
        let destination = [0xBBu8; 32];
        let msg1 = compute_owner_close_session_message(
            &wallet,
            10,
            20,
            30,
            &session_authority,
            &destination,
        );
        let msg2 = compute_owner_close_session_message(
            &wallet,
            10,
            20,
            30,
            &session_authority,
            &destination,
        );
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_owner_close_session_message_includes_destination() {
        let wallet = Pubkey::new_unique();
        let session_authority = [0xAAu8; 32];
        let dest_a = [0xBBu8; 32];
        let dest_b = [0xCCu8; 32];
        let msg1 =
            compute_owner_close_session_message(&wallet, 10, 20, 30, &session_authority, &dest_a);
        let msg2 =
            compute_owner_close_session_message(&wallet, 10, 20, 30, &session_authority, &dest_b);
        assert_ne!(msg1, msg2);
    }
}
