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

/// Domain separator for RevokeSession messages, prevents cross-protocol collision.
const REVOKE_SESSION_TAG: &[u8] = b"machine_wallet_revoke_session_v0";

/// Compute revoke-session message: keccak256(REVOKE_SESSION_TAG || wallet_pda || creation_slot_le || nonce_le || max_slot_le || session_authority)
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_revoke_session_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    session_authority: &[u8; 32],
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        REVOKE_SESSION_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        session_authority,
    ])
    .to_bytes()
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    secp256r1_ix_index: u8,
    max_slot: u64,
    session_authority: [u8; 32],
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;
    let session_account = next_account_info(account_iter)?;

    // 1. Basic validation: cheap field checks first — fail fast before any syscall
    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }
    if !session_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if session_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // Validate instructions sysvar
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    // 2. Anti-reentry: stack_height == TRANSACTION_LEVEL_STACK_HEIGHT guarantees we are
    //    a top-level instruction, not reached via any CPI chain.
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 3. Signature expiry: current slot must not exceed max_slot
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 4. Load and deserialize MachineWallet state
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 5. Verify wallet PDA using cached bump (cheaper than find_program_address)
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 6. Deserialize SessionState — validates it is a real initialized session
    let session_data_borrowed = session_account.try_borrow_data()?;
    let session = SessionState::deserialize_runtime(&session_data_borrowed)?;
    drop(session_data_borrowed);

    if session.wallet != wallet_account.key.to_bytes() {
        return Err(MachineWalletError::SessionWalletMismatch.into());
    }
    if session.authority != session_authority {
        return Err(MachineWalletError::SessionAuthorityMismatch.into());
    }
    if session.wallet_creation_slot != wallet.creation_slot {
        return Err(MachineWalletError::SessionWalletMismatch.into());
    }

    // 6b. Idempotent: already-revoked session needs no signature or nonce bump
    if session.revoked {
        return Ok(());
    }

    // 7. Verify session PDA using stored bump (cheaper than find_program_address)
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

    // 8. Compute expected message
    let expected_message = compute_revoke_session_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        &session_authority,
    );

    // 9. Signature verification.
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        program_id,
        &wallet,
        secp256r1_ix_index,
        &expected_message,
    )?;

    // 10. Increment nonce (CEI pattern — state change before side effects)
    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;
    {
        let mut wallet_data = wallet_account.try_borrow_mut_data()?;
        let nonce_off = wallet.nonce_offset();
        wallet_data[nonce_off..nonce_off + 8].copy_from_slice(&new_nonce.to_le_bytes());
    }

    // 11. Set revoked flag
    let mut session_data = session_account.try_borrow_mut_data()?;
    session_data[SessionState::REVOKED_OFFSET] = 1;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_revoke_session_message_deterministic() {
        let wallet = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let msg1 = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);
        let msg2 = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_compute_revoke_session_message_includes_nonce() {
        let wallet = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let msg1 = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);
        let msg2 = compute_revoke_session_message(&wallet, 100, 1, 500, &session_auth);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_revoke_session_message_includes_wallet() {
        let wallet1 = Pubkey::new_unique();
        let wallet2 = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let msg1 = compute_revoke_session_message(&wallet1, 100, 0, 500, &session_auth);
        let msg2 = compute_revoke_session_message(&wallet2, 100, 0, 500, &session_auth);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_revoke_session_message_includes_creation_slot() {
        let wallet = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let msg1 = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);
        let msg2 = compute_revoke_session_message(&wallet, 200, 0, 500, &session_auth);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_revoke_session_message_includes_max_slot() {
        let wallet = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let msg1 = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);
        let msg2 = compute_revoke_session_message(&wallet, 100, 0, 600, &session_auth);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_revoke_session_message_includes_session_authority() {
        let wallet = Pubkey::new_unique();
        let auth1 = [0xAAu8; 32];
        let auth2 = [0xBBu8; 32];
        let msg1 = compute_revoke_session_message(&wallet, 100, 0, 500, &auth1);
        let msg2 = compute_revoke_session_message(&wallet, 100, 0, 500, &auth2);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_domain_tag_prevents_cross_message_collision() {
        // RevokeSession and AdvanceNonce with same parameters should differ due to domain tags
        let wallet = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let revoke_msg = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);

        let advance_msg =
            crate::processor::advance_nonce::compute_advance_nonce_message(&wallet, 100, 0, 500);

        assert_ne!(revoke_msg, advance_msg);
    }

    #[test]
    fn test_domain_tag_prevents_cross_revoke_create_collision() {
        // RevokeSession and CreateSession messages with same nonce/slot params must differ
        let wallet = Pubkey::new_unique();
        let session_auth = [0xBBu8; 32];
        let revoke_msg = compute_revoke_session_message(&wallet, 100, 0, 500, &session_auth);

        // CreateSession includes session_data_hash at the end, not session_authority directly
        // Still verify the domain tags produce distinct hashes when content overlaps
        let create_msg = crate::processor::create_session::compute_create_session_message(
            &wallet,
            100,
            0,
            500,
            &session_auth, // reusing session_auth as session_data_hash for collision test
        );

        assert_ne!(revoke_msg, create_msg);
    }
}
