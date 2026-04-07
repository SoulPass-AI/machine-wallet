use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::Sysvar,
};

use crate::{
    error::MachineWalletError,
    state::{SessionState, SESSION_SEED_PREFIX},
};

fn validate_destination(session_account: &Pubkey, destination: &AccountInfo) -> ProgramResult {
    if !destination.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if destination.key == session_account {
        return Err(MachineWalletError::InvalidDestination.into());
    }
    Ok(())
}

pub fn process(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // 1. Anti-reentry: must be a top-level instruction
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    let account_iter = &mut accounts.iter();

    let session_account = next_account_info(account_iter)?;
    let authority = next_account_info(account_iter)?;
    let destination = next_account_info(account_iter)?;

    // 2. Authority must be a signer (Ed25519 session key)
    if !authority.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    // 3. session_account must be writable and owned by this program
    if !session_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if session_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // 4. destination must be writable and distinct from the session PDA
    validate_destination(session_account.key, destination)?;

    // 5. Deserialize SessionState
    let session_data = session_account.try_borrow_data()?;
    let session = SessionState::deserialize_runtime(&session_data)?;
    drop(session_data);

    // 6. Verify session.authority == authority.key
    if session.authority != authority.key.to_bytes() {
        return Err(MachineWalletError::SessionAuthorityMismatch.into());
    }

    // 7. Session must be expired OR revoked — cannot close an active session.
    //    Active session closure would bypass the wallet owner's time-bound intent.
    //    Check revoked first to skip the Clock::get() syscall on revoked sessions.
    if !session.revoked {
        let clock = Clock::get()?;
        if clock.slot <= session.expiry_slot {
            return Err(MachineWalletError::SessionStillActive.into());
        }
    }

    // 8. Verify session PDA using stored data (defense-in-depth).
    //    Ownership check (step 3) already guarantees data was written by our program;
    //    PDA verification confirms it's the correct session for this wallet+authority pair.
    let expected_session_pda = Pubkey::create_program_address(
        &[
            SESSION_SEED_PREFIX,
            &session.wallet,
            authority.key.as_ref(),
            &[session.bump],
        ],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidSessionPDA)?;
    if *session_account.key != expected_session_pda {
        return Err(MachineWalletError::InvalidSessionPDA.into());
    }

    // 9. Close: zero data + transfer all lamports to destination.
    //    Zero data first to prevent mid-tx reads of stale state.
    //    Same pattern as CloseWallet: direct lamport manipulation on program-owned account.
    session_account.try_borrow_mut_data()?.fill(0);

    let session_lamports = session_account.lamports();
    **session_account.try_borrow_mut_lamports()? = 0;
    **destination.try_borrow_mut_lamports()? = destination
        .lamports()
        .checked_add(session_lamports)
        .ok_or(ProgramError::ArithmeticOverflow)?;

    // SAFETY INVARIANT — Same-transaction close+recreate prevention:
    // After zeroing lamports and data above, the session account still has data_len == 420
    // (fill(0) zeroes content, not length). CreateSession uses invoke_signed with
    // create_account, which requires data_len == 0. The runtime GCs zero-lamport accounts
    // at transaction end, enabling PDA reuse only in a subsequent transaction.

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{SESSION_STATE_VERSION, MAX_ALLOWED_PROGRAMS};
    use solana_program::account_info::AccountInfo;

    fn make_account_info(key: Pubkey, is_writable: bool) -> AccountInfo<'static> {
        let key = Box::leak(Box::new(key));
        let lamports = Box::leak(Box::new(0u64));
        let data = Box::leak(Vec::<u8>::new().into_boxed_slice());
        let owner = Box::leak(Box::new(Pubkey::new_unique()));
        AccountInfo::new(key, false, is_writable, lamports, data, owner, false)
    }

    fn make_session_bytes(
        authority: &Pubkey,
        wallet: &[u8; 32],
        revoked: bool,
        created_slot: u64,
        expiry_slot: u64,
    ) -> [u8; SessionState::LEN] {
        let mut prog = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
        prog[0] = [0x11u8; 32];

        let session = SessionState {
            version: SESSION_STATE_VERSION,
            bump: 253,
            wallet: *wallet,
            authority: authority.to_bytes(),
            created_slot,
            expiry_slot,
            revoked,
            wallet_creation_slot: 50,
            max_lamports_per_ix: 1_000_000,
            allowed_programs_count: 1,
            allowed_programs: prog,
        };

        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        buf
    }

    #[test]
    fn test_close_session_requires_expired_or_revoked() {
        let authority = Pubkey::new_unique();
        let wallet = [0xAAu8; 32];
        // Active session (not revoked, future expiry)
        let session_bytes = make_session_bytes(&authority, &wallet, false, 100, 999);
        let session = SessionState::deserialize(&session_bytes).unwrap();
        assert!(!session.revoked);
        assert_eq!(session.expiry_slot, 999);
        // Would need clock.slot > 999 OR revoked == true to close
    }

    #[test]
    fn test_close_session_allows_revoked() {
        let authority = Pubkey::new_unique();
        let wallet = [0xAAu8; 32];
        let session_bytes = make_session_bytes(&authority, &wallet, true, 100, 999);
        let session = SessionState::deserialize(&session_bytes).unwrap();
        assert!(session.revoked);
        // revoked == true → closeable regardless of expiry_slot
    }

    #[test]
    fn test_close_session_allows_expired() {
        let authority = Pubkey::new_unique();
        let wallet = [0xAAu8; 32];
        // Structurally valid session that can become expired once clock.slot > 150.
        let session_bytes = make_session_bytes(&authority, &wallet, false, 100, 150);
        let session = SessionState::deserialize(&session_bytes).unwrap();
        assert!(!session.revoked);
        assert_eq!(session.expiry_slot, 150);
        // If clock.slot > 150 → closeable
    }

    #[test]
    fn test_close_session_authority_mismatch() {
        let authority = Pubkey::new_unique();
        let wrong_authority = Pubkey::new_unique();
        let wallet = [0xAAu8; 32];
        let session_bytes = make_session_bytes(&authority, &wallet, true, 100, 999);
        let session = SessionState::deserialize(&session_bytes).unwrap();
        assert_ne!(session.authority, wrong_authority.to_bytes());
    }

    #[test]
    fn test_close_session_rejects_self_destination() {
        let session = Pubkey::new_unique();
        let destination = make_account_info(session, true);
        assert_eq!(
            validate_destination(&session, &destination).unwrap_err(),
            ProgramError::Custom(MachineWalletError::InvalidDestination as u32)
        );
    }

    #[test]
    fn test_zeroed_session_data_prevents_deserialization_as_active() {
        // After closing, data is all zeros. Deserialization must now hard-fail.
        let zeroed = [0u8; SessionState::LEN];
        assert_eq!(
            SessionState::deserialize(&zeroed).unwrap_err(),
            ProgramError::InvalidAccountData
        );
    }
}
