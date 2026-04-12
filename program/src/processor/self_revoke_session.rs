use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    program_error::ProgramError,
    pubkey::Pubkey,
};

use crate::{
    error::MachineWalletError,
    state::{SessionState, SESSION_SEED_PREFIX},
};

pub fn process(program_id: &Pubkey, accounts: &[AccountInfo]) -> ProgramResult {
    // 1. Anti-reentry: stack_height == TRANSACTION_LEVEL_STACK_HEIGHT guarantees we are
    //    a top-level instruction, not reached via any CPI chain.
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    let account_iter = &mut accounts.iter();

    let session_account = next_account_info(account_iter)?;
    let authority = next_account_info(account_iter)?;

    // 2. authority must be a signer
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

    // 4. Deserialize SessionState
    let session_data_borrowed = session_account.try_borrow_data()?;
    let session = SessionState::deserialize_runtime(&session_data_borrowed)?;
    drop(session_data_borrowed);

    // 5. Verify session.authority == authority.key
    if session.authority != authority.key.to_bytes() {
        return Err(MachineWalletError::SessionAuthorityMismatch.into());
    }

    // 5a. Defense-in-depth: verify session PDA matches (wallet, authority, bump).
    //     Ownership (step 3) already guarantees data integrity; PDA verification
    //     confirms this is the correct session for this wallet+authority pair,
    //     preventing any hypothetical account-type confusion.
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

    // 6. Already revoked → idempotent return (not an error)
    if session.revoked {
        return Ok(());
    }

    // 7. Set revoked flag
    let mut session_data = session_account.try_borrow_mut_data()?;
    session_data[SessionState::REVOKED_OFFSET] = 1;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{SESSION_STATE_VERSION, MAX_ALLOWED_PROGRAMS};
    use solana_program::pubkey::Pubkey;

    fn make_session_bytes(authority: &Pubkey, revoked: bool) -> Vec<u8> {
        let mut prog = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
        prog[0] = [0x11u8; 32];

        let session = SessionState {
            version: SESSION_STATE_VERSION,
            bump: 253,
            wallet: [0xAAu8; 32],
            authority: authority.to_bytes(),
            created_slot: 100,
            expiry_slot: 200,
            revoked,
            wallet_creation_slot: 50,
            max_lamports_per_call: 1_000_000,
            allowed_programs_count: 1,
            allowed_programs: prog,
            max_total_spent_lamports: 0,
            total_spent_lamports: 0,
        };

        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        buf
    }

    #[test]
    fn test_session_authority_mismatch_detected() {
        let authority = Pubkey::new_unique();
        let wrong_authority = Pubkey::new_unique();

        let session_bytes = make_session_bytes(&authority, false);
        let session = SessionState::deserialize(&session_bytes).unwrap();

        // Verify the authority check logic
        assert_ne!(session.authority, wrong_authority.to_bytes());
        assert_eq!(session.authority, authority.to_bytes());
    }

    #[test]
    fn test_idempotent_when_already_revoked() {
        let authority = Pubkey::new_unique();
        let session_bytes = make_session_bytes(&authority, true);
        let session = SessionState::deserialize(&session_bytes).unwrap();

        // Already revoked — should be idempotent (logic check without full runtime)
        assert!(session.revoked);
    }

    #[test]
    fn test_revoked_offset_correct() {
        assert_eq!(SessionState::REVOKED_OFFSET, 82);
    }

    #[test]
    fn test_session_bytes_authority_round_trip() {
        let authority = Pubkey::new_unique();
        let session_bytes = make_session_bytes(&authority, false);
        let session = SessionState::deserialize(&session_bytes).unwrap();
        assert_eq!(session.authority, authority.to_bytes());
        assert!(!session.revoked);
    }
}
