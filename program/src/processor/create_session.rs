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
    processor::init_pda_account::init_pda_account,
    state::{
        MachineWallet, SessionState, MAX_ALLOWED_PROGRAMS, SESSION_SEED_PREFIX,
        SESSION_STATE_VERSION,
    },
    threshold,
};

/// Domain separator for CreateSession messages, prevents cross-protocol collision.
const CREATE_SESSION_TAG: &[u8] = b"machine_wallet_create_session_v0";

/// Hash session data: keccak256(session_authority || expiry_slot_le || max_lamports_per_ix_le || count_byte || programs...)
/// Zero copy — passes each program as a separate slice to hashv, avoiding the
/// 256-byte flat buffer + memcpy. hashv feeds slices sequentially into the hasher,
/// producing identical output to hashing the concatenation.
pub fn hash_session_data(
    session_authority: &[u8; 32],
    expiry_slot: u64,
    max_lamports_per_ix: u64,
    allowed_programs_count: u8,
    allowed_programs: &[[u8; 32]],
) -> [u8; 32] {
    let expiry_slot_bytes = expiry_slot.to_le_bytes();
    let max_lamports_bytes = max_lamports_per_ix.to_le_bytes();
    let count_byte = [allowed_programs_count];

    // 4 fixed slices + up to MAX_ALLOWED_PROGRAMS program slices, no flat buffer copy
    let mut slices: [&[u8]; 4 + MAX_ALLOWED_PROGRAMS] = [&[]; 4 + MAX_ALLOWED_PROGRAMS];
    slices[0] = session_authority;
    slices[1] = &expiry_slot_bytes;
    slices[2] = &max_lamports_bytes;
    slices[3] = &count_byte;
    let count = allowed_programs.len();
    for (i, prog) in allowed_programs.iter().enumerate() {
        slices[4 + i] = prog;
    }

    keccak::hashv(&slices[..4 + count]).to_bytes()
}

/// Compute create-session message: keccak256(CREATE_SESSION_TAG || wallet_pda || creation_slot_le || nonce_le || max_slot_le || session_data_hash)
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_create_session_message(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    session_data_hash: &[u8; 32],
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        CREATE_SESSION_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        session_data_hash,
    ])
    .to_bytes()
}

#[allow(clippy::too_many_arguments)] // fields map 1:1 from wire format, a wrapper struct adds no clarity
pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    secp256r1_ix_index: u8,
    max_slot: u64,
    session_authority: [u8; 32],
    expiry_slot: u64,
    max_lamports_per_ix: u64,
    allowed_programs_count: u8,
    allowed_programs: [[u8; 32]; MAX_ALLOWED_PROGRAMS],
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;
    let session_account = next_account_info(account_iter)?;
    let system_program = next_account_info(account_iter)?;

    // 1. Basic validation: cheap field checks first — fail fast before any syscall
    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable || !session_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    // 2. Anti-reentry: stack_height == TRANSACTION_LEVEL_STACK_HEIGHT guarantees we are
    //    a top-level instruction, not reached via any CPI chain.
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 3. Validate: allowed_programs_count == 0 → useless session
    if session_authority == [0u8; 32] {
        return Err(MachineWalletError::InvalidSessionData.into());
    }
    if allowed_programs_count == 0 {
        return Err(MachineWalletError::InvalidSessionData.into());
    }
    if allowed_programs_count as usize > MAX_ALLOWED_PROGRAMS {
        return Err(MachineWalletError::TooManyAllowedPrograms.into());
    }
    for i in 0..allowed_programs_count as usize {
        for j in 0..i {
            if allowed_programs[i] == allowed_programs[j] {
                return Err(MachineWalletError::InvalidSessionData.into());
            }
        }
    }

    // 4. Signature expiry: current slot must not exceed max_slot
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 5. Session expiry must be in the future
    if expiry_slot <= clock.slot {
        return Err(MachineWalletError::InvalidSessionData.into());
    }

    // 6. Load and deserialize MachineWallet state
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 7. Verify wallet PDA using cached bump
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 8. Compute session data hash, then full message hash
    let session_data_hash = hash_session_data(
        &session_authority,
        expiry_slot,
        max_lamports_per_ix,
        allowed_programs_count,
        &allowed_programs[..allowed_programs_count as usize],
    );
    let expected_message = compute_create_session_message(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        &session_data_hash,
    );

    // 9. Signature verification (v0: single precompile, v1: threshold scan)
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        &wallet,
        secp256r1_ix_index,
        &expected_message,
    )?;

    // 10. Derive session PDA and verify matches provided account.
    // Uses find_program_address because the session doesn't exist yet (no cached bump).
    let (expected_session_pda, session_bump) = Pubkey::find_program_address(
        &[
            SESSION_SEED_PREFIX,
            wallet_account.key.as_ref(),
            &session_authority,
        ],
        program_id,
    );
    if *session_account.key != expected_session_pda {
        return Err(MachineWalletError::InvalidSessionPDA.into());
    }

    // 11. Increment nonce BEFORE side effects (CEI pattern).
    //
    // SECURITY: The nonce must be incremented before creating the session account as
    // defense-in-depth. The primary anti-reentry guard is get_stack_height() (step 2).
    // The CEI nonce pattern is a secondary safeguard: even if the stack-height check
    // were somehow bypassed, a reentrant CreateSession would compute a different message
    // hash (nonce+1 vs signed nonce), causing MessageMismatch. If the create_account CPI
    // fails, the entire transaction rolls back atomically (nonce included).
    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;
    {
        let mut data = wallet_account.try_borrow_mut_data()?;
        let nonce_off = wallet.nonce_offset();
        data[nonce_off..nonce_off + 8].copy_from_slice(&new_nonce.to_le_bytes());
    }

    // 12. Initialize session PDA in a dust-resistant way.
    let session_signer_seeds: &[&[u8]] = &[
        SESSION_SEED_PREFIX,
        wallet_account.key.as_ref(),
        &session_authority,
        &[session_bump],
    ];
    init_pda_account(
        fee_payer,
        session_account,
        system_program,
        program_id,
        SessionState::LEN,
        session_signer_seeds,
        MachineWalletError::SessionAlreadyExists,
    )?;

    // 13. Initialize SessionState with all fields
    let session = SessionState {
        version: SESSION_STATE_VERSION,
        bump: session_bump,
        wallet: wallet_account.key.to_bytes(),
        authority: session_authority,
        created_slot: clock.slot,
        expiry_slot,
        revoked: false,
        wallet_creation_slot: wallet.creation_slot,
        max_lamports_per_ix,
        allowed_programs_count,
        allowed_programs,
    };

    // 14. Serialize SessionState into session account data
    let mut session_data = session_account.try_borrow_mut_data()?;
    session.serialize(&mut session_data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_session_data_deterministic() {
        let authority = [0xAAu8; 32];
        let prog = [0x11u8; 32];
        let hash1 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog]);
        let hash2 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_session_data_different_authority() {
        let auth1 = [0xAAu8; 32];
        let auth2 = [0xBBu8; 32];
        let prog = [0x11u8; 32];
        let hash1 = hash_session_data(&auth1, 1000, 5_000_000, 1, &[prog]);
        let hash2 = hash_session_data(&auth2, 1000, 5_000_000, 1, &[prog]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_session_data_different_expiry() {
        let authority = [0xAAu8; 32];
        let prog = [0x11u8; 32];
        let hash1 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog]);
        let hash2 = hash_session_data(&authority, 2000, 5_000_000, 1, &[prog]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_session_data_different_max_lamports() {
        let authority = [0xAAu8; 32];
        let prog = [0x11u8; 32];
        let hash1 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog]);
        let hash2 = hash_session_data(&authority, 1000, 10_000_000, 1, &[prog]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_session_data_different_programs() {
        let authority = [0xAAu8; 32];
        let prog1 = [0x11u8; 32];
        let prog2 = [0x22u8; 32];
        let hash1 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog1]);
        let hash2 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog2]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_session_data_different_count() {
        let authority = [0xAAu8; 32];
        let prog = [0x11u8; 32];
        // Same single program, but different count byte changes the hash
        let hash1 = hash_session_data(&authority, 1000, 5_000_000, 1, &[prog]);
        let hash2 = hash_session_data(&authority, 1000, 5_000_000, 2, &[prog]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_create_session_message_deterministic() {
        let wallet = Pubkey::new_unique();
        let session_hash = [0x42u8; 32];
        let msg1 = compute_create_session_message(&wallet, 100, 0, 500, &session_hash);
        let msg2 = compute_create_session_message(&wallet, 100, 0, 500, &session_hash);
        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_compute_create_session_message_includes_nonce() {
        let wallet = Pubkey::new_unique();
        let session_hash = [0x42u8; 32];
        let msg1 = compute_create_session_message(&wallet, 100, 0, 500, &session_hash);
        let msg2 = compute_create_session_message(&wallet, 100, 1, 500, &session_hash);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_create_session_message_includes_wallet() {
        let wallet1 = Pubkey::new_unique();
        let wallet2 = Pubkey::new_unique();
        let session_hash = [0x42u8; 32];
        let msg1 = compute_create_session_message(&wallet1, 100, 0, 500, &session_hash);
        let msg2 = compute_create_session_message(&wallet2, 100, 0, 500, &session_hash);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_create_session_message_includes_creation_slot() {
        let wallet = Pubkey::new_unique();
        let session_hash = [0x42u8; 32];
        let msg1 = compute_create_session_message(&wallet, 100, 0, 500, &session_hash);
        let msg2 = compute_create_session_message(&wallet, 200, 0, 500, &session_hash);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_create_session_message_includes_max_slot() {
        let wallet = Pubkey::new_unique();
        let session_hash = [0x42u8; 32];
        let msg1 = compute_create_session_message(&wallet, 100, 0, 500, &session_hash);
        let msg2 = compute_create_session_message(&wallet, 100, 0, 600, &session_hash);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_compute_create_session_message_includes_session_hash() {
        let wallet = Pubkey::new_unique();
        let hash1 = [0x42u8; 32];
        let hash2 = [0x43u8; 32];
        let msg1 = compute_create_session_message(&wallet, 100, 0, 500, &hash1);
        let msg2 = compute_create_session_message(&wallet, 100, 0, 500, &hash2);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_domain_tag_prevents_cross_message_collision() {
        // CreateSession and Execute messages with same parameters should differ due to domain tags
        let wallet = Pubkey::new_unique();
        let data_hash = [0x42u8; 32];
        let session_msg = compute_create_session_message(&wallet, 100, 0, 500, &data_hash);

        // Manually compute an execute-style hash to verify they differ
        let execute_msg =
            crate::processor::execute::compute_message_hash(&wallet, 100, 0, 500, &data_hash);

        assert_ne!(session_msg, execute_msg);
    }
}
