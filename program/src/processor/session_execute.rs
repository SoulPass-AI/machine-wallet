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
    instruction::InnerInstructionRef,
    state::{MachineWallet, SessionState, SESSION_SEED_PREFIX, SYSTEM_PROGRAM_ID},
};

use super::execute::find_program_account_index;

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    inner_instructions: Vec<InnerInstructionRef<'_>>,
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let session_account = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let authority = next_account_info(account_iter)?;
    let vault_account = next_account_info(account_iter)?;
    let remaining_accounts = &accounts[4..];

    // 1. Reject empty inner_instructions
    if inner_instructions.is_empty() {
        return Err(ProgramError::InvalidInstructionData);
    }

    // 2. Anti-reentry: must be a top-level instruction
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 3. Authority must be a signer (Ed25519 session key)
    if !authority.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    // 4. Writable + ownership checks
    //    wallet_account is intentionally readonly — SessionExecute does not
    //    modify wallet state (no nonce increment). Minimum-privilege principle.
    if !vault_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if session_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // 5. Deserialize SessionState
    let session_data = session_account.try_borrow_data()?;
    let session = SessionState::deserialize_runtime(&session_data)?;
    drop(session_data);

    // 6. Verify session.wallet == wallet_account.key
    if session.wallet != wallet_account.key.to_bytes() {
        return Err(MachineWalletError::SessionWalletMismatch.into());
    }

    // 7. Verify session.authority == authority.key
    if session.authority != authority.key.to_bytes() {
        return Err(MachineWalletError::SessionAuthorityMismatch.into());
    }

    // 8. Session must not be revoked
    if session.revoked {
        return Err(MachineWalletError::SessionRevoked.into());
    }

    // 9. Session must not be expired
    let clock = Clock::get()?;
    if clock.slot > session.expiry_slot {
        return Err(MachineWalletError::SessionExpired.into());
    }

    // 10. Validate inner instructions BEFORE expensive PDA verification (~4500 CU saved
    //     on validation failures: program whitelist, writable flags, missing accounts).
    //     This loop only depends on session state (loaded above) and remaining_accounts.
    let mut program_entries = Vec::with_capacity(inner_instructions.len());

    for inner_ix in &inner_instructions {
        let target_program_id = Pubkey::new_from_array(inner_ix.program_id);
        if target_program_id == *program_id {
            return Err(MachineWalletError::CpiToSelfDenied.into());
        }

        // Program whitelist check
        if !session.is_program_allowed(&inner_ix.program_id) {
            return Err(MachineWalletError::ProgramNotAllowed.into());
        }

        for entry in inner_ix.accounts() {
            let account = remaining_accounts
                .get(entry.index as usize)
                .ok_or(ProgramError::NotEnoughAccountKeys)?;
            if entry.is_writable() && !account.is_writable {
                return Err(MachineWalletError::AccountNotWritable.into());
            }
        }

        let idx = find_program_account_index(remaining_accounts, &target_program_id)?;
        program_entries.push((idx, target_program_id));
    }

    // 11. Load wallet state (needed for creation_slot check + vault_bump for CPI)
    let wallet_data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&wallet_data)?;
    drop(wallet_data);

    // 12. Verify session.wallet_creation_slot == wallet.creation_slot
    //     Prevents session resurrection after close+recreate
    if session.wallet_creation_slot != wallet.creation_slot {
        return Err(MachineWalletError::SessionWalletMismatch.into());
    }

    // 13. Verify session PDA (create_program_address ~1500 CU)
    let expected_session_pda = Pubkey::create_program_address(
        &[
            SESSION_SEED_PREFIX,
            wallet_account.key.as_ref(),
            authority.key.as_ref(),
            &[session.bump],
        ],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidSessionPDA)?;
    if *session_account.key != expected_session_pda {
        return Err(MachineWalletError::InvalidSessionPDA.into());
    }

    // 14. Verify wallet PDA (create_program_address ~1500 CU)
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 15. Verify vault PDA (create_program_address ~1500 CU)
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

    // 16. Vault must be system-owned
    if *vault_account.owner != SYSTEM_PROGRAM_ID {
        return Err(MachineWalletError::InvalidVaultOwner.into());
    }

    // 17. Record vault lamports BEFORE CPI if max_lamports_per_ix > 0.
    //     This cap is per SessionExecute invocation — a single transaction may
    //     contain multiple SessionExecute instructions, each independently capped.
    //     When max_lamports_per_ix == 0, no cap is enforced (the wallet owner
    //     explicitly authorized an uncapped session).
    let vault_lamports_before = if session.max_lamports_per_ix > 0 {
        vault_account.lamports()
    } else {
        0
    };

    // 18. Execute CPI for each inner instruction (shared with execute.rs)
    let vault_signer_seeds: &[&[u8]] = &[
        MachineWallet::VAULT_SEED_PREFIX,
        wallet_account.key.as_ref(),
        &[wallet.vault_bump],
    ];

    super::execute::execute_cpi_loop(
        &inner_instructions,
        program_entries,
        remaining_accounts,
        vault_account,
        vault_signer_seeds,
    )?;

    // 19. SOL outflow cap check (atomic rollback if exceeded).
    //     SECURITY: We compare against the starting balance, not net flow.
    //     If vault_lamports_after > vault_lamports_before (vault received SOL during CPI),
    //     outflow is 0 — we do NOT allow inflows to offset a larger subsequent withdrawal.
    //     This prevents a "deposit-then-overdraw" attack where an attacker deposits X SOL
    //     into the vault mid-CPI, then withdraws X + cap, making the net look within cap.
    //     The cap bounds the maximum SOL the vault can LOSE in a single SessionExecute.
    if session.max_lamports_per_ix > 0 {
        let vault_lamports_after = vault_account.lamports();
        if vault_lamports_after < vault_lamports_before {
            let outflow = vault_lamports_before - vault_lamports_after;
            if outflow > session.max_lamports_per_ix {
                return Err(MachineWalletError::IxAmountExceeded.into());
            }
        }
    }

    Ok(())
}
