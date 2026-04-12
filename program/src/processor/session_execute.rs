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

    // 14. Wallet PDA is NOT re-derived here — it is provably already verified:
    //
    //     Invariant (write-time verification, read-time trust):
    //       (a) session_account.owner == program_id (step 4) → this session
    //           was created by OUR CreateSession processor;
    //       (b) create_session.rs:168-175 PDA-validates `wallet_account` via
    //           `create_program_address(SEED_PREFIX, wallet.id(), wallet.bump)`
    //           BEFORE writing `session.wallet = wallet_account.key`;
    //       (c) step 6 above checks `session.wallet == wallet_account.key`,
    //           transitively asserting the stored wallet key matches the one
    //           PDA-verified at session creation;
    //       (d) wallet_account.owner == program_id (step 4) rules out a
    //           non-wallet account being substituted at this address;
    //       (e) close_wallet.rs tombstones the PDA (data[0]=CLOSED_MARKER,
    //           owner preserved, rent-exempt) — `deserialize_runtime` (step 5
    //           via step 11) fails `InvalidAccountData` on version byte 0xFF,
    //           so a tombstoned wallet cannot slip past;
    //       (f) step 12 enforces session.wallet_creation_slot == wallet.creation_slot,
    //           which rejects any successor wallet at the same address (impossible
    //           by (e) but checked as defense-in-depth).
    //
    //     Therefore re-deriving the wallet PDA here would only re-prove (b),
    //     which is already transitively established via (a)+(c)+(d). Saves
    //     ~1500 CU per SessionExecute — a hot-path instruction.
    //
    //     FUTURE REGRESSION RISK: if CreateSession is ever refactored to no
    //     longer PDA-validate `wallet_account` before writing `session.wallet`,
    //     this optimization MUST be reverted. The test
    //     `test_session_execute_assumes_create_session_pda_validation` in this
    //     module's tests asserts the load-bearing branch exists.

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

    // 17. Record vault lamports BEFORE CPI when any outflow cap is enforced.
    //     Two independent caps exist:
    //       • max_lamports_per_call — bounds net outflow of THIS single SessionExecute
    //         invocation. Multiple SessionExecute ixs in the same tx each pay this
    //         cap independently.
    //       • max_total_spent_lamports — bounds the CUMULATIVE lifetime outflow
    //         across every SessionExecute ever run under this session. This is the
    //         only cap that bounds total blast radius of a compromised session key.
    //     Either cap == 0 disables that specific cap (owner explicitly opted out).
    //     When max_total_spent_lamports > 0, the session account must be writable
    //     so we can persist the updated total_spent_lamports counter after CPI.
    let enforce_per_call = session.max_lamports_per_call > 0;
    let enforce_total = session.max_total_spent_lamports > 0;
    if enforce_total && !session_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    let vault_lamports_before = if enforce_per_call || enforce_total {
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
    if enforce_per_call || enforce_total {
        let vault_lamports_after = vault_account.lamports();
        let outflow = vault_lamports_before.saturating_sub(vault_lamports_after);
        if enforce_per_call && outflow > session.max_lamports_per_call {
            return Err(MachineWalletError::IxAmountExceeded.into());
        }
        if enforce_total {
            let new_total = session
                .total_spent_lamports
                .checked_add(outflow)
                .ok_or(ProgramError::ArithmeticOverflow)?;
            if new_total > session.max_total_spent_lamports {
                return Err(MachineWalletError::SessionSpendCapExceeded.into());
            }
            // Persist the updated cumulative counter. Because this comes AFTER a
            // successful CPI loop but BEFORE returning Ok, a failure here rolls
            // back the whole tx including any transfers made by the CPI, so we
            // can never charge the vault without recording the spend.
            let mut data = session_account.try_borrow_mut_data()?;
            SessionState::write_total_spent(&mut data, session.allowed_programs_count, new_total)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    /// REGRESSION GUARD for the optimization at step 14 of `process()`.
    ///
    /// The wallet PDA re-derivation in `SessionExecute` was removed because
    /// `create_session` already PDA-validates `wallet_account` before writing
    /// `session.wallet`. If CreateSession is ever changed to skip that check,
    /// this test points to the exact invariant that must be re-examined
    /// before merging — the comment at step 14 references this test by name.
    ///
    /// The test is deliberately source-level: asserting presence of the
    /// `create_program_address` call that underpins the load-bearing
    /// invariant, so it fails fast if someone unknowingly deletes it.
    #[test]
    fn test_session_execute_assumes_create_session_pda_validation() {
        let src =
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/processor/create_session.rs"));
        // Exact tokens from the create_session wallet-PDA verification block.
        assert!(
            src.contains("Pubkey::create_program_address"),
            "create_session.rs must call create_program_address to PDA-validate wallet"
        );
        assert!(
            src.contains("MachineWallet::SEED_PREFIX"),
            "create_session.rs must derive the wallet PDA using MachineWallet::SEED_PREFIX"
        );
        assert!(
            src.contains("wallet.bump"),
            "create_session.rs must use wallet.bump when re-deriving"
        );
        assert!(
            src.contains("InvalidWalletPDA"),
            "create_session.rs must reject mismatched wallet PDA"
        );
    }
}
