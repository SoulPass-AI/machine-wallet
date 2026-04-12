use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    keccak,
    program::invoke_signed,
    program_error::ProgramError,
    pubkey::Pubkey,
    rent::Rent,
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

    // 11. Signature verification.
    threshold::verify_wallet_signatures(
        instructions_sysvar,
        program_id,
        &wallet,
        secp256r1_ix_index,
        &expected_message,
    )?;

    // 14. Drain vault SOL to destination via system_program::transfer CPI.
    //     Vault stays system-owned. Vault reuse across a close+recreate is NOT
    //     a replay concern by itself — the vault PDA is deterministic from the
    //     wallet pubkey, and any asset it holds is reachable only via signing
    //     over the wallet's state machine. The replay surface is the wallet
    //     PDA; the vault tombstone follows from tombstoning the wallet.
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

    // 15. TOMBSTONE the wallet PDA — first-principles fix for the
    //     close-then-recreate replay surface.
    //
    //     PROBLEM (what a plain zero-lamport drain would allow):
    //     Solana's `Clock::slot` advances per-slot, NOT per-transaction. All
    //     transactions within a single slot observe the same `clock.slot`
    //     value. If we drained this PDA to zero lamports, the runtime would
    //     reclaim the account at tx end (owner → SystemProgram, data_len → 0),
    //     making the seeds available to a fresh `CreateWallet` in ANY later
    //     transaction — including a LATER TX IN THE SAME SLOT. The resurrected
    //     wallet would carry identical `wallet_id`, identical wallet PDA,
    //     identical vault PDA, identical `creation_slot` (same slot!), and
    //     `nonce = 0`. Any owner-signed Execute with `(nonce=0,
    //     creation_slot=S)` pre-signed but not yet broadcast would match
    //     byte-for-byte and replay against the new wallet, with full signing
    //     authority over the vault's still-extant SPL-token / NFT associated
    //     accounts.
    //
    //     The weaker "historical authority" variant does not even require
    //     same-slot timing: a previously-rotated-out authority whose private
    //     key was compromised can call permissionless `CreateWallet` at any
    //     future slot, seize the seeds, and drain residual vault-ATA assets
    //     with freshly-signed Executes.
    //
    //     FIX: keep the PDA alive forever as a 1-byte tombstone:
    //       (a) owned by `program_id` — so `init_pda_account`'s
    //           `owner == program_id → WalletAlreadyInitialized` short-circuit
    //           blocks every future `CreateWallet` on these seeds, forever;
    //       (b) funded at exactly `rent.minimum_balance(1)` — so the runtime's
    //           zero-lamport GC never fires and (a) cannot be undone by
    //           reassignment;
    //       (c) stamped with `data[0] = CLOSED_MARKER (0xFF)` — so every
    //           state-reading processor's `deserialize_runtime` returns
    //           `InvalidAccountData` at the version-dispatch step, hard-
    //           failing any accidental or malicious re-entry against this
    //           PDA without requiring per-processor guards.
    //
    //     COST — WHY WE SHRINK TO 1 BYTE:
    //     The original account was sized for its authority list (87 B for a
    //     1-auth v1 wallet, up to 597 B for a 16-auth v1 wallet). Keeping
    //     that full size as a tombstone would permanently lock ~0.0015 to
    //     ~0.0050 SOL per closed wallet. But the tombstone only needs ONE
    //     byte of data — byte 0 carrying `CLOSED_MARKER`. We therefore
    //     `resize(1)` first, which drops the permanent lock to
    //     `rent.minimum_balance(1) ≈ 0.000898 SOL` (all within the Solana
    //     128-byte per-account overhead — shrinking to 0 bytes would only
    //     save ~7k lamports more while losing the explicit sentinel, so
    //     1 byte is the right point on the size/clarity curve).
    //
    //     The rent freed by shrinking (plus any historical over-funding or
    //     `RemoveAuthority`-shrink residue) flows to `destination`. For a
    //     max-size wallet this returns ~82% of the original rent.
    const TOMBSTONE_SIZE: usize = 1;
    wallet_account.resize(TOMBSTONE_SIZE)?;
    let rent = Rent::get()?;
    let required_tombstone_lamports = rent.minimum_balance(TOMBSTONE_SIZE);
    let current_wallet_lamports = wallet_account.lamports();
    // `resize` does not touch lamports. `minimum_balance(1)` is well below
    // `minimum_balance(old_size)` for any real wallet, so `current >=
    // required` always holds and `saturating_sub` is defensive only.
    let excess = current_wallet_lamports.saturating_sub(required_tombstone_lamports);
    if excess > 0 {
        **wallet_account.try_borrow_mut_lamports()? = current_wallet_lamports - excess;
        **destination.try_borrow_mut_lamports()? = destination
            .lamports()
            .checked_add(excess)
            .ok_or(ProgramError::ArithmeticOverflow)?;
    }

    // 16. Stamp the CLOSED_MARKER. After this point:
    //       - deserialize_runtime(this account) → InvalidAccountData (version
    //         byte is 0xFF, not 0 or 1; and shrunk buffers fail the length
    //         check inside deserialize_v0/v1 anyway — defense-in-depth)
    //       - init_pda_account(this PDA) → WalletAlreadyInitialized (owner is
    //         program_id, short-circuits before the SystemProgram / empty /
    //         executable checks)
    //     Both branches fail closed without any version-aware logic in callers.
    {
        let mut data = wallet_account.try_borrow_mut_data()?;
        data[0] = MachineWallet::CLOSED_MARKER;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tombstone sentinel MUST live outside the valid-version space.
    /// If a future layout upgrade introduced `version = 0xFF` as a real
    /// format, tombstoned wallets could be mis-deserialized as live state
    /// and re-authorized — collapsing the whole close-replay defense.
    /// Keeping this at 0xFF (and keeping valid versions monotonically small)
    /// is a load-bearing invariant. This test fails loudly if either side
    /// drifts.
    #[test]
    fn test_closed_marker_disjoint_from_valid_versions() {
        assert_eq!(MachineWallet::CLOSED_MARKER, 0xFF);
        assert_ne!(MachineWallet::CLOSED_MARKER, 0); // v0
        assert_ne!(MachineWallet::CLOSED_MARKER, 1); // v1
    }

    /// A tombstoned wallet's bytes must not parse back into any live layout.
    /// This is the property that lets every other processor fail-closed on
    /// tombstones without per-processor guards. Covers three buffer sizes
    /// that could plausibly appear after a resize:
    ///   - 1 byte (the canonical post-close tombstone shape)
    ///   - v0/v1 1-auth layout (if a future patch ever forgot to shrink)
    ///   - max-size v1 layout with 16 authorities
    #[test]
    fn test_tombstoned_bytes_fail_deserialize() {
        let tombstone_buf = [MachineWallet::CLOSED_MARKER];
        let mut legacy_buf = [0u8; MachineWallet::LEN];
        legacy_buf[0] = MachineWallet::CLOSED_MARKER;
        let mut big_buf = vec![0u8; MachineWallet::v1_account_size(16)];
        big_buf[0] = MachineWallet::CLOSED_MARKER;

        assert!(MachineWallet::deserialize(&tombstone_buf).is_err());
        assert!(MachineWallet::deserialize_runtime(&tombstone_buf).is_err());
        assert!(MachineWallet::deserialize(&legacy_buf).is_err());
        assert!(MachineWallet::deserialize_runtime(&legacy_buf).is_err());
        assert!(MachineWallet::deserialize(&big_buf).is_err());
        assert!(MachineWallet::deserialize_runtime(&big_buf).is_err());
    }

    /// Rent floor sanity check — documents the permanent-lock cost and
    /// catches accidental drift if `TOMBSTONE_SIZE` ever gets bumped. The
    /// absolute numbers below depend on Solana's rent parameters; we assert
    /// only structural inequalities that remain true under any reasonable
    /// parameter change.
    #[test]
    fn test_tombstone_rent_is_minimal() {
        use solana_program::rent::Rent;
        let rent = Rent::default();
        let tombstone_rent = rent.minimum_balance(1);
        // A 1-byte tombstone must cost strictly less than any real wallet
        // size — otherwise shrinking provides no refund.
        let min_wallet_rent = rent.minimum_balance(MachineWallet::LEN);
        let max_wallet_rent = rent.minimum_balance(MachineWallet::v1_account_size(16));
        assert!(tombstone_rent < min_wallet_rent);
        assert!(tombstone_rent < max_wallet_rent);
    }

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
