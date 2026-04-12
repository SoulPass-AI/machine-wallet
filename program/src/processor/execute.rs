use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, AccountMeta, Instruction, TRANSACTION_LEVEL_STACK_HEIGHT},
    keccak,
    program::invoke_signed,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::Sysvar,
};

use crate::{
    error::MachineWalletError,
    instruction::{InnerInstructionRef, MAX_INNER_INSTRUCTIONS},
    state::{MachineWallet, SYSTEM_PROGRAM_ID},
    threshold,
};

/// Domain separator for Execute messages, prevents cross-protocol collision.
const EXECUTE_TAG: &[u8] = b"machine_wallet_execute_v0";

/// Hash inner instructions: keccak256(ix0_hash || ix1_hash || ...)
/// Each ix_hash = keccak256(program_id || accounts_len || (pubkey, flags)... || data_len || data)
/// Length prefixes + flags prevent domain confusion, while concrete pubkey binding
/// prevents a relay from swapping `remaining_accounts` without invalidating the signature.
///
/// Uses incremental Hasher — only available off-chain (tests / client SDK).
/// On-chain code mirrors this layout in `validate_and_hash_inner_instructions`.
#[cfg(not(any(target_os = "solana", target_arch = "bpf")))]
pub fn hash_inner_instructions(
    instructions: &[crate::instruction::InnerInstruction],
    remaining_account_keys: &[Pubkey],
) -> Result<[u8; 32], ProgramError> {
    let mut outer = keccak::Hasher::default();
    for ix in instructions {
        let accounts_len = ix.accounts.len() as u16;
        let data_len = ix.data.len() as u16;
        let mut inner = keccak::Hasher::default();
        inner.hash(&ix.program_id);
        inner.hash(&accounts_len.to_le_bytes());
        for acc in &ix.accounts {
            let key = remaining_account_keys
                .get(acc.index as usize)
                .ok_or(ProgramError::NotEnoughAccountKeys)?;
            inner.hash(key.as_ref());
            inner.hash(&[acc.flags]);
        }
        inner.hash(&data_len.to_le_bytes());
        inner.hash(&ix.data);
        outer.hash(&inner.result().to_bytes());
    }
    Ok(outer.result().to_bytes())
}

pub(crate) fn find_program_account_index(
    remaining_accounts: &[AccountInfo],
    target_program_id: &Pubkey,
) -> Result<usize, ProgramError> {
    remaining_accounts
        .iter()
        .position(|acc| acc.key == target_program_id)
        .ok_or(MachineWalletError::MissingProgramAccount.into())
}

/// Per-instruction program account index + resolved Pubkey, paired with the inner hash.
type ValidatedInstructions = (Vec<(usize, Pubkey)>, [u8; 32]);

/// Validate all inner instruction accounts AND compute the inner instructions hash
/// in a single pass. Saves one full iteration + eliminates the separate hash function.
pub(crate) fn validate_and_hash_inner_instructions(
    inner_instructions: &[InnerInstructionRef<'_>],
    remaining_accounts: &[AccountInfo],
    program_id: &Pubkey,
) -> Result<ValidatedInstructions, ProgramError> {
    let mut program_entries = Vec::with_capacity(inner_instructions.len());
    let mut hash_buf = [0u8; MAX_INNER_INSTRUCTIONS * 32];
    let mut resolved_accounts_buf = Vec::new();

    for (i, inner_ix) in inner_instructions.iter().enumerate() {
        let target_program_id = Pubkey::new_from_array(inner_ix.program_id);
        if target_program_id == *program_id {
            return Err(MachineWalletError::CpiToSelfDenied.into());
        }

        resolved_accounts_buf.clear();
        let resolved_accounts_len = inner_ix
            .account_count()
            .checked_mul(33)
            .ok_or(ProgramError::InvalidInstructionData)?;
        resolved_accounts_buf.reserve(resolved_accounts_len);

        for entry in inner_ix.accounts() {
            let account = remaining_accounts
                .get(entry.index as usize)
                .ok_or(ProgramError::NotEnoughAccountKeys)?;
            if entry.is_writable() && !account.is_writable {
                return Err(MachineWalletError::AccountNotWritable.into());
            }
            resolved_accounts_buf.extend_from_slice(account.key.as_ref());
            resolved_accounts_buf.push(entry.flags);
        }

        let idx = find_program_account_index(remaining_accounts, &target_program_id)?;
        program_entries.push((idx, target_program_id));

        // Hash the resolved instruction in the same pass.
        // Safety: Solana tx max 1232 bytes, so neither value can exceed u16::MAX.
        debug_assert!(inner_ix.account_count() <= u16::MAX as usize);
        debug_assert!(inner_ix.data.len() <= u16::MAX as usize);
        let accounts_len_bytes = (inner_ix.account_count() as u16).to_le_bytes();
        let data_len_bytes = (inner_ix.data.len() as u16).to_le_bytes();
        let h = keccak::hashv(&[
            &inner_ix.program_id,
            &accounts_len_bytes,
            &resolved_accounts_buf,
            &data_len_bytes,
            inner_ix.data,
        ]);
        hash_buf[i * 32..(i + 1) * 32].copy_from_slice(h.as_ref());
    }

    let inner_hash = keccak::hash(&hash_buf[..inner_instructions.len() * 32]).to_bytes();
    Ok((program_entries, inner_hash))
}

/// Execute CPI for each inner instruction via invoke_signed with vault PDA seeds.
/// Shared by Execute (P-256 path) and SessionExecute (Ed25519 path).
pub(crate) fn execute_cpi_loop(
    inner_instructions: &[InnerInstructionRef<'_>],
    program_entries: Vec<(usize, Pubkey)>,
    remaining_accounts: &[AccountInfo],
    vault_account: &AccountInfo,
    vault_signer_seeds: &[&[u8]],
) -> ProgramResult {
    // Pre-size to the worst-case inner instruction so the first iteration
    // doesn't trigger a heap reallocation. `+1` on account_infos accounts
    // for the program account pushed at the end of each iteration.
    let max_accounts = inner_instructions
        .iter()
        .map(|ix| ix.account_count())
        .max()
        .unwrap_or(0);
    let max_data = inner_instructions
        .iter()
        .map(|ix| ix.data.len())
        .max()
        .unwrap_or(0);
    let mut account_infos = Vec::with_capacity(max_accounts + 1);
    let mut account_metas = Vec::with_capacity(max_accounts);
    let mut cpi_data = Vec::with_capacity(max_data);
    for (inner_ix, (program_account_index, target_program_id)) in
        inner_instructions.iter().zip(program_entries.into_iter())
    {
        account_infos.clear();
        account_metas.clear();
        cpi_data.clear();
        cpi_data.extend_from_slice(inner_ix.data);

        for entry in inner_ix.accounts() {
            let acc = remaining_accounts
                .get(entry.index as usize)
                .ok_or(ProgramError::NotEnoughAccountKeys)?;

            let is_signer = acc.key == vault_account.key;

            if entry.is_writable() {
                account_metas.push(AccountMeta::new(*acc.key, is_signer));
            } else {
                account_metas.push(AccountMeta::new_readonly(*acc.key, is_signer));
            }

            account_infos.push(acc.clone());
        }

        let program_account = &remaining_accounts[program_account_index];
        account_infos.push(program_account.clone());

        let cpi_instruction = Instruction {
            program_id: target_program_id,
            accounts: core::mem::take(&mut account_metas),
            data: core::mem::take(&mut cpi_data),
        };

        invoke_signed(&cpi_instruction, &account_infos, &[vault_signer_seeds])?;
        account_metas = cpi_instruction.accounts;
        cpi_data = cpi_instruction.data;
    }
    Ok(())
}

/// Compute message: keccak256(EXECUTE_TAG || wallet_address || creation_slot_le || nonce_le || max_slot_le || inner_instructions_hash)
/// Domain tag differentiates Execute from CloseWallet messages.
/// creation_slot binds the signature to a specific wallet lifetime, preventing replay after close+recreate.
/// max_slot enforces signature expiry — the transaction must land before this slot.
/// Zero heap allocation — uses hashv with stack-local byte arrays.
pub fn compute_message_hash(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    inner_hash: &[u8; 32],
) -> [u8; 32] {
    let creation_slot_bytes = creation_slot.to_le_bytes();
    let nonce_bytes = nonce.to_le_bytes();
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        EXECUTE_TAG,
        wallet_address.as_ref(),
        &creation_slot_bytes,
        &nonce_bytes,
        &max_slot_bytes,
        inner_hash,
    ])
    .to_bytes()
}

/// Execute preamble. Validates accounts, enforces anti-reentry, verifies PDAs,
/// loads wallet state, and computes the inner-ix hash. Threshold signature
/// verification happens next in `process`, and — via the Evidence Sidecar
/// model in `threshold::verify_threshold_signatures` — covers every wallet
/// authority scheme (SECP256R1, ED25519, WEBAUTHN, and future PQ) uniformly.
fn prepare_execute_context(
    program_id: &Pubkey,
    instructions_sysvar: &AccountInfo,
    wallet_account: &AccountInfo,
    fee_payer: &AccountInfo,
    vault_account: &AccountInfo,
    remaining_accounts: &[AccountInfo],
    max_slot: u64,
    inner_instructions: &[InnerInstructionRef<'_>],
) -> Result<(MachineWallet, Vec<(usize, Pubkey)>, [u8; 32]), ProgramError> {
    // 1. Reject empty execute (no-op would waste a nonce)
    if inner_instructions.is_empty() {
        return Err(ProgramError::InvalidInstructionData);
    }

    // 2. Cheap field checks first — fail fast before any syscall
    if !fee_payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }
    if !wallet_account.is_writable || !vault_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    if wallet_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // 3. Validate instructions sysvar
    if *instructions_sysvar.key != solana_program::sysvar::instructions::ID {
        return Err(ProgramError::InvalidAccountData);
    }

    // 4. Anti-reentry: stack_height == TRANSACTION_LEVEL_STACK_HEIGHT guarantees we are
    //    a top-level instruction, not reached via any CPI chain (direct or indirect).
    //    This is strictly stronger than get_instruction_relative(0), which only compared
    //    program_id and could pass in indirect callback scenarios (A → X → Y → A).
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 5. Pre-validate all CPI account references AND compute inner instruction hash
    //    in a single pass — saves one full iteration over inner instructions.
    let (program_entries, inner_hash) =
        validate_and_hash_inner_instructions(inner_instructions, remaining_accounts, program_id)?;

    // 6. Signature expiry: current slot must not exceed max_slot
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 7. Load and deserialize MachineWallet state
    let data = wallet_account.try_borrow_data()?;
    let wallet = MachineWallet::deserialize_runtime(&data)?;
    drop(data);

    // 8. Verify wallet PDA using cached bump (id computed from authority, ~100 CU syscall)
    let id = wallet.id();
    let expected_wallet_pda = Pubkey::create_program_address(
        &[MachineWallet::SEED_PREFIX, &id, &[wallet.bump]],
        program_id,
    )
    .map_err(|_| MachineWalletError::InvalidWalletPDA)?;
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 9. Verify vault PDA using cached vault_bump (saves compute vs find_program_address)
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

    // 10. Verify vault is system-owned (vault stays system-owned so that
    //     system_program::transfer CPI works naturally as the source).
    if *vault_account.owner != SYSTEM_PROGRAM_ID {
        return Err(MachineWalletError::InvalidVaultOwner.into());
    }

    Ok((wallet, program_entries, inner_hash))
}

pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    secp256r1_ix_index: u8,
    max_slot: u64,
    inner_instructions: Vec<InnerInstructionRef<'_>>,
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let fee_payer = next_account_info(account_iter)?;
    let vault_account = next_account_info(account_iter)?;
    let remaining_accounts = &accounts[4..];

    let (wallet, program_entries, inner_hash) = prepare_execute_context(
        program_id,
        instructions_sysvar,
        wallet_account,
        fee_payer,
        vault_account,
        remaining_accounts,
        max_slot,
        &inner_instructions,
    )?;

    // Compute expected 32-byte execute message hash.
    let expected_message = compute_message_hash(
        wallet_account.key,
        wallet.creation_slot,
        wallet.nonce,
        max_slot,
        &inner_hash,
    );

    threshold::verify_wallet_signatures(
        instructions_sysvar,
        program_id,
        &wallet,
        secp256r1_ix_index,
        &expected_message,
    )?;

    // Increment nonce BEFORE CPI (checks-effects-interactions pattern).
    //
    // SECURITY: The nonce must be incremented before executing any CPI as defense-in-depth.
    // The primary anti-reentry guard is get_stack_height() in prepare_execute_context,
    // which hard-rejects all CPI invocations regardless of call chain depth. The CEI
    // nonce pattern is a secondary safeguard: even if the stack-height check were
    // somehow bypassed, a reentrant Execute would compute a different message hash
    // (nonce+1 vs signed nonce), causing MessageMismatch. If any CPI fails, the entire
    // transaction rolls back atomically (nonce included).
    let new_nonce = wallet
        .nonce
        .checked_add(1)
        .ok_or(MachineWalletError::InvalidNonce)?;
    {
        let mut data = wallet_account.try_borrow_mut_data()?;
        let nonce_off = wallet.nonce_offset();
        data[nonce_off..nonce_off + 8].copy_from_slice(&new_nonce.to_le_bytes());
    }

    // Execute CPI for each inner instruction.
    // SECURITY: inner_hash is computed over the resolved Pubkeys, not raw indices.
    // Reordering or substituting `remaining_accounts` now changes the signed message.
    let vault_signer_seeds: &[&[u8]] = &[
        MachineWallet::VAULT_SEED_PREFIX,
        wallet_account.key.as_ref(),
        &[wallet.vault_bump],
    ];

    execute_cpi_loop(
        &inner_instructions,
        program_entries,
        remaining_accounts,
        vault_account,
        vault_signer_seeds,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::{AccountEntry, InnerInstruction};

    fn make_inner_instruction(
        program_id_byte: u8,
        accounts: &[(u8, u8)],
        data: &[u8],
    ) -> InnerInstruction {
        InnerInstruction {
            program_id: [program_id_byte; 32],
            accounts: accounts
                .iter()
                .map(|&(index, flags)| AccountEntry { index, flags })
                .collect(),
            data: data.to_vec(),
        }
    }

    fn make_account_info(key: Pubkey) -> AccountInfo<'static> {
        make_account_info_with_writable(key, false)
    }

    fn make_account_info_with_writable(key: Pubkey, is_writable: bool) -> AccountInfo<'static> {
        let key = Box::leak(Box::new(key));
        let lamports = Box::leak(Box::new(0u64));
        let data = Box::leak(Vec::<u8>::new().into_boxed_slice());
        let owner = Box::leak(Box::new(Pubkey::new_unique()));
        AccountInfo::new(key, false, is_writable, lamports, data, owner, false)
    }

    fn make_account_keys(count: usize) -> Vec<Pubkey> {
        (0..count).map(|_| Pubkey::new_unique()).collect()
    }

    #[test]
    fn test_hash_inner_instructions_deterministic() {
        let instructions = vec![
            make_inner_instruction(0xAA, &[(0, 0x01), (1, 0x00)], &[0xDE, 0xAD]),
            make_inner_instruction(0xBB, &[(2, 0x01)], &[0xBE, 0xEF]),
        ];
        let account_keys = make_account_keys(3);

        let hash1 = hash_inner_instructions(&instructions, &account_keys).unwrap();
        let hash2 = hash_inner_instructions(&instructions, &account_keys).unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_inner_instructions_different() {
        let ix1 = vec![make_inner_instruction(
            0xAA,
            &[(0, 0x01), (1, 0x00)],
            &[0xDE, 0xAD],
        )];
        let ix2 = vec![make_inner_instruction(
            0xBB,
            &[(0, 0x01), (1, 0x00)],
            &[0xDE, 0xAD],
        )];
        let account_keys = make_account_keys(2);

        let hash1 = hash_inner_instructions(&ix1, &account_keys).unwrap();
        let hash2 = hash_inner_instructions(&ix2, &account_keys).unwrap();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_inner_instructions_different_data() {
        let ix1 = vec![make_inner_instruction(0xAA, &[(0, 0x01)], &[0x01])];
        let ix2 = vec![make_inner_instruction(0xAA, &[(0, 0x01)], &[0x02])];
        let account_keys = make_account_keys(1);

        assert_ne!(
            hash_inner_instructions(&ix1, &account_keys).unwrap(),
            hash_inner_instructions(&ix2, &account_keys).unwrap()
        );
    }

    #[test]
    fn test_hash_inner_instructions_different_account_resolution() {
        let ix = vec![make_inner_instruction(
            0xAA,
            &[(0, 0x01), (1, 0x00)],
            &[0x01],
        )];
        let keys_a = make_account_keys(2);
        let mut keys_b = keys_a.clone();
        keys_b.swap(0, 1);

        assert_ne!(
            hash_inner_instructions(&ix, &keys_a).unwrap(),
            hash_inner_instructions(&ix, &keys_b).unwrap()
        );
    }

    #[test]
    fn test_hash_inner_instructions_different_flags() {
        // Same indices but different writable flags must produce different hashes
        let ix1 = vec![make_inner_instruction(0xAA, &[(0, 0x01)], &[0x01])]; // writable
        let ix2 = vec![make_inner_instruction(0xAA, &[(0, 0x00)], &[0x01])]; // readonly
        let account_keys = make_account_keys(1);

        assert_ne!(
            hash_inner_instructions(&ix1, &account_keys).unwrap(),
            hash_inner_instructions(&ix2, &account_keys).unwrap()
        );
    }

    #[test]
    fn test_hash_inner_instructions_empty() {
        let hash = hash_inner_instructions(&[], &[]).unwrap();
        // Should still produce a deterministic hash (keccak of empty)
        let hash2 = hash_inner_instructions(&[], &[]).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_validate_and_hash_caches_program_indices() {
        let program_a = Pubkey::new_unique();
        let program_b = Pubkey::new_unique();
        let account_0 = Pubkey::new_unique();
        let account_1 = Pubkey::new_unique();
        let remaining_accounts = vec![
            make_account_info_with_writable(account_0, true),
            make_account_info(program_a),
            make_account_info(account_1),
            make_account_info(program_b),
        ];
        let inner_instructions = vec![
            InnerInstructionRef::new(
                program_a.to_bytes(),
                &[0, AccountEntry::FLAG_WRITABLE],
                &[1, 2],
            ),
            InnerInstructionRef::new(program_b.to_bytes(), &[2, 0], &[3]),
        ];

        let (entries, inner_hash) = validate_and_hash_inner_instructions(
            &inner_instructions,
            &remaining_accounts,
            &Pubkey::new_unique(),
        )
        .unwrap();

        let indices: Vec<usize> = entries.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(indices, vec![1, 3]);
        // Hash should be deterministic
        let (_, inner_hash2) = validate_and_hash_inner_instructions(
            &inner_instructions,
            &remaining_accounts,
            &Pubkey::new_unique(),
        )
        .unwrap();
        assert_eq!(inner_hash, inner_hash2);

        let expected_hash = hash_inner_instructions(
            &[
                InnerInstruction {
                    program_id: program_a.to_bytes(),
                    accounts: vec![AccountEntry {
                        index: 0,
                        flags: AccountEntry::FLAG_WRITABLE,
                    }],
                    data: vec![1, 2],
                },
                InnerInstruction {
                    program_id: program_b.to_bytes(),
                    accounts: vec![AccountEntry { index: 2, flags: 0 }],
                    data: vec![3],
                },
            ],
            &[account_0, program_a, account_1, program_b],
        )
        .unwrap();
        assert_eq!(inner_hash, expected_hash);
    }

    #[test]
    fn test_validate_and_hash_rejects_self_cpi() {
        let program_id = Pubkey::new_unique();
        let remaining_accounts = vec![make_account_info(Pubkey::new_unique())];
        let inner_instructions = vec![InnerInstructionRef::new(program_id.to_bytes(), &[], &[])];

        let err = validate_and_hash_inner_instructions(
            &inner_instructions,
            &remaining_accounts,
            &program_id,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::CpiToSelfDenied as u32)
        );
    }

    #[test]
    fn test_validate_and_hash_rejects_writable_mismatch() {
        let program_id = Pubkey::new_unique();
        let cpi_program = Pubkey::new_unique();
        let remaining_accounts = vec![
            make_account_info_with_writable(Pubkey::new_unique(), false),
            make_account_info(cpi_program),
        ];
        let inner_instructions = vec![InnerInstructionRef::new(
            cpi_program.to_bytes(),
            &[0, AccountEntry::FLAG_WRITABLE],
            &[1],
        )];

        let err = validate_and_hash_inner_instructions(
            &inner_instructions,
            &remaining_accounts,
            &program_id,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::AccountNotWritable as u32)
        );
    }

    #[test]
    fn test_compute_message_hash_deterministic() {
        let wallet = Pubkey::new_unique();
        let inner_hash = [0x42u8; 32];
        let hash1 = compute_message_hash(&wallet, 100, 0, 500, &inner_hash);
        let hash2 = compute_message_hash(&wallet, 100, 0, 500, &inner_hash);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_message_hash_includes_nonce() {
        let wallet = Pubkey::new_unique();
        let inner_hash = [0x42u8; 32];
        let hash1 = compute_message_hash(&wallet, 100, 0, 500, &inner_hash);
        let hash2 = compute_message_hash(&wallet, 100, 1, 500, &inner_hash);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_message_hash_includes_creation_slot() {
        let wallet = Pubkey::new_unique();
        let inner_hash = [0x42u8; 32];
        let hash1 = compute_message_hash(&wallet, 100, 0, 500, &inner_hash);
        let hash2 = compute_message_hash(&wallet, 200, 0, 500, &inner_hash);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_message_hash_includes_wallet() {
        let wallet1 = Pubkey::new_unique();
        let wallet2 = Pubkey::new_unique();
        let inner_hash = [0x42u8; 32];
        let hash1 = compute_message_hash(&wallet1, 100, 0, 500, &inner_hash);
        let hash2 = compute_message_hash(&wallet2, 100, 0, 500, &inner_hash);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_message_hash_includes_inner_hash() {
        let wallet = Pubkey::new_unique();
        let inner1 = [0x42u8; 32];
        let inner2 = [0x43u8; 32];
        let hash1 = compute_message_hash(&wallet, 100, 0, 500, &inner1);
        let hash2 = compute_message_hash(&wallet, 100, 0, 500, &inner2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_message_hash_includes_max_slot() {
        let wallet = Pubkey::new_unique();
        let inner_hash = [0x42u8; 32];
        let hash1 = compute_message_hash(&wallet, 100, 0, 500, &inner_hash);
        let hash2 = compute_message_hash(&wallet, 100, 0, 600, &inner_hash);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hash_domain_separation() {
        // Verify that swapping bytes between accounts and data produces different hashes
        let ix1 = vec![make_inner_instruction(
            0xAA,
            &[(0, 0x01), (1, 0x00)],
            &[0x02],
        )];
        let ix2 = vec![make_inner_instruction(0xAA, &[(0, 0x01)], &[1, 0x02])];
        let account_keys = make_account_keys(2);
        // With length prefixes + flags these must differ
        assert_ne!(
            hash_inner_instructions(&ix1, &account_keys).unwrap(),
            hash_inner_instructions(&ix2, &account_keys[..1]).unwrap()
        );
    }

    #[test]
    fn test_domain_tag_prevents_cross_message_collision() {
        // Execute and close messages with same parameters should differ due to domain tags
        let wallet = Pubkey::new_unique();
        let inner_hash = [0x42u8; 32];
        let execute_hash = compute_message_hash(&wallet, 100, 0, 500, &inner_hash);

        // Manually compute a close-style hash to verify they differ
        let mut close_msg = Vec::new();
        close_msg.extend_from_slice(b"machine_wallet_close_v0");
        close_msg.extend_from_slice(wallet.as_ref());
        close_msg.extend_from_slice(&100u64.to_le_bytes());
        close_msg.extend_from_slice(&0u64.to_le_bytes());
        close_msg.extend_from_slice(&500u64.to_le_bytes());
        let close_hash = keccak::hash(&close_msg).to_bytes();

        assert_ne!(execute_hash, close_hash);
    }
}
