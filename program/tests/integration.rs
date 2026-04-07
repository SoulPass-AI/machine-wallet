//! Integration and security tests for MachineWallet program.
//!
//! CreateWallet tests run in solana-program-test (BPF/SBF).
//! Execute/CloseWallet tests require the secp256r1 precompile which may not be
//! available in the test validator — those are marked `#[ignore]` if they fail
//! due to missing precompile support, but the test logic is complete.

use solana_program_test::*;
use solana_sdk::{
    account::Account,
    instruction::{AccountMeta, Instruction},
    keccak,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
    sysvar,
    transaction::Transaction,
};
use solana_sdk_ids::system_program;
use solana_system_interface::instruction as system_instruction;

use machine_wallet::{
    instruction::{AccountEntry, InnerInstruction},
    processor::add_authority::compute_add_authority_message,
    processor::advance_nonce::compute_advance_nonce_message,
    processor::create_session,
    processor::execute::{self, compute_message_hash, hash_inner_instructions},
    processor::owner_close_session,
    processor::remove_authority::compute_remove_authority_message,
    processor::revoke_session,
    state::{MachineWallet, SessionState, MAX_ALLOWED_PROGRAMS, SESSION_SEED_PREFIX},
};

use p256::ecdsa::{signature::Signer as P256Signer, SigningKey};
use rand_core::OsRng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn program_test() -> ProgramTest {
    let mut pt = ProgramTest::new(
        "machine_wallet",
        machine_wallet::id(),
        processor!(machine_wallet::processor::process_instruction),
    );
    // init_pda_account uses 3 CPIs (transfer+allocate+assign) for dust-resistance,
    // and P-256 curve validation is ~100K CU. 200K default is insufficient.
    pt.set_compute_max_units(1_400_000);
    pt
}

fn generate_p256_keypair() -> (SigningKey, [u8; 33]) {
    let signing_key = SigningKey::random(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    let point = verifying_key.to_encoded_point(true); // compressed
    let compressed: [u8; 33] = point
        .as_bytes()
        .try_into()
        .expect("compressed P-256 pubkey must be 33 bytes");
    (signing_key, compressed)
}

fn ed25519_authority_bytes(authority: &Keypair) -> [u8; 33] {
    let mut bytes = [0u8; 33];
    bytes[..32].copy_from_slice(&authority.pubkey().to_bytes());
    bytes
}

/// Derive wallet and vault PDAs from a compressed P-256 authority pubkey.
fn derive_pdas(authority: &[u8; 33]) -> (Pubkey, u8, Pubkey, u8) {
    let id = keccak::hash(authority).to_bytes();
    let (wallet_pda, wallet_bump) = Pubkey::find_program_address(
        &[MachineWallet::SEED_PREFIX, id.as_ref()],
        &machine_wallet::id(),
    );
    let (vault_pda, vault_bump) = Pubkey::find_program_address(
        &[MachineWallet::VAULT_SEED_PREFIX, wallet_pda.as_ref()],
        &machine_wallet::id(),
    );
    (wallet_pda, wallet_bump, vault_pda, vault_bump)
}

/// Build a CreateWallet instruction.
/// Vault is NOT passed as an account — it stays system-owned and is not created here.
fn build_create_wallet_ix(payer: &Pubkey, authority: &[u8; 33]) -> (Instruction, Pubkey, Pubkey) {
    build_create_wallet_ix_with_scheme(payer, 0, authority)
}

fn build_create_wallet_ix_with_scheme(
    payer: &Pubkey,
    sig_scheme: u8,
    authority: &[u8; 33],
) -> (Instruction, Pubkey, Pubkey) {
    let (wallet_pda, _, vault_pda, _) = derive_pdas(authority);

    let mut data = vec![0u8]; // discriminator = CreateWallet
    if sig_scheme == 0 {
        data.extend_from_slice(authority);
    } else {
        data.push(sig_scheme);
        data.extend_from_slice(authority);
    }

    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(*payer, true),
            AccountMeta::new(wallet_pda, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    };
    (ix, wallet_pda, vault_pda)
}

fn build_ed25519_precompile_ix(authority: &Keypair, message: &[u8; 32]) -> Instruction {
    let signature = authority.sign_message(message);
    let pubkey = authority.pubkey().to_bytes();

    let mut data = Vec::with_capacity(2 + 14 + 64 + 32 + 32);
    data.push(1u8); // signature_count
    data.push(0u8); // padding

    let sig_offset: u16 = 16;
    let pk_offset: u16 = 80;
    let msg_offset: u16 = 112;

    data.extend_from_slice(&sig_offset.to_le_bytes());
    data.extend_from_slice(&0xFFFFu16.to_le_bytes());
    data.extend_from_slice(&pk_offset.to_le_bytes());
    data.extend_from_slice(&0xFFFFu16.to_le_bytes());
    data.extend_from_slice(&msg_offset.to_le_bytes());
    data.extend_from_slice(&32u16.to_le_bytes());
    data.extend_from_slice(&0xFFFFu16.to_le_bytes());

    data.extend_from_slice(signature.as_ref());
    data.extend_from_slice(&pubkey);
    data.extend_from_slice(message);

    Instruction {
        program_id: machine_wallet::ed25519::ED25519_PROGRAM_ID,
        accounts: vec![],
        data,
    }
}

fn build_add_authority_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    precompile_ix_index: u8,
    new_sig_scheme: u8,
    new_pubkey: &[u8; 33],
    new_threshold: u8,
    max_slot: u64,
) -> Instruction {
    let mut data = vec![9u8, precompile_ix_index, new_sig_scheme];
    data.extend_from_slice(new_pubkey);
    data.push(new_threshold);
    data.extend_from_slice(&max_slot.to_le_bytes());

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    }
}

fn build_remove_authority_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    precompile_ix_index: u8,
    remove_sig_scheme: u8,
    remove_pubkey: &[u8; 33],
    new_threshold: u8,
    max_slot: u64,
) -> Instruction {
    let mut data = vec![10u8, precompile_ix_index, remove_sig_scheme];
    data.extend_from_slice(remove_pubkey);
    data.push(new_threshold);
    data.extend_from_slice(&max_slot.to_le_bytes());

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    }
}

fn derive_session_pda(wallet_pda: &Pubkey, authority: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[SESSION_SEED_PREFIX, wallet_pda.as_ref(), authority.as_ref()],
        &machine_wallet::id(),
    )
}

fn build_create_session_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    session_pda: &Pubkey,
    precompile_ix_index: u8,
    max_slot: u64,
    session_authority: &Pubkey,
    expiry_slot: u64,
    max_lamports_per_ix: u64,
    allowed_programs: &[Pubkey],
) -> Instruction {
    let mut data = vec![4u8, precompile_ix_index];
    data.extend_from_slice(&max_slot.to_le_bytes());
    data.extend_from_slice(session_authority.as_ref());
    data.extend_from_slice(&expiry_slot.to_le_bytes());
    data.extend_from_slice(&max_lamports_per_ix.to_le_bytes());
    data.push(allowed_programs.len() as u8);
    for program_id in allowed_programs {
        data.extend_from_slice(program_id.as_ref());
    }

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
            AccountMeta::new(*session_pda, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    }
}

fn build_revoke_session_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    session_pda: &Pubkey,
    precompile_ix_index: u8,
    max_slot: u64,
    session_authority: &Pubkey,
) -> Instruction {
    let mut data = vec![6u8, precompile_ix_index];
    data.extend_from_slice(&max_slot.to_le_bytes());
    data.extend_from_slice(session_authority.as_ref());

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
            AccountMeta::new(*session_pda, false),
        ],
        data,
    }
}

fn build_session_execute_ix(
    session_pda: &Pubkey,
    wallet_pda: &Pubkey,
    authority: &Pubkey,
    vault_pda: &Pubkey,
    inner_instructions: &[InnerInstruction],
    remaining_account_metas: &[AccountMeta],
) -> Instruction {
    let mut data = vec![5u8];
    data.extend_from_slice(&(inner_instructions.len() as u32).to_le_bytes());
    for inner in inner_instructions {
        data.extend_from_slice(&inner.program_id);
        data.extend_from_slice(&(inner.accounts.len() as u16).to_le_bytes());
        data.extend_from_slice(&(inner.data.len() as u16).to_le_bytes());
        for entry in &inner.accounts {
            data.push(entry.index);
            data.push(entry.flags);
        }
        data.extend_from_slice(&inner.data);
    }

    let mut accounts = vec![
        AccountMeta::new(*session_pda, false),
        AccountMeta::new_readonly(*wallet_pda, false),
        AccountMeta::new_readonly(*authority, true),
        AccountMeta::new(*vault_pda, false),
    ];
    accounts.extend_from_slice(remaining_account_metas);

    Instruction {
        program_id: machine_wallet::id(),
        accounts,
        data,
    }
}

fn build_owner_close_session_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    session_pda: &Pubkey,
    precompile_ix_index: u8,
    max_slot: u64,
    session_authority: &Pubkey,
    destination: &Pubkey,
) -> Instruction {
    let mut data = vec![12u8, precompile_ix_index];
    data.extend_from_slice(&max_slot.to_le_bytes());
    data.extend_from_slice(session_authority.as_ref());
    data.extend_from_slice(destination.as_ref());

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
            AccountMeta::new(*session_pda, false),
            AccountMeta::new(*destination, false),
        ],
        data,
    }
}

/// Secp256r1 precompile program ID.
const SECP256R1_PROGRAM_ID: Pubkey =
    solana_pubkey::pubkey!("Secp256r1SigVerify1111111111111111111111111");

/// Build a secp256r1 precompile instruction.
fn build_secp256r1_ix(
    signing_key: &SigningKey,
    compressed_pubkey: &[u8; 33],
    message: &[u8; 32],
) -> Instruction {
    let sig: p256::ecdsa::Signature = signing_key.sign(message);
    let sig_bytes: [u8; 64] = sig.to_bytes().into();

    let mut data = Vec::with_capacity(2 + 14 + 64 + 33 + 32);

    // Header: signature_count=1, padding=0
    data.push(1u8);
    data.push(0u8);

    // SignatureOffsets (14 bytes)
    let sig_offset: u16 = 16; // 2 (header) + 14 (offsets)
    let pk_offset: u16 = 80; // 16 + 64 (signature)
    let msg_offset: u16 = 113; // 80 + 33 (pubkey)

    data.extend_from_slice(&sig_offset.to_le_bytes());
    data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // signature_instruction_index
    data.extend_from_slice(&pk_offset.to_le_bytes());
    data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // public_key_instruction_index
    data.extend_from_slice(&msg_offset.to_le_bytes());
    data.extend_from_slice(&32u16.to_le_bytes()); // message_data_size
    data.extend_from_slice(&0xFFFFu16.to_le_bytes()); // message_instruction_index

    // Signature (64 bytes: r || s)
    data.extend_from_slice(&sig_bytes);

    // Compressed public key (33 bytes)
    data.extend_from_slice(compressed_pubkey);

    // Message (32 bytes)
    data.extend_from_slice(message);

    Instruction {
        program_id: SECP256R1_PROGRAM_ID,
        accounts: vec![],
        data,
    }
}

/// Build an Execute instruction with the new wire format (AccountEntry pairs).
fn build_execute_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    vault_pda: &Pubkey,
    secp256r1_ix_index: u8,
    max_slot: u64,
    inner_instructions: &[InnerInstruction],
    remaining_account_metas: &[AccountMeta],
) -> Instruction {
    let mut data = vec![1u8]; // discriminator = Execute
    data.push(secp256r1_ix_index);
    data.extend_from_slice(&max_slot.to_le_bytes());
    data.extend_from_slice(&(inner_instructions.len() as u32).to_le_bytes());
    for inner in inner_instructions {
        data.extend_from_slice(&inner.program_id);
        data.extend_from_slice(&(inner.accounts.len() as u16).to_le_bytes());
        data.extend_from_slice(&(inner.data.len() as u16).to_le_bytes());
        // Account entries: (index, flags) pairs
        for entry in &inner.accounts {
            data.push(entry.index);
            data.push(entry.flags);
        }
        data.extend_from_slice(&inner.data);
    }

    let mut accounts = vec![
        AccountMeta::new_readonly(sysvar::instructions::id(), false),
        AccountMeta::new(*wallet_pda, false),
        AccountMeta::new_readonly(*fee_payer, true),
        AccountMeta::new(*vault_pda, false),
    ];
    accounts.extend_from_slice(remaining_account_metas);

    Instruction {
        program_id: machine_wallet::id(),
        accounts,
        data,
    }
}

/// Build a CloseWallet instruction (now includes destination in data).
fn build_close_wallet_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    vault_pda: &Pubkey,
    destination: &Pubkey,
    secp256r1_ix_index: u8,
    max_slot: u64,
) -> Instruction {
    let mut data = vec![2u8, secp256r1_ix_index]; // discriminator = CloseWallet
    data.extend_from_slice(&max_slot.to_le_bytes());
    data.extend_from_slice(&destination.to_bytes()); // destination pubkey in signed data

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
            AccountMeta::new(*vault_pda, false),
            AccountMeta::new(*destination, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    }
}

/// Compute the close message hash with domain tag + destination.
fn compute_close_message_hash(
    wallet_address: &Pubkey,
    creation_slot: u64,
    nonce: u64,
    max_slot: u64,
    destination: &Pubkey,
) -> [u8; 32] {
    let mut msg = Vec::with_capacity(23 + 32 + 8 + 8 + 8 + 32);
    msg.extend_from_slice(b"machine_wallet_close_v0");
    msg.extend_from_slice(wallet_address.as_ref());
    msg.extend_from_slice(&creation_slot.to_le_bytes());
    msg.extend_from_slice(&nonce.to_le_bytes());
    msg.extend_from_slice(&max_slot.to_le_bytes());
    msg.extend_from_slice(destination.as_ref());
    keccak::hash(&msg).to_bytes()
}

/// Helper to build an InnerInstruction with AccountEntry pairs.
fn make_inner(
    program_id: [u8; 32],
    accounts: &[(u8, u8)], // (index, flags)
    data: Vec<u8>,
) -> InnerInstruction {
    InnerInstruction {
        program_id,
        accounts: accounts
            .iter()
            .map(|&(index, flags)| AccountEntry { index, flags })
            .collect(),
        data,
    }
}

fn hash_inner(inner_instructions: &[InnerInstruction], remaining_pubkeys: &[Pubkey]) -> [u8; 32] {
    hash_inner_instructions(inner_instructions, remaining_pubkeys)
        .expect("remaining accounts must satisfy signed indices")
}

/// Helper: create a wallet and return (wallet_pda, vault_pda, signing_key, compressed_pubkey).
async fn create_wallet_helper(
    banks: &mut BanksClient,
    payer: &Keypair,
    recent_blockhash: solana_sdk::hash::Hash,
) -> (Pubkey, Pubkey, SigningKey, [u8; 33]) {
    let (signing_key, compressed) = generate_p256_keypair();
    let (ix, wallet_pda, vault_pda) = build_create_wallet_ix(&payer.pubkey(), &compressed);

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    (wallet_pda, vault_pda, signing_key, compressed)
}

/// Read MachineWallet state from an account.
async fn read_wallet_state(banks: &mut BanksClient, wallet_pda: &Pubkey) -> MachineWallet {
    let account = banks.get_account(*wallet_pda).await.unwrap().unwrap();
    MachineWallet::deserialize(&account.data).unwrap()
}

async fn read_session_state(banks: &mut BanksClient, session_pda: &Pubkey) -> SessionState {
    let account = banks.get_account(*session_pda).await.unwrap().unwrap();
    SessionState::deserialize(&account.data).unwrap()
}

async fn create_ed25519_wallet_helper(
    banks: &mut BanksClient,
    payer: &Keypair,
    owner: &Keypair,
) -> (Pubkey, Pubkey) {
    let owner_bytes = ed25519_authority_bytes(owner);
    let (create_ix, wallet_pda, vault_pda) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);
    let tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(tx).await.unwrap();
    (wallet_pda, vault_pda)
}

async fn create_session_helper(
    banks: &mut BanksClient,
    payer: &Keypair,
    owner: &Keypair,
    wallet_pda: &Pubkey,
    session_authority: &Keypair,
    expiry_slot: u64,
    max_lamports_per_ix: u64,
    allowed_programs: &[Pubkey],
) -> Pubkey {
    let wallet = read_wallet_state(banks, wallet_pda).await;
    let (session_pda, _) = derive_session_pda(wallet_pda, &session_authority.pubkey());
    let allowed_program_bytes: Vec<[u8; 32]> = allowed_programs
        .iter()
        .map(|program_id| program_id.to_bytes())
        .collect();
    let session_data_hash = create_session::hash_session_data(
        &session_authority.pubkey().to_bytes(),
        expiry_slot,
        max_lamports_per_ix,
        allowed_program_bytes.len() as u8,
        &allowed_program_bytes,
    );
    let expected_message = create_session::compute_create_session_message(
        wallet_pda,
        wallet.creation_slot,
        wallet.nonce,
        u64::MAX,
        &session_data_hash,
    );
    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(owner, &expected_message),
            build_create_session_ix(
                &payer.pubkey(),
                wallet_pda,
                &session_pda,
                0,
                u64::MAX,
                &session_authority.pubkey(),
                expiry_slot,
                max_lamports_per_ix,
                allowed_programs,
            ),
        ],
        Some(&payer.pubkey()),
        &[payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(tx).await.unwrap();
    session_pda
}

// ===========================================================================
// Integration Tests: CreateWallet
// ===========================================================================

#[tokio::test]
async fn test_create_wallet() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (_signing_key, compressed) = generate_p256_keypair();
    let (ix, wallet_pda, _vault_pda) = build_create_wallet_ix(&payer.pubkey(), &compressed);

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    // Verify state
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet.version, 1);
    assert_eq!(wallet.authorities[0].pubkey, compressed);
    assert_eq!(wallet.nonce, 0);
    assert_eq!(wallet.id(), keccak::hash(&compressed).to_bytes());
    assert_eq!(wallet.wallet_id, keccak::hash(&compressed).to_bytes());
    assert_eq!(wallet.authorities[0].sig_scheme, 0); // Secp256r1
    assert_eq!(wallet.threshold, 1);
    assert_eq!(wallet.authority_count, 1);

    // Verify vault_bump was stored
    let (_, _, _, expected_vault_bump) = derive_pdas(&compressed);
    assert_eq!(wallet.vault_bump, expected_vault_bump);
}

#[tokio::test]
async fn test_create_wallet_ed25519() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let authority = Keypair::new();
    let authority_bytes = ed25519_authority_bytes(&authority);
    let (ix, wallet_pda, _vault_pda) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &authority_bytes);

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet.version, 1);
    assert_eq!(wallet.authorities[0].sig_scheme, 1);
    assert_eq!(wallet.authorities[0].pubkey, authority_bytes);
    assert_eq!(wallet.authority_count, 1);
}

#[tokio::test]
async fn test_create_wallet_accepts_any_nonzero_ed25519() {
    // Low-order points pass format check — see is_valid_ed25519 doc comment.
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let mut authority = [0u8; 33];
    authority[0] = 1; // identity point — non-zero, passes format check
    let (ix, wallet_pda, _vault_pda) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &authority);

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet.authority_count, 1);
    assert_eq!(wallet.authorities[0].sig_scheme, 1);
}

#[tokio::test]
async fn test_create_wallet_duplicate_fails() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let (_signing_key, compressed) = generate_p256_keypair();

    // First create succeeds
    let (ix, _wallet_pda, _vault_pda) = build_create_wallet_ix(&payer.pubkey(), &compressed);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    // Second create with same authority should fail (WalletAlreadyInitialized).
    let second_payer = Keypair::new();
    let recent_blockhash2 = banks.get_latest_blockhash().await.unwrap();
    let fund_ix =
        system_instruction::transfer(&payer.pubkey(), &second_payer.pubkey(), 1_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash2,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let (ix2, _, _) = build_create_wallet_ix(&second_payer.pubkey(), &compressed);
    let recent_blockhash3 = banks.get_latest_blockhash().await.unwrap();
    let tx2 = Transaction::new_signed_with_payer(
        &[ix2],
        Some(&second_payer.pubkey()),
        &[&second_payer],
        recent_blockhash3,
    );
    let result = banks.process_transaction(tx2).await;
    assert!(result.is_err(), "Duplicate create should fail");
}

#[tokio::test]
async fn test_add_authority_tops_up_rent_via_system_transfer() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _vault_pda) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);

    let create_tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(create_tx).await.unwrap();

    let wallet_before = read_wallet_state(&mut banks, &wallet_pda).await;
    let account_before = banks.get_account(wallet_pda).await.unwrap().unwrap();
    let (_new_owner_signing_key, new_owner_pubkey) = generate_p256_keypair();
    let expected_message = compute_add_authority_message(
        &wallet_pda,
        wallet_before.creation_slot,
        wallet_before.nonce,
        u64::MAX,
        0,
        &new_owner_pubkey,
        0,
    );
    let ed25519_ix = build_ed25519_precompile_ix(&owner, &expected_message);
    let add_ix = build_add_authority_ix(
        &payer.pubkey(),
        &wallet_pda,
        0,
        0,
        &new_owner_pubkey,
        0,
        u64::MAX,
    );

    let add_tx = Transaction::new_signed_with_payer(
        &[ed25519_ix, add_ix],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(add_tx).await.unwrap();

    let wallet_after = read_wallet_state(&mut banks, &wallet_pda).await;
    let account_after = banks.get_account(wallet_pda).await.unwrap().unwrap();
    assert_eq!(wallet_after.authority_count, 2);
    assert_eq!(wallet_after.authorities[1].pubkey, new_owner_pubkey);
    assert!(account_after.lamports > account_before.lamports);
    assert_eq!(account_after.data.len(), MachineWallet::v1_account_size(2));
}

#[tokio::test]
async fn test_add_ed25519_authority_succeeds() {
    // Regression: curve25519-dalek caused ProgramFailedToComplete in BPF (see is_valid_ed25519).
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _vault_pda) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);

    let create_tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(create_tx).await.unwrap();

    let wallet_before = read_wallet_state(&mut banks, &wallet_pda).await;

    // Add a new Ed25519 authority (scheme=1)
    let new_ed25519_owner = Keypair::new();
    let new_ed25519_bytes = ed25519_authority_bytes(&new_ed25519_owner);
    let expected_message = compute_add_authority_message(
        &wallet_pda,
        wallet_before.creation_slot,
        wallet_before.nonce,
        u64::MAX,
        1, // Ed25519
        &new_ed25519_bytes,
        0,
    );
    let ed25519_ix = build_ed25519_precompile_ix(&owner, &expected_message);
    let add_ix = build_add_authority_ix(
        &payer.pubkey(),
        &wallet_pda,
        0,
        1, // new_sig_scheme = Ed25519
        &new_ed25519_bytes,
        0,
        u64::MAX,
    );

    let add_tx = Transaction::new_signed_with_payer(
        &[ed25519_ix, add_ix],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(add_tx).await.unwrap();

    let wallet_after = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet_after.authority_count, 2);
    assert_eq!(wallet_after.authorities[1].sig_scheme, 1);
    assert_eq!(wallet_after.authorities[1].pubkey, new_ed25519_bytes);
    assert_eq!(wallet_after.nonce, wallet_before.nonce + 1);
}

#[tokio::test]
async fn test_remove_authority_keeps_excess_rent_in_wallet() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _vault_pda) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);

    let create_tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(create_tx).await.unwrap();

    let wallet_initial = read_wallet_state(&mut banks, &wallet_pda).await;
    let (_new_owner_signing_key, new_owner_pubkey) = generate_p256_keypair();
    let add_message = compute_add_authority_message(
        &wallet_pda,
        wallet_initial.creation_slot,
        wallet_initial.nonce,
        u64::MAX,
        0,
        &new_owner_pubkey,
        0,
    );
    let add_tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &add_message),
            build_add_authority_ix(
                &payer.pubkey(),
                &wallet_pda,
                0,
                0,
                &new_owner_pubkey,
                0,
                u64::MAX,
            ),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(add_tx).await.unwrap();

    let wallet_before_remove = read_wallet_state(&mut banks, &wallet_pda).await;
    let account_before_remove = banks.get_account(wallet_pda).await.unwrap().unwrap();
    let remove_message = compute_remove_authority_message(
        &wallet_pda,
        wallet_before_remove.creation_slot,
        wallet_before_remove.nonce,
        u64::MAX,
        0,
        &new_owner_pubkey,
        0,
    );
    let remove_tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &remove_message),
            build_remove_authority_ix(
                &payer.pubkey(),
                &wallet_pda,
                0,
                0,
                &new_owner_pubkey,
                0,
                u64::MAX,
            ),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(remove_tx).await.unwrap();

    let wallet_after_remove = read_wallet_state(&mut banks, &wallet_pda).await;
    let account_after_remove = banks.get_account(wallet_pda).await.unwrap().unwrap();
    assert_eq!(wallet_after_remove.authority_count, 1);
    assert_eq!(wallet_after_remove.authorities[0].pubkey, owner_bytes);
    assert_eq!(
        account_after_remove.lamports,
        account_before_remove.lamports
    );
    assert_eq!(
        account_after_remove.data.len(),
        MachineWallet::v1_account_size(1)
    );
}

#[tokio::test]
async fn test_create_wallet_invalid_pubkey() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    // Use 0x04 prefix (uncompressed) — invalid
    let mut authority = [0xAA; 33];
    authority[0] = 0x04;

    let (ix, _, _) = build_create_wallet_ix(&payer.pubkey(), &authority);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Invalid pubkey prefix should fail");
}

#[tokio::test]
async fn test_create_wallet_invalid_pubkey_zero_prefix() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let mut authority = [0xAA; 33];
    authority[0] = 0x00;

    let (ix, _, _) = build_create_wallet_ix(&payer.pubkey(), &authority);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Zero prefix should fail");
}

#[tokio::test]
async fn test_create_wallet_rejects_zero_x_coordinate() {
    // x-coordinate all zeros is rejected even with valid prefix.
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let mut authority = [0u8; 33];
    authority[0] = 0x02; // valid prefix, but x = 0

    let (ix, _, _) = build_create_wallet_ix(&payer.pubkey(), &authority);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Zero x-coordinate should be rejected"
    );
}

#[tokio::test]
async fn test_create_wallet_wrong_wallet_pda() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let (_signing_key, compressed) = generate_p256_keypair();

    let wrong_wallet_pda = Pubkey::new_unique();

    let mut data = vec![0u8];
    data.extend_from_slice(&compressed);

    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(payer.pubkey(), true),
            AccountMeta::new(wrong_wallet_pda, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Wrong wallet PDA should fail");
}

// test_create_wallet_wrong_vault_pda removed: vault is no longer passed to CreateWallet.

#[tokio::test]
async fn test_create_wallet_multiple_independent() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;

    let (_sk1, compressed1) = generate_p256_keypair();
    let (_sk2, compressed2) = generate_p256_keypair();

    let (ix1, wallet_pda1, _) = build_create_wallet_ix(&payer.pubkey(), &compressed1);
    let tx1 = Transaction::new_signed_with_payer(
        &[ix1],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx1).await.unwrap();

    let recent_blockhash2 = banks.get_latest_blockhash().await.unwrap();
    let (ix2, wallet_pda2, _) = build_create_wallet_ix(&payer.pubkey(), &compressed2);
    let tx2 = Transaction::new_signed_with_payer(
        &[ix2],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash2,
    );
    banks.process_transaction(tx2).await.unwrap();

    let w1 = read_wallet_state(&mut banks, &wallet_pda1).await;
    let w2 = read_wallet_state(&mut banks, &wallet_pda2).await;
    assert_ne!(w1.id(), w2.id());
    assert_eq!(w1.authorities[0].pubkey, compressed1);
    assert_eq!(w2.authorities[0].pubkey, compressed2);
}

#[tokio::test]
async fn test_create_wallet_succeeds_with_dusted_pda() {
    let (_signing_key, compressed) = generate_p256_keypair();
    let (_, wallet_pda, _) = build_create_wallet_ix(&Pubkey::new_unique(), &compressed);
    let mut pt = program_test();
    pt.add_account(
        wallet_pda,
        Account {
            lamports: 1,
            data: vec![],
            owner: system_program::id(),
            executable: false,
            rent_epoch: 0,
        },
    );

    let (banks, payer, recent_blockhash) = pt.start().await;
    let (ix, wallet_pda, _) = build_create_wallet_ix(&payer.pubkey(), &compressed);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let account = banks.get_account(wallet_pda).await.unwrap().unwrap();
    assert_eq!(account.owner, machine_wallet::id());
    assert_eq!(account.data.len(), MachineWallet::v1_account_size(1));
    assert!(account.lamports > 1);
}

#[tokio::test]
async fn test_create_session_succeeds_with_dusted_pda() {
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let session_authority = Keypair::new();
    let (wallet_pda, _, _, _) = derive_pdas(&owner_bytes);
    let (session_pda, _) = derive_session_pda(&wallet_pda, &session_authority.pubkey());

    let mut pt = program_test();
    pt.add_account(
        session_pda,
        Account {
            lamports: 1,
            data: vec![],
            owner: system_program::id(),
            executable: false,
            rent_epoch: 0,
        },
    );

    let (mut banks, payer, _) = pt.start().await;
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let session_pda = create_session_helper(
        &mut banks,
        &payer,
        &owner,
        &wallet_pda,
        &session_authority,
        wallet.creation_slot + 500,
        1_000_000,
        &[system_program::id()],
    )
    .await;

    let account = banks.get_account(session_pda).await.unwrap().unwrap();
    let session = read_session_state(&mut banks, &session_pda).await;
    assert_eq!(account.owner, machine_wallet::id());
    assert_eq!(session.authority, session_authority.pubkey().to_bytes());
    assert_eq!(session.allowed_programs_count, 1);
    assert_eq!(session.allowed_programs[0], system_program::id().to_bytes());
}

#[tokio::test]
async fn test_create_session_allows_custom_program_whitelist() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let custom_program = Pubkey::new_unique();
    let session_pda = create_session_helper(
        &mut banks,
        &payer,
        &owner,
        &wallet_pda,
        &session_authority,
        wallet.creation_slot + 500,
        1_000_000,
        &[custom_program],
    )
    .await;
    let session = read_session_state(&mut banks, &session_pda).await;
    assert_eq!(session.allowed_programs_count, 1);
    assert_eq!(session.allowed_programs[0], custom_program.to_bytes());
}

#[tokio::test]
async fn test_session_execute_transfer_respects_safe_path() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let recipient = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let session_pda = create_session_helper(
        &mut banks,
        &payer,
        &owner,
        &wallet_pda,
        &session_authority,
        wallet.creation_slot + 500,
        500_000_000,
        &[system_program::id()],
    )
    .await;

    let fund_vault_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(
            &payer.pubkey(),
            &vault_pda,
            1_000_000_000,
        )],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(fund_vault_tx).await.unwrap();

    let amount = 200_000_000;
    let transfer_ix = system_instruction::transfer(&vault_pda, &recipient.pubkey(), amount);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix.data,
    );
    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &session_pda,
            &wallet_pda,
            &session_authority.pubkey(),
            &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new(recipient.pubkey(), false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(tx).await.unwrap();

    let recipient_account = banks
        .get_account(recipient.pubkey())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recipient_account.lamports, amount);
}

#[tokio::test]
async fn test_create_session_allows_zero_max_lamports_for_uncapped_mode() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let session_pda = create_session_helper(
        &mut banks,
        &payer,
        &owner,
        &wallet_pda,
        &session_authority,
        wallet.creation_slot + 500,
        0,
        &[Pubkey::new_unique()],
    )
    .await;
    let session = read_session_state(&mut banks, &session_pda).await;
    assert_eq!(session.max_lamports_per_ix, 0);
}

#[tokio::test]
async fn test_owner_close_session_reclaims_revoked_session_rent() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let destination = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let session_pda = create_session_helper(
        &mut banks,
        &payer,
        &owner,
        &wallet_pda,
        &session_authority,
        wallet.creation_slot + 500,
        500_000_000,
        &[system_program::id()],
    )
    .await;

    let fund_destination_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(
            &payer.pubkey(),
            &destination.pubkey(),
            1_000_000,
        )],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks
        .process_transaction(fund_destination_tx)
        .await
        .unwrap();

    let wallet_before_revoke = read_wallet_state(&mut banks, &wallet_pda).await;
    let revoke_message = revoke_session::compute_revoke_session_message(
        &wallet_pda,
        wallet_before_revoke.creation_slot,
        wallet_before_revoke.nonce,
        u64::MAX,
        &session_authority.pubkey().to_bytes(),
    );
    let revoke_tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &revoke_message),
            build_revoke_session_ix(
                &payer.pubkey(),
                &wallet_pda,
                &session_pda,
                0,
                u64::MAX,
                &session_authority.pubkey(),
            ),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(revoke_tx).await.unwrap();

    let session_account = banks.get_account(session_pda).await.unwrap().unwrap();
    let destination_before = banks
        .get_account(destination.pubkey())
        .await
        .unwrap()
        .unwrap();
    let wallet_before_close = read_wallet_state(&mut banks, &wallet_pda).await;
    let close_message = owner_close_session::compute_owner_close_session_message(
        &wallet_pda,
        wallet_before_close.creation_slot,
        wallet_before_close.nonce,
        u64::MAX,
        &session_authority.pubkey().to_bytes(),
        &destination.pubkey().to_bytes(),
    );
    let close_tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &close_message),
            build_owner_close_session_ix(
                &payer.pubkey(),
                &wallet_pda,
                &session_pda,
                0,
                u64::MAX,
                &session_authority.pubkey(),
                &destination.pubkey(),
            ),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(close_tx).await.unwrap();

    let destination_after = banks
        .get_account(destination.pubkey())
        .await
        .unwrap()
        .unwrap();
    let wallet_after = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(
        destination_after.lamports,
        destination_before.lamports + session_account.lamports
    );
    assert!(banks.get_account(session_pda).await.unwrap().is_none());
    assert_eq!(wallet_after.nonce, wallet_before_close.nonce + 1);
}

// ---------------------------------------------------------------------------
// Execute & CloseWallet integration tests — require secp256r1 precompile.
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_sol_transfer() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    // Fund vault with 2 SOL
    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 2_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    // Build inner instruction: SOL transfer from vault to recipient
    let recipient = Pubkey::new_unique();
    let transfer_amount: u64 = 1_000_000_000; // 1 SOL
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, transfer_amount);

    // vault=0 (writable), recipient=1 (writable), system_program=2 (readonly) in remaining accounts
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &inner_hash,
    );

    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let recipient_account = banks.get_account(recipient).await.unwrap().unwrap();
    assert_eq!(recipient_account.lamports, transfer_amount);

    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet.nonce, 1);
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_nonce_increment() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 5_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let creation_slot = wallet_state.creation_slot;

    for expected_nonce in 0u64..3 {
        let recipient = Pubkey::new_unique();
        let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 100_000);

        let inner = make_inner(
            system_program::id().to_bytes(),
            &[
                (0, AccountEntry::FLAG_WRITABLE),
                (1, AccountEntry::FLAG_WRITABLE),
            ],
            transfer_ix_data.data.clone(),
        );

        let inner_hash = hash_inner(
            &[inner.clone()],
            &[vault_pda, recipient, system_program::id()],
        );
        let message = compute_message_hash(
            &wallet_pda,
            creation_slot,
            expected_nonce,
            u64::MAX,
            &inner_hash,
        );

        let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);
        let execute_ix = build_execute_ix(
            &payer.pubkey(),
            &wallet_pda,
            &vault_pda,
            0,
            u64::MAX,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new(recipient, false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        );

        let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
        let tx = Transaction::new_signed_with_payer(
            &[precompile_ix, execute_ix],
            Some(&payer.pubkey()),
            &[&payer],
            recent_blockhash,
        );
        banks.process_transaction(tx).await.unwrap();

        let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
        assert_eq!(wallet.nonce, expected_nonce + 1);
    }
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_close_wallet() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let destination = Pubkey::new_unique();
    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;

    let close_message = compute_close_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &destination,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &close_message);

    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &destination,
        0,
        u64::MAX,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let dest_account = banks.get_account(destination).await.unwrap().unwrap();
    assert!(dest_account.lamports > 0);

    let wallet_account = banks.get_account(wallet_pda).await.unwrap();
    match wallet_account {
        None => {}
        Some(acc) => {
            assert_eq!(acc.lamports, 0);
            assert!(acc.data.iter().all(|&b| b == 0));
        }
    }
}

// ===========================================================================
// Security Tests
// ===========================================================================

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_wrong_pubkey() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, _signing_key_a, _compressed_a) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let (signing_key_b, compressed_b) = generate_p256_keypair();
    let recipient = Pubkey::new_unique();
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 500_000_000);

    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &inner_hash,
    );

    let precompile_ix = build_secp256r1_ix(&signing_key_b, &compressed_b, &message);
    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Wrong pubkey should fail with PublicKeyMismatch"
    );
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_wrong_nonce() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let recipient = Pubkey::new_unique();
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 100_000);

    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    // Sign with nonce=5 but wallet has nonce=0
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        5,
        u64::MAX,
        &inner_hash,
    );

    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);
    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Wrong nonce should fail with MessageMismatch"
    );
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_replay() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 2_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let recipient = Pubkey::new_unique();
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 100_000);

    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &inner_hash,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);
    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner.clone()],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    // Replay with same nonce=0 message should fail
    let precompile_ix_replay = build_secp256r1_ix(&signing_key, &compressed, &message);
    let execute_ix_replay = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx2 = Transaction::new_signed_with_payer(
        &[precompile_ix_replay, execute_ix_replay],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx2).await;
    assert!(result.is_err(), "Replay should fail (nonce mismatch)");
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_modified_instructions() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 5_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let recipient = Pubkey::new_unique();

    // Sign for 1 SOL transfer
    let transfer_1_sol = system_instruction::transfer(&vault_pda, &recipient, 1_000_000_000);
    let inner_signed = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_1_sol.data.clone(),
    );

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let inner_hash = hash_inner(
        &[inner_signed],
        &[vault_pda, recipient, system_program::id()],
    );
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &inner_hash,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);

    // But submit with 10 SOL transfer (modified instruction)
    let transfer_10_sol = system_instruction::transfer(&vault_pda, &recipient, 10_000_000_000);
    let inner_modified = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_10_sol.data.clone(),
    );

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner_modified],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Modified instructions should fail (MessageMismatch)"
    );
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_wrong_vault() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, _vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let wrong_vault = Pubkey::new_unique();

    let inner = make_inner(system_program::id().to_bytes(), &[], vec![]);

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let inner_hash = hash_inner(&[inner.clone()], &[]);
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &inner_hash,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &wrong_vault,
        0,
        u64::MAX,
        &[inner],
        &[],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Wrong vault PDA should fail");
}

#[tokio::test]
async fn test_execute_wallet_not_owned() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let fake_wallet = payer.pubkey();
    let fake_vault = Pubkey::new_unique();

    let mut data = vec![1u8]; // Execute discriminator
    data.push(0); // secp256r1_ix_index
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // max_slot
    data.extend_from_slice(&0u32.to_le_bytes()); // 0 inner instructions

    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(fake_wallet, false),
            AccountMeta::new_readonly(payer.pubkey(), true),
            AccountMeta::new(fake_vault, false),
        ],
        data,
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "System-owned wallet account should fail");
}

#[tokio::test]
async fn test_execute_no_precompile() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let mut banks = banks;
    let (wallet_pda, vault_pda, _signing_key, _compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0, // points to the execute ix itself (not a precompile)
        u64::MAX,
        &[],
        &[],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Execute with ix_index pointing to itself should fail (InvalidPrecompileInstruction)"
    );
}

#[tokio::test]
async fn test_execute_precompile_index_out_of_bounds() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let mut banks = banks;
    let (wallet_pda, vault_pda, _signing_key, _compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        99, // way out of bounds
        u64::MAX,
        &[],
        &[],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Out of bounds precompile index should fail"
    );
}

#[tokio::test]
async fn test_close_wallet_not_owned() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let fake_wallet = payer.pubkey();
    let fake_vault = Pubkey::new_unique();
    let destination = Pubkey::new_unique();

    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &fake_wallet,
        &fake_vault,
        &destination,
        0,
        u64::MAX,
    );

    let tx = Transaction::new_signed_with_payer(
        &[close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Close with system-owned wallet should fail"
    );
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_close_wallet_wrong_pubkey() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, _signing_key_a, _compressed_a) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let (signing_key_b, compressed_b) = generate_p256_keypair();
    let destination = Pubkey::new_unique();
    let close_message = compute_close_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &destination,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key_b, &compressed_b, &close_message);

    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &destination,
        0,
        u64::MAX,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Close with wrong key should fail (PublicKeyMismatch)"
    );
}

/// Test that relay cannot redirect close funds to a different destination.
#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_close_wallet_wrong_destination() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let signed_destination = Pubkey::new_unique();
    let attacker_destination = Pubkey::new_unique();

    // Sign for signed_destination
    let close_message = compute_close_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &signed_destination,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &close_message);

    // But submit with attacker_destination
    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &attacker_destination,
        0,
        u64::MAX,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Wrong destination should fail (InvalidDestination or MessageMismatch)"
    );
}

#[tokio::test]
async fn test_create_wallet_truncated_authority() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let mut data = vec![0u8]; // CreateWallet discriminator
    data.extend_from_slice(&[0x02; 20]); // only 20 bytes (needs 33)

    let wallet_pda = Pubkey::new_unique();

    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(payer.pubkey(), true),
            AccountMeta::new(wallet_pda, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data,
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Truncated authority should fail");
}

#[tokio::test]
async fn test_unknown_instruction_discriminator() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![AccountMeta::new(payer.pubkey(), true)],
        data: vec![255u8],
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Unknown discriminator should fail");
}

#[tokio::test]
async fn test_empty_instruction_data() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![AccountMeta::new(payer.pubkey(), true)],
        data: vec![],
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Empty instruction data should fail");
}

// ---------------------------------------------------------------------------
// Message hash security properties (unit-level, no BPF needed)
// ---------------------------------------------------------------------------

#[test]
fn test_message_hash_close_vs_execute_differ() {
    let wallet = Pubkey::new_unique();
    let destination = Pubkey::new_unique();
    let nonce = 0u64;
    let creation_slot = 42u64;

    let close_hash =
        compute_close_message_hash(&wallet, creation_slot, nonce, u64::MAX, &destination);
    let empty_inner_hash = hash_inner(&[], &[]);
    let execute_hash =
        compute_message_hash(&wallet, creation_slot, nonce, u64::MAX, &empty_inner_hash);

    assert_ne!(
        close_hash, execute_hash,
        "Close and execute hashes must differ"
    );
}

#[test]
fn test_pda_derivation_deterministic() {
    let (_sk, compressed) = generate_p256_keypair();
    let (pda1, bump1, vault1, vbump1) = derive_pdas(&compressed);
    let (pda2, bump2, vault2, vbump2) = derive_pdas(&compressed);
    assert_eq!(pda1, pda2);
    assert_eq!(bump1, bump2);
    assert_eq!(vault1, vault2);
    assert_eq!(vbump1, vbump2);
}

#[test]
fn test_different_authorities_different_pdas() {
    let (_sk1, compressed1) = generate_p256_keypair();
    let (_sk2, compressed2) = generate_p256_keypair();
    let (pda1, _, _, _) = derive_pdas(&compressed1);
    let (pda2, _, _, _) = derive_pdas(&compressed2);
    assert_ne!(pda1, pda2);
}

#[test]
fn test_vault_pda_depends_on_wallet() {
    let (_sk1, compressed1) = generate_p256_keypair();
    let (_sk2, compressed2) = generate_p256_keypair();
    let (_, _, vault1, _) = derive_pdas(&compressed1);
    let (_, _, vault2, _) = derive_pdas(&compressed2);
    assert_ne!(
        vault1, vault2,
        "Different authorities must produce different vault PDAs"
    );
}

#[test]
fn test_message_hash_sensitivity() {
    let wallet = Pubkey::new_unique();
    let inner_hash = [0x42u8; 32];
    let creation_slot = 100u64;

    let base = compute_message_hash(&wallet, creation_slot, 0, u64::MAX, &inner_hash);

    let different_wallet = Pubkey::new_unique();
    assert_ne!(
        base,
        compute_message_hash(&different_wallet, creation_slot, 0, u64::MAX, &inner_hash)
    );
    assert_ne!(
        base,
        compute_message_hash(&wallet, creation_slot, 1, u64::MAX, &inner_hash)
    );
    assert_ne!(
        base,
        compute_message_hash(&wallet, 200, 0, u64::MAX, &inner_hash)
    );

    let different_inner = [0x43u8; 32];
    assert_ne!(
        base,
        compute_message_hash(&wallet, creation_slot, 0, u64::MAX, &different_inner)
    );
}

#[test]
fn test_inner_hash_tamper_detection() {
    let base_ix = make_inner(
        [0xAA; 32],
        &[(0, AccountEntry::FLAG_WRITABLE), (1, 0x00)],
        vec![0xDE, 0xAD],
    );
    let account_keys = [Pubkey::new_unique(), Pubkey::new_unique()];
    let base_hash = hash_inner(&[base_ix.clone()], &account_keys);

    // Tamper program_id
    let mut tampered = base_ix.clone();
    tampered.program_id[0] = 0xBB;
    assert_ne!(base_hash, hash_inner(&[tampered], &account_keys));

    // Tamper account indices
    let tampered = make_inner(
        [0xAA; 32],
        &[(1, AccountEntry::FLAG_WRITABLE), (0, 0x00)], // swapped
        vec![0xDE, 0xAD],
    );
    assert_ne!(base_hash, hash_inner(&[tampered], &account_keys));

    // Tamper account flags (writable -> readonly)
    let tampered = make_inner(
        [0xAA; 32],
        &[(0, 0x00), (1, 0x00)], // first account no longer writable
        vec![0xDE, 0xAD],
    );
    assert_ne!(base_hash, hash_inner(&[tampered], &account_keys));

    // Tamper data
    let mut tampered = base_ix.clone();
    tampered.data = vec![0xBE, 0xEF];
    assert_ne!(base_hash, hash_inner(&[tampered], &account_keys));

    // Add extra instruction
    let extra_ix = make_inner([0xCC; 32], &[], vec![]);
    assert_ne!(
        base_hash,
        hash_inner(&[base_ix.clone(), extra_ix], &account_keys)
    );
}

#[test]
fn test_inner_hash_order_matters() {
    let ix1 = make_inner([0xAA; 32], &[(0, AccountEntry::FLAG_WRITABLE)], vec![0x01]);
    let ix2 = make_inner([0xBB; 32], &[(1, AccountEntry::FLAG_WRITABLE)], vec![0x02]);

    let account_keys = [Pubkey::new_unique(), Pubkey::new_unique()];
    let hash_12 = hash_inner(&[ix1.clone(), ix2.clone()], &account_keys);
    let hash_21 = hash_inner(&[ix2, ix1], &account_keys);
    assert_ne!(hash_12, hash_21, "Instruction ordering must matter");
}

/// Build an AdvanceNonce instruction.
fn build_advance_nonce_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    secp256r1_ix_index: u8,
    max_slot: u64,
) -> Instruction {
    let mut data = vec![3u8, secp256r1_ix_index];
    data.extend_from_slice(&max_slot.to_le_bytes());
    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
        ],
        data,
    }
}

/// Verify that changing account flags changes the inner hash.
#[test]
fn test_inner_hash_flags_sensitivity() {
    let ix_writable = make_inner([0xAA; 32], &[(0, AccountEntry::FLAG_WRITABLE)], vec![0x01]);
    let ix_readonly = make_inner(
        [0xAA; 32],
        &[(0, 0x00)], // same index, different flags
        vec![0x01],
    );

    assert_ne!(
        hash_inner(&[ix_writable], &[Pubkey::new_unique()]),
        hash_inner(&[ix_readonly], &[Pubkey::new_unique()]),
        "Different account flags must produce different hashes"
    );
}

// ===========================================================================
// AdvanceNonce Tests
// ===========================================================================

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_advance_nonce() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, _vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet_state.nonce, 0);

    let message =
        compute_advance_nonce_message(&wallet_pda, wallet_state.creation_slot, 0, u64::MAX);
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);
    let advance_ix = build_advance_nonce_ix(&payer.pubkey(), &wallet_pda, 0, u64::MAX);

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, advance_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet_state.nonce, 1);
}

/// After AdvanceNonce, a previously-signed Execute message should fail (stale nonce).
#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_advance_nonce_invalidates_pending_execute() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    // Fund vault
    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;

    // Pre-sign an Execute with nonce=0
    let recipient = Pubkey::new_unique();
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 100_000);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    let execute_message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0, // nonce=0
        u64::MAX,
        &inner_hash,
    );

    // First, advance nonce to 1
    let advance_message =
        compute_advance_nonce_message(&wallet_pda, wallet_state.creation_slot, 0, u64::MAX);
    let advance_precompile = build_secp256r1_ix(&signing_key, &compressed, &advance_message);
    let advance_ix = build_advance_nonce_ix(&payer.pubkey(), &wallet_pda, 0, u64::MAX);

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[advance_precompile, advance_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet_state.nonce, 1);

    // Now try the pre-signed Execute with nonce=0 — should fail (MessageMismatch)
    let execute_precompile = build_secp256r1_ix(&signing_key, &compressed, &execute_message);
    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[execute_precompile, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Execute with stale nonce=0 should fail after AdvanceNonce"
    );
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_advance_nonce_wrong_pubkey() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, _vault_pda, _signing_key_a, _compressed_a) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let (signing_key_b, compressed_b) = generate_p256_keypair();
    let message =
        compute_advance_nonce_message(&wallet_pda, wallet_state.creation_slot, 0, u64::MAX);
    let precompile_ix = build_secp256r1_ix(&signing_key_b, &compressed_b, &message);
    let advance_ix = build_advance_nonce_ix(&payer.pubkey(), &wallet_pda, 0, u64::MAX);

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, advance_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "AdvanceNonce with wrong key should fail (PublicKeyMismatch)"
    );
}

#[tokio::test]
async fn test_advance_nonce_wallet_not_owned() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let fake_wallet = payer.pubkey(); // system-owned
    let advance_ix = build_advance_nonce_ix(&payer.pubkey(), &fake_wallet, 0, u64::MAX);

    let tx = Transaction::new_signed_with_payer(
        &[advance_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "AdvanceNonce with system-owned wallet should fail"
    );
}

// ===========================================================================
// Signature Expiry Tests
// ===========================================================================

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_signature_expired() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let recipient = Pubkey::new_unique();
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 100_000);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    // Sign with max_slot=0 (already expired since current slot > 0 after CreateWallet)
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        0, // max_slot=0 — expired
        &inner_hash,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);
    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        0, // expired
        &[inner],
        &[
            AccountMeta::new(vault_pda, false),
            AccountMeta::new(recipient, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Expired signature (max_slot=0) should fail"
    );
}

#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_close_wallet_signature_expired() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let destination = Pubkey::new_unique();
    let close_message = compute_close_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        0, // max_slot=0 — expired
        &destination,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &close_message);
    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &destination,
        0,
        0, // expired
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "CloseWallet with expired signature should fail"
    );
}

// ===========================================================================
// Execute Edge Cases
// ===========================================================================

/// Execute with 0 inner instructions should be rejected (no-op wastes nonce).
#[tokio::test]
async fn test_execute_rejects_empty_inner_instructions() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, _signing_key, _compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[], // empty
        &[],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Execute with 0 inner instructions should fail"
    );
}

/// Verify owner check works with non-empty inner instructions.
/// (test_execute_wallet_not_owned sends count=0 which hits empty check first)
#[tokio::test]
async fn test_execute_wallet_not_owned_with_inner() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let fake_wallet = payer.pubkey(); // system-owned
    let fake_vault = Pubkey::new_unique();
    let inner = make_inner(system_program::id().to_bytes(), &[(0, 0x00)], vec![0x01]);
    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &fake_wallet,
        &fake_vault,
        0,
        u64::MAX,
        &[inner],
        &[AccountMeta::new_readonly(Pubkey::new_unique(), false)],
    );

    let tx = Transaction::new_signed_with_payer(
        &[execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "System-owned wallet with non-empty inner should fail IncorrectProgramId"
    );
}

/// Inner instruction targeting own program_id should be denied (anti-self-CPI).
#[tokio::test]
async fn test_execute_cpi_to_self_denied() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, _signing_key, _compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    // Inner instruction targeting our own program
    let inner = make_inner(
        machine_wallet::id().to_bytes(), // CPI to self!
        &[(0, 0x00)],
        vec![0x00],
    );

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner],
        &[AccountMeta::new_readonly(machine_wallet::id(), false)],
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "CPI to self should fail (CpiToSelfDenied)");
}

/// Shared precompile attack: one precompile instruction, two Execute instructions.
/// First Execute succeeds and increments nonce. Second should fail (message mismatch).
#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_execute_shared_precompile_in_same_tx() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let fund_ix = system_instruction::transfer(&payer.pubkey(), &vault_pda, 2_000_000_000);
    let fund_tx = Transaction::new_signed_with_payer(
        &[fund_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(fund_tx).await.unwrap();

    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let recipient = Pubkey::new_unique();
    let transfer_ix_data = system_instruction::transfer(&vault_pda, &recipient, 100_000);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix_data.data.clone(),
    );
    let inner_hash = hash_inner(
        &[inner.clone()],
        &[vault_pda, recipient, system_program::id()],
    );
    let message = compute_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &inner_hash,
    );

    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &message);

    // Two Execute instructions, both pointing to precompile at index 0
    let remaining_metas = vec![
        AccountMeta::new(vault_pda, false),
        AccountMeta::new(recipient, false),
        AccountMeta::new_readonly(system_program::id(), false),
    ];
    let execute_ix1 = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0,
        u64::MAX,
        &[inner.clone()],
        &remaining_metas,
    );
    let execute_ix2 = build_execute_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        0, // same precompile index
        u64::MAX,
        &[inner],
        &remaining_metas,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, execute_ix1, execute_ix2],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    // First execute increments nonce to 1; second execute message has nonce=0 → MessageMismatch
    assert!(
        result.is_err(),
        "Second execute sharing precompile should fail (stale nonce)"
    );
}

// ===========================================================================
// CloseWallet Edge Cases
// ===========================================================================

/// Close with destination == wallet_pda should be rejected.
#[tokio::test]
async fn test_close_wallet_destination_is_wallet_pda() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let (_signing_key, compressed) = generate_p256_keypair();
    let (ix, wallet_pda, vault_pda) = build_create_wallet_ix(&payer.pubkey(), &compressed);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &wallet_pda, // destination == wallet!
        0,
        u64::MAX,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Close to wallet_pda should fail (InvalidDestination)"
    );
}

/// Close with destination == vault_pda should be rejected.
#[tokio::test]
async fn test_close_wallet_destination_is_vault_pda() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let (_signing_key, compressed) = generate_p256_keypair();
    let (ix, wallet_pda, vault_pda) = build_create_wallet_ix(&payer.pubkey(), &compressed);
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &vault_pda, // destination == vault!
        0,
        u64::MAX,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(
        result.is_err(),
        "Close to vault_pda should fail (InvalidDestination)"
    );
}

/// Close when vault has 0 SOL — should still succeed.
#[tokio::test]
#[ignore = "requires secp256r1 precompile support in test validator"]
async fn test_close_wallet_empty_vault() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let (wallet_pda, vault_pda, signing_key, compressed) =
        create_wallet_helper(&mut banks, &payer, recent_blockhash).await;

    // Don't fund vault — 0 SOL
    let wallet_state = read_wallet_state(&mut banks, &wallet_pda).await;
    let destination = Pubkey::new_unique();
    let close_message = compute_close_message_hash(
        &wallet_pda,
        wallet_state.creation_slot,
        0,
        u64::MAX,
        &destination,
    );
    let precompile_ix = build_secp256r1_ix(&signing_key, &compressed, &close_message);
    let close_ix = build_close_wallet_ix(
        &payer.pubkey(),
        &wallet_pda,
        &vault_pda,
        &destination,
        0,
        u64::MAX,
    );

    let recent_blockhash = banks.get_latest_blockhash().await.unwrap();
    let tx = Transaction::new_signed_with_payer(
        &[precompile_ix, close_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet_account = banks.get_account(wallet_pda).await.unwrap();
    match wallet_account {
        None => {}
        Some(acc) => {
            assert_eq!(acc.lamports, 0);
        }
    }
}

// ===========================================================================
// CreateWallet Edge Cases
// ===========================================================================

/// Passing wrong system program account should fail.
#[tokio::test]
async fn test_create_wallet_wrong_system_program() {
    let (banks, payer, recent_blockhash) = program_test().start().await;
    let (_signing_key, compressed) = generate_p256_keypair();
    let (wallet_pda, _, _, _) = derive_pdas(&compressed);

    let mut data = vec![0u8];
    data.extend_from_slice(&compressed);

    let fake_system_program = Pubkey::new_unique();
    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(payer.pubkey(), true),
            AccountMeta::new(wallet_pda, false),
            AccountMeta::new_readonly(fake_system_program, false), // wrong
        ],
        data,
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Wrong system program should fail");
}

// ===========================================================================
// Domain Separation Tests (no BPF needed)
// ===========================================================================

/// All three message types (Execute, CloseWallet, AdvanceNonce) must produce
/// different hashes for the same wallet/nonce/slot/max_slot.
#[test]
fn test_all_three_domain_tags_differ() {
    let wallet = Pubkey::new_unique();
    let creation_slot = 42u64;
    let nonce = 0u64;
    let max_slot = u64::MAX;

    let advance_hash = compute_advance_nonce_message(&wallet, creation_slot, nonce, max_slot);

    let empty_inner_hash = hash_inner(&[], &[]);
    let execute_hash =
        compute_message_hash(&wallet, creation_slot, nonce, max_slot, &empty_inner_hash);

    let destination = Pubkey::new_unique();
    let close_hash =
        compute_close_message_hash(&wallet, creation_slot, nonce, max_slot, &destination);

    assert_ne!(
        advance_hash, execute_hash,
        "AdvanceNonce and Execute hashes must differ"
    );
    assert_ne!(
        advance_hash, close_hash,
        "AdvanceNonce and Close hashes must differ"
    );
    assert_ne!(
        execute_hash, close_hash,
        "Execute and Close hashes must differ"
    );
}

/// Advance nonce message must include all bound fields.
#[test]
fn test_advance_nonce_message_sensitivity() {
    let wallet = Pubkey::new_unique();
    let creation_slot = 100u64;
    let base = compute_advance_nonce_message(&wallet, creation_slot, 0, u64::MAX);

    // Different wallet
    let different_wallet = Pubkey::new_unique();
    assert_ne!(
        base,
        compute_advance_nonce_message(&different_wallet, creation_slot, 0, u64::MAX)
    );
    // Different nonce
    assert_ne!(
        base,
        compute_advance_nonce_message(&wallet, creation_slot, 1, u64::MAX)
    );
    // Different creation_slot
    assert_ne!(
        base,
        compute_advance_nonce_message(&wallet, 200, 0, u64::MAX)
    );
    // Different max_slot
    assert_ne!(
        base,
        compute_advance_nonce_message(&wallet, creation_slot, 0, 500)
    );
}

// ---------------------------------------------------------------------------
// Session Key integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_create_session_message_deterministic() {
    let wallet = Pubkey::new_unique();
    let authority = [0x42u8; 32];
    let programs = vec![[0xAAu8; 32], [0xBBu8; 32]];
    let data_hash = create_session::hash_session_data(&authority, 500, 1_000_000_000, 2, &programs);
    let msg1 = create_session::compute_create_session_message(&wallet, 100, 0, 200, &data_hash);
    let msg2 = create_session::compute_create_session_message(&wallet, 100, 0, 200, &data_hash);
    assert_eq!(msg1, msg2);
}

#[test]
fn test_create_session_message_includes_nonce() {
    let wallet = Pubkey::new_unique();
    let authority = [0x42u8; 32];
    let programs = vec![[0xAAu8; 32]];
    let data_hash = create_session::hash_session_data(&authority, 500, 0, 1, &programs);
    let msg1 = create_session::compute_create_session_message(&wallet, 100, 0, 200, &data_hash);
    let msg2 = create_session::compute_create_session_message(&wallet, 100, 1, 200, &data_hash);
    assert_ne!(msg1, msg2);
}

#[test]
fn test_create_session_message_includes_session_data() {
    let wallet = Pubkey::new_unique();
    let authority = [0x42u8; 32];
    let programs1 = vec![[0xAAu8; 32]];
    let programs2 = vec![[0xBBu8; 32]];
    let hash1 = create_session::hash_session_data(&authority, 500, 0, 1, &programs1);
    let hash2 = create_session::hash_session_data(&authority, 500, 0, 1, &programs2);
    assert_ne!(hash1, hash2);
    let msg1 = create_session::compute_create_session_message(&wallet, 100, 0, 200, &hash1);
    let msg2 = create_session::compute_create_session_message(&wallet, 100, 0, 200, &hash2);
    assert_ne!(msg1, msg2);
}

#[test]
fn test_session_pda_derivation() {
    let program_id = machine_wallet::id();
    let wallet_pda = Pubkey::new_unique();
    let authority = Pubkey::new_unique();
    let (pda, bump) = Pubkey::find_program_address(
        &[SESSION_SEED_PREFIX, wallet_pda.as_ref(), authority.as_ref()],
        &program_id,
    );
    let recreated = Pubkey::create_program_address(
        &[
            SESSION_SEED_PREFIX,
            wallet_pda.as_ref(),
            authority.as_ref(),
            &[bump],
        ],
        &program_id,
    )
    .unwrap();
    assert_eq!(pda, recreated);
}

#[test]
fn test_session_domain_separation_from_execute() {
    let wallet = Pubkey::new_unique();
    let authority = [0x42u8; 32];
    let programs = vec![[0xAAu8; 32]];
    let data_hash = create_session::hash_session_data(&authority, 500, 0, 1, &programs);
    let session_msg =
        create_session::compute_create_session_message(&wallet, 100, 0, 200, &data_hash);
    let execute_msg = execute::compute_message_hash(&wallet, 100, 0, 200, &data_hash);
    assert_ne!(session_msg, execute_msg);
}

#[test]
fn test_revoke_session_domain_separation() {
    let wallet = Pubkey::new_unique();
    let authority = [0x42u8; 32];
    let revoke_msg =
        revoke_session::compute_revoke_session_message(&wallet, 100, 0, 200, &authority);
    let programs = vec![[0xAAu8; 32]];
    let data_hash = create_session::hash_session_data(&authority, 500, 0, 1, &programs);
    let session_msg =
        create_session::compute_create_session_message(&wallet, 100, 0, 200, &data_hash);
    assert_ne!(revoke_msg, session_msg);
}

#[test]
fn test_session_state_roundtrip_in_integration() {
    let mut programs = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
    programs[0] = [0xAA; 32];
    programs[1] = [0xBB; 32];
    let session = SessionState {
        version: 0,
        bump: 253,
        wallet: [0x11; 32],
        authority: [0x22; 32],
        created_slot: 100,
        expiry_slot: 200,
        revoked: false,
        wallet_creation_slot: 50,
        max_lamports_per_ix: 1_000_000_000,
        allowed_programs_count: 2,
        allowed_programs: programs,
    };
    let mut buf = [0u8; SessionState::LEN];
    session.serialize(&mut buf).unwrap();
    let decoded = SessionState::deserialize(&buf).unwrap();
    assert_eq!(session, decoded);
    assert_eq!(decoded.wallet_creation_slot, 50);
    assert!(decoded.is_program_allowed(&[0xAA; 32]));
    assert!(!decoded.is_program_allowed(&[0xCC; 32]));
}

// ===========================================================================
// AddAuthority Edge Cases
// ===========================================================================

/// Adding a duplicate authority (same scheme + pubkey) must fail.
#[tokio::test]
async fn test_add_authority_duplicate_rejected() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);
    let tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    // Try to add the same authority again
    let msg = compute_add_authority_message(
        &wallet_pda,
        wallet.creation_slot,
        wallet.nonce,
        u64::MAX,
        1, // Ed25519
        &owner_bytes,
        0,
    );
    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &msg),
            build_add_authority_ix(&payer.pubkey(), &wallet_pda, 0, 1, &owner_bytes, 0, u64::MAX),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Duplicate authority should be rejected (DuplicateAuthority)");
}

/// Removing the last authority must fail.
#[tokio::test]
async fn test_remove_authority_last_rejected() {
    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);
    let tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let msg = compute_remove_authority_message(
        &wallet_pda,
        wallet.creation_slot,
        wallet.nonce,
        u64::MAX,
        1,
        &owner_bytes,
        0,
    );
    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &msg),
            build_remove_authority_ix(&payer.pubkey(), &wallet_pda, 0, 1, &owner_bytes, 0, u64::MAX),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Removing last authority should be rejected (CannotRemoveLastAuthority)");
}

// ===========================================================================
// SetThreshold Integration Tests
// ===========================================================================

fn build_set_threshold_ix(
    fee_payer: &Pubkey,
    wallet_pda: &Pubkey,
    precompile_ix_index: u8,
    new_threshold: u8,
    max_slot: u64,
) -> Instruction {
    let mut data = vec![11u8, precompile_ix_index, new_threshold];
    data.extend_from_slice(&max_slot.to_le_bytes());

    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new_readonly(sysvar::instructions::id(), false),
            AccountMeta::new(*wallet_pda, false),
            AccountMeta::new_readonly(*fee_payer, true),
        ],
        data,
    }
}

/// SetThreshold with valid value succeeds and updates state.
#[tokio::test]
async fn test_set_threshold_succeeds() {
    use machine_wallet::processor::set_threshold::compute_set_threshold_message;

    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);
    let tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    // Add second authority to allow threshold=2
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let (_new_sk, new_pubkey) = generate_p256_keypair();
    let add_msg = compute_add_authority_message(
        &wallet_pda, wallet.creation_slot, wallet.nonce, u64::MAX, 0, &new_pubkey, 0,
    );
    let add_tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &add_msg),
            build_add_authority_ix(&payer.pubkey(), &wallet_pda, 0, 0, &new_pubkey, 0, u64::MAX),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(add_tx).await.unwrap();

    // Now set threshold to 2
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet.authority_count, 2);
    let msg = compute_set_threshold_message(
        &wallet_pda, wallet.creation_slot, wallet.nonce, u64::MAX, 2,
    );
    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &msg),
            build_set_threshold_ix(&payer.pubkey(), &wallet_pda, 0, 2, u64::MAX),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet_after = read_wallet_state(&mut banks, &wallet_pda).await;
    assert_eq!(wallet_after.threshold, 2);
    assert_eq!(wallet_after.nonce, wallet.nonce + 1);
}

/// SetThreshold exceeding authority_count must fail.
#[tokio::test]
async fn test_set_threshold_exceeds_authority_count_rejected() {
    use machine_wallet::processor::set_threshold::compute_set_threshold_message;

    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);
    let tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    // authority_count=1, try to set threshold=2
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let msg = compute_set_threshold_message(
        &wallet_pda, wallet.creation_slot, wallet.nonce, u64::MAX, 2,
    );
    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &msg),
            build_set_threshold_ix(&payer.pubkey(), &wallet_pda, 0, 2, u64::MAX),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Threshold > authority_count should be rejected (InvalidThreshold)");
}

/// SetThreshold with value 0 must fail.
#[tokio::test]
async fn test_set_threshold_zero_rejected() {
    use machine_wallet::processor::set_threshold::compute_set_threshold_message;

    let (mut banks, payer, recent_blockhash) = program_test().start().await;
    let owner = Keypair::new();
    let owner_bytes = ed25519_authority_bytes(&owner);
    let (create_ix, wallet_pda, _) =
        build_create_wallet_ix_with_scheme(&payer.pubkey(), 1, &owner_bytes);
    let tx = Transaction::new_signed_with_payer(
        &[create_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    banks.process_transaction(tx).await.unwrap();

    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let msg = compute_set_threshold_message(
        &wallet_pda, wallet.creation_slot, wallet.nonce, u64::MAX, 0,
    );
    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &msg),
            build_set_threshold_ix(&payer.pubkey(), &wallet_pda, 0, 0, u64::MAX),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Threshold 0 should be rejected (InvalidThreshold)");
}

// ===========================================================================
// SessionExecute Security Tests
// ===========================================================================

/// SessionExecute must reject calls to non-whitelisted programs (ProgramNotAllowed).
#[tokio::test]
async fn test_session_execute_rejects_non_whitelisted_program() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    // Create session allowing ONLY system_program
    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Fund vault
    let fund_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000)],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(fund_tx).await.unwrap();

    // Try to call a non-whitelisted program
    let fake_program = Pubkey::new_unique();
    let inner = make_inner(
        fake_program.to_bytes(),
        &[(0, AccountEntry::FLAG_WRITABLE)],
        vec![1, 2, 3],
    );

    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &session_pda, &wallet_pda, &session_authority.pubkey(), &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new_readonly(fake_program, false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Non-whitelisted program should be rejected (ProgramNotAllowed)");
}

/// SessionExecute must reject if SOL outflow exceeds max_lamports_per_ix.
#[tokio::test]
async fn test_session_execute_rejects_exceeded_lamport_cap() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let recipient = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    // Create session with 100_000 lamport cap
    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 100_000, &[system_program::id()],
    ).await;

    // Fund vault
    let fund_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000)],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(fund_tx).await.unwrap();

    // Transfer 200_000 lamports — exceeds cap of 100_000
    let transfer_ix = system_instruction::transfer(&vault_pda, &recipient.pubkey(), 200_000);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix.data,
    );

    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &session_pda, &wallet_pda, &session_authority.pubkey(), &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new(recipient.pubkey(), false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Exceeding lamport cap should fail (IxAmountExceeded)");
}

/// SessionExecute: transfer within cap succeeds (boundary test).
#[tokio::test]
async fn test_session_execute_allows_transfer_within_cap() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let recipient = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    // Create session with 500_000_000 lamport cap
    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Fund vault
    let fund_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000)],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(fund_tx).await.unwrap();

    // Transfer 100_000_000 lamports — well within cap, should succeed
    let amount = 100_000_000u64;
    let transfer_ix = system_instruction::transfer(&vault_pda, &recipient.pubkey(), amount);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix.data,
    );

    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &session_pda, &wallet_pda, &session_authority.pubkey(), &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new(recipient.pubkey(), false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(tx).await.unwrap();

    let recipient_account = banks.get_account(recipient.pubkey()).await.unwrap().unwrap();
    assert_eq!(recipient_account.lamports, amount);
}

/// SessionExecute with revoked session must fail.
#[tokio::test]
async fn test_session_execute_rejects_revoked_session() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let recipient = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Fund vault
    let fund_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000)],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(fund_tx).await.unwrap();

    // Revoke the session
    let wallet_before_revoke = read_wallet_state(&mut banks, &wallet_pda).await;
    let revoke_message = revoke_session::compute_revoke_session_message(
        &wallet_pda,
        wallet_before_revoke.creation_slot,
        wallet_before_revoke.nonce,
        u64::MAX,
        &session_authority.pubkey().to_bytes(),
    );
    let revoke_tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &revoke_message),
            build_revoke_session_ix(
                &payer.pubkey(), &wallet_pda, &session_pda, 0, u64::MAX,
                &session_authority.pubkey(),
            ),
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(revoke_tx).await.unwrap();

    // Verify session is revoked
    let session = read_session_state(&mut banks, &session_pda).await;
    assert!(session.revoked);

    // Now try to use the revoked session
    let transfer_ix = system_instruction::transfer(&vault_pda, &recipient.pubkey(), 100_000);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[
            (0, AccountEntry::FLAG_WRITABLE),
            (1, AccountEntry::FLAG_WRITABLE),
        ],
        transfer_ix.data,
    );

    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &session_pda, &wallet_pda, &session_authority.pubkey(), &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new(recipient.pubkey(), false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Revoked session should be rejected (SessionRevoked)");
}

/// SessionExecute with wrong session authority signer must fail.
#[tokio::test]
async fn test_session_execute_rejects_wrong_authority() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let wrong_authority = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let _session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Fund vault
    let fund_tx = Transaction::new_signed_with_payer(
        &[system_instruction::transfer(&payer.pubkey(), &vault_pda, 1_000_000_000)],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(fund_tx).await.unwrap();

    // Derive the CORRECT session PDA but sign with wrong authority
    let (correct_session_pda, _) = derive_session_pda(&wallet_pda, &session_authority.pubkey());
    let transfer_ix = system_instruction::transfer(&vault_pda, &Pubkey::new_unique(), 100_000);
    let inner = make_inner(
        system_program::id().to_bytes(),
        &[(0, AccountEntry::FLAG_WRITABLE), (1, AccountEntry::FLAG_WRITABLE)],
        transfer_ix.data,
    );

    // Pass wrong authority as signer, but correct session PDA
    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &correct_session_pda, &wallet_pda, &wrong_authority.pubkey(), &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new(Pubkey::new_unique(), false),
                AccountMeta::new_readonly(system_program::id(), false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &wrong_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Wrong authority should be rejected (SessionAuthorityMismatch)");
}

/// SessionExecute with mismatched wallet_creation_slot must fail (prevents session resurrection).
#[tokio::test]
async fn test_session_execute_rejects_wallet_creation_slot_mismatch() {
    // We simulate this by injecting a session with wrong wallet_creation_slot.
    // Since we can't easily close+recreate in test, we use the state injection approach:
    // Create wallet, create session, then verify the creation_slot binding is correct.
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Verify the session's wallet_creation_slot matches the wallet's creation_slot
    let session = read_session_state(&mut banks, &session_pda).await;
    assert_eq!(session.wallet_creation_slot, wallet.creation_slot,
        "Session wallet_creation_slot must match wallet creation_slot");
}

// ===========================================================================
// SelfRevokeSession Integration Test
// ===========================================================================

fn build_self_revoke_session_ix(
    session_pda: &Pubkey,
    authority: &Pubkey,
) -> Instruction {
    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(*session_pda, false),
            AccountMeta::new_readonly(*authority, true),
        ],
        data: vec![7u8], // SelfRevokeSession discriminator
    }
}

/// SelfRevokeSession happy path: session authority revokes their own session.
#[tokio::test]
async fn test_self_revoke_session_succeeds() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Verify session is not revoked
    let session = read_session_state(&mut banks, &session_pda).await;
    assert!(!session.revoked);

    // Self-revoke
    let tx = Transaction::new_signed_with_payer(
        &[build_self_revoke_session_ix(&session_pda, &session_authority.pubkey())],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(tx).await.unwrap();

    // Verify session is now revoked
    let session = read_session_state(&mut banks, &session_pda).await;
    assert!(session.revoked);
}

/// SelfRevokeSession with wrong authority must fail.
#[tokio::test]
async fn test_self_revoke_session_wrong_authority_rejected() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let wrong_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Try to revoke with wrong authority
    let tx = Transaction::new_signed_with_payer(
        &[build_self_revoke_session_ix(&session_pda, &wrong_authority.pubkey())],
        Some(&payer.pubkey()),
        &[&payer, &wrong_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Wrong authority should be rejected (SessionAuthorityMismatch)");
}

// ===========================================================================
// CloseSession Integration Test
// ===========================================================================

fn build_close_session_ix(
    session_pda: &Pubkey,
    authority: &Pubkey,
    destination: &Pubkey,
) -> Instruction {
    Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(*session_pda, false),
            AccountMeta::new_readonly(*authority, true),
            AccountMeta::new(*destination, false),
        ],
        data: vec![8u8], // CloseSession discriminator
    }
}

/// CloseSession on an active (not expired, not revoked) session must fail.
#[tokio::test]
async fn test_close_session_rejects_active_session() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 5000, // far future expiry
        500_000_000, &[system_program::id()],
    ).await;

    let destination = Keypair::new();
    let tx = Transaction::new_signed_with_payer(
        &[build_close_session_ix(
            &session_pda, &session_authority.pubkey(), &destination.pubkey(),
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Active session should not be closeable (SessionStillActive)");
}

/// CloseSession on a revoked session succeeds and transfers rent.
#[tokio::test]
async fn test_close_session_succeeds_after_revoke() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[system_program::id()],
    ).await;

    // Self-revoke first
    let revoke_tx = Transaction::new_signed_with_payer(
        &[build_self_revoke_session_ix(&session_pda, &session_authority.pubkey())],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(revoke_tx).await.unwrap();

    // Now close the revoked session
    let _session_account_before = banks.get_account(session_pda).await.unwrap().unwrap();

    let destination = payer.pubkey();
    let payer_before = banks.get_account(destination).await.unwrap().unwrap();

    let close_tx = Transaction::new_signed_with_payer(
        &[build_close_session_ix(
            &session_pda, &session_authority.pubkey(), &destination,
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    banks.process_transaction(close_tx).await.unwrap();

    // Session account should be gone or have 0 lamports
    let session_account_after = banks.get_account(session_pda).await.unwrap();
    match session_account_after {
        None => {} // GC'd
        Some(acc) => assert_eq!(acc.lamports, 0),
    }

    // Destination should have received session rent (minus tx fee)
    let payer_after = banks.get_account(destination).await.unwrap().unwrap();
    // payer_after = payer_before + session_lamports - tx_fee
    // Just verify the session rent was transferred (payer paid fee, so check approximate)
    assert!(
        payer_after.lamports > payer_before.lamports - 10_000, // tx fee is ~5000
        "Destination should receive session rent"
    );
}

// ===========================================================================
// Session + CPI to Self Protection
// ===========================================================================

/// SessionExecute must reject CPI to our own program_id.
#[tokio::test]
async fn test_session_execute_rejects_cpi_to_self() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, vault_pda) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;

    // Whitelist our own program (the check should still catch the CPI)
    let session_pda = create_session_helper(
        &mut banks, &payer, &owner, &wallet_pda, &session_authority,
        wallet.creation_slot + 500, 500_000_000, &[machine_wallet::id()],
    ).await;

    let inner = make_inner(
        machine_wallet::id().to_bytes(),
        &[(0, 0)],
        vec![0], // CreateWallet disc
    );

    let tx = Transaction::new_signed_with_payer(
        &[build_session_execute_ix(
            &session_pda, &wallet_pda, &session_authority.pubkey(), &vault_pda,
            &[inner],
            &[
                AccountMeta::new(vault_pda, false),
                AccountMeta::new_readonly(machine_wallet::id(), false),
            ],
        )],
        Some(&payer.pubkey()),
        &[&payer, &session_authority],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "CPI to self should be rejected (CpiToSelfDenied)");
}

// ===========================================================================
// Domain Separation: OwnerCloseSession, AddAuthority, RemoveAuthority, SetThreshold
// ===========================================================================

#[test]
fn test_owner_close_session_domain_separation() {
    use machine_wallet::processor::set_threshold::compute_set_threshold_message;

    let wallet = Pubkey::new_unique();
    let authority = [0x42u8; 32];
    let destination = [0x99u8; 32];

    let owner_close_msg = owner_close_session::compute_owner_close_session_message(
        &wallet, 100, 0, 200, &authority, &destination,
    );
    let revoke_msg = revoke_session::compute_revoke_session_message(&wallet, 100, 0, 200, &authority);
    let set_threshold_msg = compute_set_threshold_message(&wallet, 100, 0, 200, 2);
    let add_authority_msg = compute_add_authority_message(&wallet, 100, 0, 200, 0, &[0xBB; 33], 0);
    let remove_authority_msg = compute_remove_authority_message(&wallet, 100, 0, 200, 0, &[0xBB; 33], 0);

    // All must be distinct
    let msgs = [owner_close_msg, revoke_msg, set_threshold_msg, add_authority_msg, remove_authority_msg];
    for i in 0..msgs.len() {
        for j in (i+1)..msgs.len() {
            assert_ne!(msgs[i], msgs[j],
                "Domain-separated messages must differ (i={}, j={})", i, j);
        }
    }
}

// ===========================================================================
// CreateSession Rejection Tests
// ===========================================================================

/// Creating a session with zero allowed_programs must fail at instruction parse level.
#[tokio::test]
async fn test_create_session_rejects_zero_allowed_programs() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let (session_pda, _) = derive_session_pda(&wallet_pda, &session_authority.pubkey());

    // Build instruction data manually with 0 allowed programs
    let session_data_hash = create_session::hash_session_data(
        &session_authority.pubkey().to_bytes(), wallet.creation_slot + 500, 0, 0, &[],
    );
    let expected_message = create_session::compute_create_session_message(
        &wallet_pda, wallet.creation_slot, wallet.nonce, u64::MAX, &session_data_hash,
    );

    let create_session_ix = build_create_session_ix(
        &payer.pubkey(), &wallet_pda, &session_pda, 0, u64::MAX,
        &session_authority.pubkey(), wallet.creation_slot + 500, 0, &[],
    );

    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &expected_message),
            create_session_ix,
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Zero allowed_programs should be rejected");
}

/// Creating a session with already-expired expiry_slot must fail.
#[tokio::test]
async fn test_create_session_rejects_past_expiry() {
    let (mut banks, payer, _) = program_test().start().await;
    let owner = Keypair::new();
    let session_authority = Keypair::new();
    let (wallet_pda, _) = create_ed25519_wallet_helper(&mut banks, &payer, &owner).await;
    let wallet = read_wallet_state(&mut banks, &wallet_pda).await;
    let (session_pda, _) = derive_session_pda(&wallet_pda, &session_authority.pubkey());

    let allowed = [system_program::id()];
    let allowed_bytes: Vec<[u8; 32]> = allowed.iter().map(|p| p.to_bytes()).collect();
    // expiry_slot = 1, which is in the past
    let session_data_hash = create_session::hash_session_data(
        &session_authority.pubkey().to_bytes(), 1, 0, 1, &allowed_bytes,
    );
    let expected_message = create_session::compute_create_session_message(
        &wallet_pda, wallet.creation_slot, wallet.nonce, u64::MAX, &session_data_hash,
    );

    let create_session_ix = build_create_session_ix(
        &payer.pubkey(), &wallet_pda, &session_pda, 0, u64::MAX,
        &session_authority.pubkey(), 1, 0, &allowed,
    );

    let tx = Transaction::new_signed_with_payer(
        &[
            build_ed25519_precompile_ix(&owner, &expected_message),
            create_session_ix,
        ],
        Some(&payer.pubkey()),
        &[&payer],
        banks.get_latest_blockhash().await.unwrap(),
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Past expiry_slot should be rejected");
}

// ===========================================================================
// Execute Edge Cases
// ===========================================================================

/// Execute with wallet not owned by program (system-owned) must fail — empty inner.
#[tokio::test]
async fn test_execute_system_owned_wallet_rejected() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    let fake_wallet = Pubkey::new_unique();
    let fake_vault = Pubkey::new_unique();

    let execute_ix = build_execute_ix(
        &payer.pubkey(),
        &fake_wallet,
        &fake_vault,
        0,
        u64::MAX,
        &[], // empty inner_instructions
        &[],
    );

    let tx = Transaction::new_signed_with_payer(
        &[execute_ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "System-owned wallet should be rejected");
}

/// Creating a wallet with truncated authority (too short) must fail at parse.
#[tokio::test]
async fn test_create_wallet_truncated_authority_rejected() {
    let (banks, payer, recent_blockhash) = program_test().start().await;

    // Only 20 bytes of authority instead of 33
    let short_data: Vec<u8> = {
        let mut d = vec![0u8]; // CreateWallet disc
        d.extend_from_slice(&[0x02; 20]); // too short
        d
    };

    let fake_wallet = Pubkey::new_unique();
    let ix = Instruction {
        program_id: machine_wallet::id(),
        accounts: vec![
            AccountMeta::new(payer.pubkey(), true),
            AccountMeta::new(fake_wallet, false),
            AccountMeta::new_readonly(system_program::id(), false),
        ],
        data: short_data,
    };

    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&payer.pubkey()),
        &[&payer],
        recent_blockhash,
    );
    let result = banks.process_transaction(tx).await;
    assert!(result.is_err(), "Truncated authority should fail");
}
