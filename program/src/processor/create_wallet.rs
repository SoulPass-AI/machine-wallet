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
    state::{AuthoritySlot, MachineWallet, MAX_AUTHORITIES, SIG_SCHEME_ED25519, SIG_SCHEME_SECP256R1},
    threshold,
};

/// Domain separator for CreateWallet messages.
const CREATE_WALLET_TAG: &[u8] = b"machine_wallet_create_wallet_v0";

/// Compute create-wallet message:
/// keccak256(CREATE_WALLET_TAG || wallet_pda || max_slot || sig_scheme || authority)
///
/// The wallet PDA remains derived only from `authority` so the address is stable
/// across future authority/scheme rotation. `sig_scheme` is still signed here so
/// a WebAuthn public key cannot be front-run into a raw Secp256r1 wallet.
pub fn compute_create_wallet_message(
    wallet_address: &Pubkey,
    max_slot: u64,
    sig_scheme: u8,
    authority: &[u8; 33],
) -> [u8; 32] {
    let max_slot_bytes = max_slot.to_le_bytes();
    keccak::hashv(&[
        CREATE_WALLET_TAG,
        wallet_address.as_ref(),
        &max_slot_bytes,
        &[sig_scheme],
        authority,
    ])
    .to_bytes()
}

/// Verify the initial authority self-signed this exact create message.
///
/// CreateWallet has no stored authority set yet — the proof is that whoever
/// claims to be `authority_slot` *also* produces a signature here. Routed
/// through the unified `verify_threshold_signatures` entry point (single audit
/// surface) with a 1-element slice; the threshold scanner enforces scheme-
/// strict matching, at-most-once counting, and (for WEBAUTHN) sidecar binding.
///
/// Passing a 1-element slice instead of a full `[EMPTY; MAX_AUTHORITIES]` array
/// drops ~544 B of stack at the call site.
fn verify_create_authority_proof(
    instructions_sysvar: &AccountInfo,
    program_id: &Pubkey,
    authority_slot: AuthoritySlot,
    expected_message: &[u8; 32],
) -> ProgramResult {
    let authorities = [authority_slot];
    threshold::verify_threshold_signatures(
        instructions_sysvar,
        program_id,
        &authorities,
        1,
        1,
        expected_message,
    )
}

/// Create a new MachineWallet for the given authority.
///
/// Any payer can sponsor the rent, but creation must include a scheme-specific
/// proof from the initial authority. This prevents a known WebAuthn P-256 public
/// key from being front-run into a raw Secp256r1 wallet at the same PDA.
///
/// The vault PDA is NOT created here — it remains system-owned so that
/// `system_program::transfer` CPI works naturally from Execute. The vault_bump
/// is computed and cached in wallet state for later `invoke_signed` calls.
///
/// Creates v1 layout directly (version=1, single authority). The account size
/// for 1 authority (87 bytes) is identical to the old v0 layout.
pub fn process(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    max_slot: u64,
    sig_scheme: u8,
    authority: [u8; 33],
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let instructions_sysvar = next_account_info(account_iter)?;
    let payer = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let system_program = next_account_info(account_iter)?;

    // 1. Validate payer is signer
    if !payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    // 2. System instructions below require writable destination accounts.
    if !wallet_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }
    super::require_instructions_sysvar(instructions_sysvar)?;

    // 3. Create must be top-level. Signature evidence lives in the transaction
    // instruction list and precompiles cannot be safely supplied over CPI.
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }

    // 4. Validate authority using AuthoritySlot (supports Secp256r1, Ed25519,
    // and WebAuthn).
    let authority_slot = AuthoritySlot {
        sig_scheme,
        pubkey: authority,
    };
    if !authority_slot.is_valid() {
        return match sig_scheme {
            SIG_SCHEME_ED25519 => {
                Err(crate::error::MachineWalletError::InvalidEd25519Pubkey.into())
            }
            SIG_SCHEME_SECP256R1 => Err(ProgramError::InvalidInstructionData),
            _ => Err(ProgramError::InvalidInstructionData),
        };
    }

    // 5. Compute wallet ID: keccak256(authority) — stored permanently for stable PDA
    let id = MachineWallet::compute_id(&authority);

    // 6. Derive wallet PDA and verify
    let (expected_wallet_pda, wallet_bump) =
        Pubkey::find_program_address(&[MachineWallet::SEED_PREFIX, &id], program_id);
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 7. Signature expiry
    let clock = Clock::get()?;
    if clock.slot > max_slot {
        return Err(MachineWalletError::SignatureExpired.into());
    }

    // 8. Verify the initial authority approved this exact create scheme.
    let expected_message =
        compute_create_wallet_message(wallet_account.key, max_slot, sig_scheme, &authority);
    verify_create_authority_proof(
        instructions_sysvar,
        program_id,
        authority_slot,
        &expected_message,
    )?;

    // 9. Derive vault PDA bump (vault is not created — stays system-owned)
    let (_, vault_bump) = Pubkey::find_program_address(
        &[
            MachineWallet::VAULT_SEED_PREFIX,
            expected_wallet_pda.as_ref(),
        ],
        program_id,
    );

    // 10. Initialize wallet PDA in a dust-resistant way.
    let account_size = MachineWallet::v1_account_size(1);
    let wallet_signer_seeds: &[&[u8]] = &[MachineWallet::SEED_PREFIX, &id, &[wallet_bump]];
    init_pda_account(
        payer,
        wallet_account,
        system_program,
        program_id,
        account_size,
        wallet_signer_seeds,
        MachineWalletError::WalletAlreadyInitialized,
    )?;

    // 11. Initialize wallet state as v1 layout (including cached vault_bump)
    let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
    authorities[0] = authority_slot;
    let wallet = MachineWallet {
        version: 1,
        bump: wallet_bump,
        wallet_id: id,
        threshold: 1,
        authority_count: 1,
        authorities,
        nonce: 0,
        creation_slot: clock.slot,
        vault_bump,
    };

    let mut data = wallet_account.try_borrow_mut_data()?;
    wallet.serialize_v1(&mut data)?;

    Ok(())
}
