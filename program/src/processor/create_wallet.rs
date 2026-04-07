use solana_program::{
    account_info::{next_account_info, AccountInfo},
    clock::Clock,
    entrypoint::ProgramResult,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::Sysvar,
};

use crate::{
    error::MachineWalletError,
    processor::init_pda_account::init_pda_account,
    state::{
        AuthoritySlot, MachineWallet, SIG_SCHEME_ED25519, SIG_SCHEME_SECP256R1,
    },
};

/// Create a new MachineWallet for the given authority.
///
/// NOTE: Permissionless — any payer can create a wallet for any valid pubkey.
/// This is safe because only the private key holder can execute, close, or
/// advance nonce. The payer only pays rent. Front-running creation does not grant
/// the attacker any control over the wallet.
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
    sig_scheme: u8,
    authority: [u8; 33],
) -> ProgramResult {
    let account_iter = &mut accounts.iter();

    let payer = next_account_info(account_iter)?;
    let wallet_account = next_account_info(account_iter)?;
    let system_program = next_account_info(account_iter)?;

    // 1. Validate payer is signer
    if !payer.is_signer {
        return Err(ProgramError::MissingRequiredSignature);
    }

    // 2a. System instructions below require writable destination accounts.
    if !wallet_account.is_writable {
        return Err(MachineWalletError::AccountNotWritable.into());
    }

    // 3. Validate authority using AuthoritySlot (supports both Secp256r1 and Ed25519).
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

    // 4. Compute wallet ID: keccak256(authority) — stored permanently for stable PDA
    let id = MachineWallet::compute_id(&authority);

    // 5. Derive wallet PDA and verify
    let (expected_wallet_pda, wallet_bump) =
        Pubkey::find_program_address(&[MachineWallet::SEED_PREFIX, &id], program_id);
    if *wallet_account.key != expected_wallet_pda {
        return Err(MachineWalletError::InvalidWalletPDA.into());
    }

    // 6. Derive vault PDA bump (vault is not created — stays system-owned)
    let (_, vault_bump) = Pubkey::find_program_address(
        &[
            MachineWallet::VAULT_SEED_PREFIX,
            expected_wallet_pda.as_ref(),
        ],
        program_id,
    );

    // 7. Initialize wallet PDA in a dust-resistant way.
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

    // 8. Initialize wallet state as v1 layout (including cached vault_bump)
    let clock = Clock::get()?;
    let mut authorities = [AuthoritySlot::EMPTY; crate::state::MAX_AUTHORITIES as usize];
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
