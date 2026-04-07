pub mod add_authority;
pub mod advance_nonce;
pub mod close_session;
pub mod close_wallet;
pub mod create_session;
pub mod create_wallet;
pub mod execute;
pub(crate) mod init_pda_account;
pub mod owner_close_session;
pub mod remove_authority;
pub mod revoke_session;
pub mod self_revoke_session;
pub mod session_execute;
pub mod set_threshold;

use solana_program::{account_info::AccountInfo, entrypoint::ProgramResult, pubkey::Pubkey};

use crate::instruction::MachineWalletInstruction;

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = MachineWalletInstruction::unpack(instruction_data)?;

    match instruction {
        MachineWalletInstruction::CreateWallet {
            sig_scheme,
            authority,
        } => create_wallet::process(program_id, accounts, sig_scheme, authority),
        MachineWalletInstruction::Execute {
            secp256r1_ix_index,
            max_slot,
            inner_instructions,
        } => execute::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            inner_instructions,
        ),
        MachineWalletInstruction::CloseWallet {
            secp256r1_ix_index,
            max_slot,
            destination,
        } => close_wallet::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            destination,
        ),
        MachineWalletInstruction::AdvanceNonce {
            secp256r1_ix_index,
            max_slot,
        } => advance_nonce::process(program_id, accounts, secp256r1_ix_index, max_slot),
        MachineWalletInstruction::CreateSession {
            secp256r1_ix_index,
            max_slot,
            session_authority,
            expiry_slot,
            max_lamports_per_ix,
            allowed_programs_count,
            allowed_programs,
        } => create_session::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            session_authority,
            expiry_slot,
            max_lamports_per_ix,
            allowed_programs_count,
            allowed_programs,
        ),
        MachineWalletInstruction::SessionExecute { inner_instructions } => {
            session_execute::process(program_id, accounts, inner_instructions)
        }
        MachineWalletInstruction::RevokeSession {
            secp256r1_ix_index,
            max_slot,
            session_authority,
        } => revoke_session::process(
            program_id,
            accounts,
            secp256r1_ix_index,
            max_slot,
            session_authority,
        ),
        MachineWalletInstruction::SelfRevokeSession => {
            self_revoke_session::process(program_id, accounts)
        }
        MachineWalletInstruction::CloseSession => close_session::process(program_id, accounts),
        MachineWalletInstruction::AddAuthority {
            precompile_ix_index,
            new_sig_scheme,
            new_pubkey,
            new_threshold,
            max_slot,
        } => add_authority::process(
            program_id,
            accounts,
            precompile_ix_index,
            new_sig_scheme,
            new_pubkey,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::RemoveAuthority {
            precompile_ix_index,
            remove_sig_scheme,
            remove_pubkey,
            new_threshold,
            max_slot,
        } => remove_authority::process(
            program_id,
            accounts,
            precompile_ix_index,
            remove_sig_scheme,
            remove_pubkey,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::SetThreshold {
            precompile_ix_index,
            new_threshold,
            max_slot,
        } => set_threshold::process(
            program_id,
            accounts,
            precompile_ix_index,
            new_threshold,
            max_slot,
        ),
        MachineWalletInstruction::OwnerCloseSession {
            precompile_ix_index,
            max_slot,
            session_authority,
            destination,
        } => owner_close_session::process(
            program_id,
            accounts,
            precompile_ix_index,
            max_slot,
            session_authority,
            destination,
        ),
    }
}
