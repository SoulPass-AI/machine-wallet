#![allow(unexpected_cfgs)]

solana_program::declare_id!("7VD7mx5bYgmSJY7D1etvADEdDXijdp3UMz79M53vTdMo");

pub mod ed25519;
pub mod error;
pub mod instruction;
pub mod processor;
pub mod secp256r1;
pub mod state;
pub mod threshold;
pub mod webauthn;

solana_program::entrypoint!(process_instruction);

pub fn process_instruction(
    program_id: &solana_program::pubkey::Pubkey,
    accounts: &[solana_program::account_info::AccountInfo],
    instruction_data: &[u8],
) -> solana_program::entrypoint::ProgramResult {
    processor::process_instruction(program_id, accounts, instruction_data)
}
