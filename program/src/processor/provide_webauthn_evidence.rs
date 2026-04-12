use solana_program::{
    account_info::AccountInfo,
    entrypoint::ProgramResult,
    instruction::{get_stack_height, TRANSACTION_LEVEL_STACK_HEIGHT},
    pubkey::Pubkey,
};

use crate::error::MachineWalletError;

/// Sidecar for `ProvideWebAuthnEvidence`: parks authenticatorData +
/// clientDataJSON in the tx so a consuming wallet instruction can match them
/// against a WEBAUTHN authority via `load_instruction_at_checked`.
///
/// Intentionally stateless — no accounts, no mutation, no crypto. Structural
/// validation (length bounds) happens in `parse_webauthn_evidence_payload`;
/// semantic validation (rpIdHash, UP/UV, type, challenge binding) happens in
/// `threshold::verify_threshold_signatures`, which knows the consuming op's
/// expected hash.
///
/// Must run at top-level (like a precompile): CPI invocation is rejected to
/// match precompile placement requirements.
pub fn process(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo],
    _auth_data: &[u8],
    _client_data_json: &[u8],
) -> ProgramResult {
    if get_stack_height() > TRANSACTION_LEVEL_STACK_HEIGHT {
        return Err(MachineWalletError::CpiReentryDenied.into());
    }
    Ok(())
}
