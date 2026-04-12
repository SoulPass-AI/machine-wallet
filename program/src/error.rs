use solana_program::program_error::ProgramError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineWalletError {
    InvalidPrecompileInstruction = 0,
    PublicKeyMismatch = 1,
    MessageMismatch = 2,
    InvalidNonce = 3,
    InvalidSignatureOffsets = 4,
    WalletAlreadyInitialized = 5,
    WalletNotInitialized = 6,
    InvalidVaultPDA = 7,
    InvalidWalletPDA = 8,
    InstructionMissing = 9,
    SignatureExpired = 10,
    CpiReentryDenied = 11,
    CpiToSelfDenied = 12,
    VaultOwnerMismatch = 13,
    TooManyInnerInstructions = 14,
    MissingProgramAccount = 15,
    InvalidDestination = 16,
    InvalidVaultOwner = 17,
    AccountNotWritable = 18,
    SessionExpired = 19,
    SessionRevoked = 20,
    SessionAuthorityMismatch = 21,
    SessionWalletMismatch = 22,
    ProgramNotAllowed = 23,
    SessionAlreadyExists = 24,
    InvalidSessionPDA = 25,
    IxAmountExceeded = 26,
    InvalidSessionData = 27,
    TooManyAllowedPrograms = 28,
    SessionStillActive = 29,
    InsufficientSignatures = 30,
    AuthorityLimitExceeded = 31,
    DuplicateAuthority = 32,
    CannotRemoveLastAuthority = 33,
    InvalidThreshold = 34,
    AuthorityNotFound = 35,
    InvalidEd25519Pubkey = 36,
    InvalidWebAuthnAuthData = 40,
    InvalidWebAuthnClientDataJson = 41,
    WebAuthnChallengeMismatch = 42,
    WebAuthnInvalidType = 43,
    WebAuthnUserNotPresent = 44,
    WebAuthnDuplicateField = 45,
    WebAuthnUserNotVerified = 46,
    WebAuthnRpIdMismatch = 47,
    SessionSpendCapExceeded = 48,
    TooManyWebAuthnEvidence = 50,
}

impl From<MachineWalletError> for ProgramError {
    fn from(e: MachineWalletError) -> Self {
        ProgramError::Custom(e as u32)
    }
}

impl std::fmt::Display for MachineWalletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPrecompileInstruction => write!(f, "Invalid precompile instruction"),
            Self::PublicKeyMismatch => write!(f, "Public key mismatch"),
            Self::MessageMismatch => write!(f, "Message mismatch"),
            Self::InvalidNonce => write!(f, "Invalid nonce"),
            Self::InvalidSignatureOffsets => write!(f, "Invalid signature offsets"),
            Self::WalletAlreadyInitialized => write!(f, "Wallet already initialized"),
            Self::WalletNotInitialized => write!(f, "Wallet not initialized"),
            Self::InvalidVaultPDA => write!(f, "Invalid vault PDA"),
            Self::InvalidWalletPDA => write!(f, "Invalid wallet PDA"),
            Self::InstructionMissing => write!(f, "Instruction missing"),
            Self::SignatureExpired => write!(f, "Signature expired"),
            Self::CpiReentryDenied => write!(f, "CPI reentry denied"),
            Self::CpiToSelfDenied => write!(f, "CPI to self denied"),
            Self::VaultOwnerMismatch => write!(
                f,
                "Vault owner mismatch (unused, kept for discriminant stability)"
            ),
            Self::TooManyInnerInstructions => write!(f, "Too many inner instructions"),
            Self::MissingProgramAccount => write!(f, "Missing program account for CPI"),
            Self::InvalidDestination => write!(f, "Invalid destination account"),
            Self::InvalidVaultOwner => write!(f, "Vault account not owned by System Program"),
            Self::AccountNotWritable => write!(f, "Account must be writable"),
            Self::SessionExpired => write!(f, "Session has expired"),
            Self::SessionRevoked => write!(f, "Session has been revoked"),
            Self::SessionAuthorityMismatch => write!(f, "Session authority mismatch"),
            Self::SessionWalletMismatch => write!(f, "Session wallet mismatch"),
            Self::ProgramNotAllowed => write!(f, "Program not in session whitelist"),
            Self::SessionAlreadyExists => write!(f, "Session already exists"),
            Self::InvalidSessionPDA => write!(f, "Invalid session PDA"),
            Self::IxAmountExceeded => write!(f, "Instruction amount exceeds session limit"),
            Self::InvalidSessionData => write!(f, "Invalid session data"),
            Self::TooManyAllowedPrograms => write!(f, "Too many allowed programs"),
            Self::SessionStillActive => {
                write!(f, "Session is still active (not expired or revoked)")
            }
            Self::InsufficientSignatures => write!(f, "Matched signatures below threshold"),
            Self::AuthorityLimitExceeded => write!(f, "Authority count would exceed maximum"),
            Self::DuplicateAuthority => write!(f, "Authority already exists"),
            Self::CannotRemoveLastAuthority => write!(f, "Cannot remove last authority"),
            Self::InvalidThreshold => write!(f, "Invalid threshold value"),
            Self::AuthorityNotFound => write!(f, "Authority not found"),
            Self::InvalidEd25519Pubkey => write!(f, "Invalid Ed25519 public key"),
            Self::InvalidWebAuthnAuthData => write!(f, "Invalid WebAuthn authenticatorData"),
            Self::InvalidWebAuthnClientDataJson => write!(f, "Invalid WebAuthn clientDataJSON"),
            Self::WebAuthnChallengeMismatch => write!(f, "WebAuthn challenge mismatch"),
            Self::WebAuthnInvalidType => write!(f, "WebAuthn type must be webauthn.get"),
            Self::WebAuthnUserNotPresent => {
                write!(f, "WebAuthn authenticatorData UP flag must be set")
            }
            Self::WebAuthnDuplicateField => {
                write!(f, "WebAuthn clientDataJSON contains duplicate key")
            }
            Self::WebAuthnUserNotVerified => {
                write!(f, "WebAuthn authenticatorData UV flag must be set")
            }
            Self::WebAuthnRpIdMismatch => {
                write!(f, "WebAuthn rpIdHash does not match expected RP")
            }
            Self::SessionSpendCapExceeded => {
                write!(f, "Session cumulative spend cap exceeded")
            }
            Self::TooManyWebAuthnEvidence => {
                write!(f, "Too many ProvideWebAuthnEvidence sidecar instructions")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_to_program_error() {
        let err: ProgramError = MachineWalletError::InvalidPrecompileInstruction.into();
        assert_eq!(err, ProgramError::Custom(0));

        let err: ProgramError = MachineWalletError::InstructionMissing.into();
        assert_eq!(err, ProgramError::Custom(9));
    }

    #[test]
    fn test_error_discriminants() {
        assert_eq!(MachineWalletError::InvalidPrecompileInstruction as u32, 0);
        assert_eq!(MachineWalletError::PublicKeyMismatch as u32, 1);
        assert_eq!(MachineWalletError::MessageMismatch as u32, 2);
        assert_eq!(MachineWalletError::InvalidNonce as u32, 3);
        assert_eq!(MachineWalletError::InvalidSignatureOffsets as u32, 4);
        assert_eq!(MachineWalletError::WalletAlreadyInitialized as u32, 5);
        assert_eq!(MachineWalletError::WalletNotInitialized as u32, 6);
        assert_eq!(MachineWalletError::InvalidVaultPDA as u32, 7);
        assert_eq!(MachineWalletError::InvalidWalletPDA as u32, 8);
        assert_eq!(MachineWalletError::InstructionMissing as u32, 9);
        assert_eq!(MachineWalletError::SignatureExpired as u32, 10);
        assert_eq!(MachineWalletError::CpiReentryDenied as u32, 11);
        assert_eq!(MachineWalletError::CpiToSelfDenied as u32, 12);
        assert_eq!(MachineWalletError::VaultOwnerMismatch as u32, 13);
        assert_eq!(MachineWalletError::TooManyInnerInstructions as u32, 14);
        assert_eq!(MachineWalletError::MissingProgramAccount as u32, 15);
        assert_eq!(MachineWalletError::InvalidDestination as u32, 16);
        assert_eq!(MachineWalletError::InvalidVaultOwner as u32, 17);
        assert_eq!(MachineWalletError::AccountNotWritable as u32, 18);
        assert_eq!(MachineWalletError::SessionExpired as u32, 19);
        assert_eq!(MachineWalletError::SessionRevoked as u32, 20);
        assert_eq!(MachineWalletError::SessionAuthorityMismatch as u32, 21);
        assert_eq!(MachineWalletError::SessionWalletMismatch as u32, 22);
        assert_eq!(MachineWalletError::ProgramNotAllowed as u32, 23);
        assert_eq!(MachineWalletError::SessionAlreadyExists as u32, 24);
        assert_eq!(MachineWalletError::InvalidSessionPDA as u32, 25);
        assert_eq!(MachineWalletError::IxAmountExceeded as u32, 26);
        assert_eq!(MachineWalletError::InvalidSessionData as u32, 27);
        assert_eq!(MachineWalletError::TooManyAllowedPrograms as u32, 28);
        assert_eq!(MachineWalletError::SessionStillActive as u32, 29);
        assert_eq!(MachineWalletError::InsufficientSignatures as u32, 30);
        assert_eq!(MachineWalletError::AuthorityLimitExceeded as u32, 31);
        assert_eq!(MachineWalletError::DuplicateAuthority as u32, 32);
        assert_eq!(MachineWalletError::CannotRemoveLastAuthority as u32, 33);
        assert_eq!(MachineWalletError::InvalidThreshold as u32, 34);
        assert_eq!(MachineWalletError::AuthorityNotFound as u32, 35);
        assert_eq!(MachineWalletError::InvalidEd25519Pubkey as u32, 36);
        assert_eq!(MachineWalletError::InvalidWebAuthnAuthData as u32, 40);
        assert_eq!(MachineWalletError::InvalidWebAuthnClientDataJson as u32, 41);
        assert_eq!(MachineWalletError::WebAuthnChallengeMismatch as u32, 42);
        assert_eq!(MachineWalletError::WebAuthnInvalidType as u32, 43);
        assert_eq!(MachineWalletError::WebAuthnUserNotPresent as u32, 44);
        assert_eq!(MachineWalletError::WebAuthnDuplicateField as u32, 45);
        assert_eq!(MachineWalletError::WebAuthnUserNotVerified as u32, 46);
        assert_eq!(MachineWalletError::WebAuthnRpIdMismatch as u32, 47);
        assert_eq!(MachineWalletError::SessionSpendCapExceeded as u32, 48);
        assert_eq!(MachineWalletError::TooManyWebAuthnEvidence as u32, 50);
    }
}
