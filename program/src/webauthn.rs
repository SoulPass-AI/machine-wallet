use solana_program::program_error::ProgramError;

use crate::error::MachineWalletError;

/// Maximum authenticatorData size. Minimum is 37 (rpIdHash 32 + flags 1 +
/// signCount 4). Upper bound sized to cover assertions with practical extensions
/// such as `largeBlob` read outputs and `prf` eval results, plus margin.
pub const MAX_AUTH_DATA_SIZE: usize = 512;
/// Minimum authenticatorData size (rpIdHash 32 + flags 1 + signCount 4).
pub const MIN_AUTH_DATA_SIZE: usize = 37;
/// Offset of the flags byte inside authenticatorData (after the 32-byte rpIdHash).
const AUTH_DATA_FLAGS_OFFSET: usize = 32;
/// Relying Party identifier for all SoulPass passkeys.
/// rpIdHash is the SHA-256 of this string and occupies authenticatorData[0..32].
pub const EXPECTED_RP_ID: &[u8] = b"soulpass.ai";
/// Pre-computed SHA-256(EXPECTED_RP_ID). Binding every WebAuthn assertion to this
/// hash prevents a credential registered under a different Relying Party (e.g. a
/// sibling subdomain or a future RP rotation) from authorizing wallet execution.
/// A startup test asserts this matches `hashv(&[EXPECTED_RP_ID])`.
pub const EXPECTED_RP_ID_HASH: [u8; 32] = [
    0x29, 0x03, 0x6d, 0x20, 0x68, 0xf7, 0xac, 0x62, 0xbe, 0x89, 0x7f, 0x31, 0x01, 0x5f, 0x3a, 0xda,
    0xce, 0x3f, 0x43, 0x8a, 0x1f, 0x4d, 0xb6, 0xbe, 0x25, 0xa7, 0x27, 0x76, 0xf2, 0xa4, 0xe3, 0x15,
];
/// User Present (bit 0) and User Verified (bit 2) bits of the flags byte.
/// Passkey assertions must have both set: UP proves a human interacted with the
/// authenticator, UV proves biometric/PIN verification succeeded. Platform
/// authenticators (Touch ID, Face ID, Windows Hello) set both by default.
const FLAG_USER_PRESENT: u8 = 0x01;
const FLAG_USER_VERIFIED: u8 = 0x04;
/// Length of a base64url-no-pad-encoded 32-byte challenge (ceil(32 * 8 / 6) = 43).
pub const CHALLENGE_B64_LEN: usize = 43;

/// Base64url-no-pad encoding of a 32-byte operation hash for use as the
/// WebAuthn challenge. Encode once per verification; reuse across sidecars.
pub fn encode_challenge(challenge: &[u8; 32]) -> [u8; CHALLENGE_B64_LEN] {
    base64url_encode_no_pad(challenge)
}

/// Verify authenticatorData length, rpIdHash, and UP/UV flags. The threshold
/// scanner calls this on the auth_data prefix it reads directly from the
/// secp256r1 precompile message (the compact-evidence path).
pub fn verify_auth_data(auth_data: &[u8]) -> Result<(), ProgramError> {
    if auth_data.len() < MIN_AUTH_DATA_SIZE || auth_data.len() > MAX_AUTH_DATA_SIZE {
        return Err(MachineWalletError::InvalidWebAuthnAuthData.into());
    }
    // rpIdHash binding: enforces that the assertion was produced against
    // EXPECTED_RP_ID. Without this check, any credential registered on any RP
    // whose user signs our challenge bytes could authorize a wallet — passkey
    // hardware scoping normally prevents this, but a deployment-level RP change
    // (subdomain takeover, rp_id rotation) would silently be accepted.
    if auth_data[..32] != EXPECTED_RP_ID_HASH {
        return Err(MachineWalletError::WebAuthnRpIdMismatch.into());
    }
    // UP and UV are cryptographically signed attestations of user interaction
    // and identity verification. Both are required for passkey authentication.
    let flags = auth_data[AUTH_DATA_FLAGS_OFFSET];
    if flags & FLAG_USER_PRESENT == 0 {
        return Err(MachineWalletError::WebAuthnUserNotPresent.into());
    }
    if flags & FLAG_USER_VERIFIED == 0 {
        return Err(MachineWalletError::WebAuthnUserNotVerified.into());
    }
    Ok(())
}

/// Base64url encode without padding (RFC 4648 section 5).
/// Input: 32 bytes. Output: 43 base64url chars (no padding).
fn base64url_encode_no_pad(data: &[u8; 32]) -> [u8; CHALLENGE_B64_LEN] {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = [0u8; CHALLENGE_B64_LEN];
    let mut oi = 0;
    let mut i = 0;

    // Process full 3-byte groups (10 groups = 30 bytes)
    while i + 2 < data.len() {
        let b0 = data[i] as usize;
        let b1 = data[i + 1] as usize;
        let b2 = data[i + 2] as usize;
        out[oi] = TABLE[(b0 >> 2) & 0x3F];
        out[oi + 1] = TABLE[((b0 << 4) | (b1 >> 4)) & 0x3F];
        out[oi + 2] = TABLE[((b1 << 2) | (b2 >> 6)) & 0x3F];
        out[oi + 3] = TABLE[b2 & 0x3F];
        oi += 4;
        i += 3;
    }

    // Remaining 2 bytes (32 = 10*3 + 2): produces 3 chars, no padding.
    let b0 = data[i] as usize;
    let b1 = data[i + 1] as usize;
    out[oi] = TABLE[(b0 >> 2) & 0x3F];
    out[oi + 1] = TABLE[((b0 << 4) | (b1 >> 4)) & 0x3F];
    out[oi + 2] = TABLE[(b1 << 2) & 0x3F];

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::hash::hashv;

    /// Build a minimal-size authenticatorData with valid rpIdHash, UP=1 and UV=1.
    fn make_auth_data<const N: usize>() -> [u8; N] {
        let mut data = [0xAAu8; N];
        data[..32].copy_from_slice(&EXPECTED_RP_ID_HASH);
        data[AUTH_DATA_FLAGS_OFFSET] = FLAG_USER_PRESENT | FLAG_USER_VERIFIED;
        data
    }

    #[test]
    fn test_base64url_encode_no_pad() {
        let input = [0u8; 32];
        let result = base64url_encode_no_pad(&input);
        let encoded = core::str::from_utf8(&result).unwrap();
        assert_eq!(encoded, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");

        let input = [0xFFu8; 32];
        let result = base64url_encode_no_pad(&input);
        let encoded = core::str::from_utf8(&result).unwrap();
        assert_eq!(encoded, "__________________________________________8");
    }

    #[test]
    fn test_verify_auth_data_valid() {
        let auth_data: [u8; 37] = make_auth_data();
        assert!(verify_auth_data(&auth_data).is_ok());
    }

    #[test]
    fn test_verify_auth_data_too_short() {
        let auth_data = [0xAAu8; MIN_AUTH_DATA_SIZE - 1];
        let err = verify_auth_data(&auth_data).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::InvalidWebAuthnAuthData as u32)
        );
    }

    #[test]
    fn test_verify_auth_data_too_long() {
        let auth_data = vec![0xAAu8; MAX_AUTH_DATA_SIZE + 1];
        let err = verify_auth_data(&auth_data).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::InvalidWebAuthnAuthData as u32)
        );
    }

    #[test]
    fn test_verify_auth_data_wrong_rpid() {
        let mut auth_data: [u8; 37] = make_auth_data();
        auth_data[0] ^= 0x01; // corrupt rpIdHash
        let err = verify_auth_data(&auth_data).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnRpIdMismatch as u32)
        );
    }

    #[test]
    fn test_verify_auth_data_up_clear() {
        let mut auth_data: [u8; 37] = make_auth_data();
        auth_data[AUTH_DATA_FLAGS_OFFSET] = FLAG_USER_VERIFIED; // UV=1, UP=0
        let err = verify_auth_data(&auth_data).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnUserNotPresent as u32)
        );
    }

    #[test]
    fn test_verify_auth_data_uv_clear() {
        let mut auth_data: [u8; 37] = make_auth_data();
        auth_data[AUTH_DATA_FLAGS_OFFSET] = FLAG_USER_PRESENT; // UP=1, UV=0
        let err = verify_auth_data(&auth_data).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnUserNotVerified as u32)
        );
    }

    #[test]
    fn test_verify_auth_data_max_size() {
        let mut auth_data = vec![0xAAu8; MAX_AUTH_DATA_SIZE];
        auth_data[..32].copy_from_slice(&EXPECTED_RP_ID_HASH);
        auth_data[AUTH_DATA_FLAGS_OFFSET] = FLAG_USER_PRESENT | FLAG_USER_VERIFIED;
        assert!(verify_auth_data(&auth_data).is_ok());
    }

    /// The hard-coded EXPECTED_RP_ID_HASH must equal SHA-256(EXPECTED_RP_ID).
    /// If either constant drifts the whole WebAuthn path breaks silently, so we
    /// lock them together with a runtime check.
    #[test]
    fn test_expected_rp_id_hash_matches() {
        let computed = hashv(&[EXPECTED_RP_ID]);
        assert_eq!(computed.as_ref(), &EXPECTED_RP_ID_HASH);
    }
}
