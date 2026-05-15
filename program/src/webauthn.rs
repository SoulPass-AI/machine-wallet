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
/// Maximum clientDataJSON size accepted by the on-chain sidecar parser.
pub const MAX_CLIENT_DATA_JSON_SIZE: usize = 1024;

/// Base64url-no-pad encoding of a 32-byte operation hash for use as the
/// WebAuthn challenge. Encode once per verification; reuse across sidecars.
pub fn encode_challenge(challenge: &[u8; 32]) -> [u8; CHALLENGE_B64_LEN] {
    base64url_encode_no_pad(challenge)
}

/// Verify authenticatorData length, rpIdHash, and UP/UV flags. The threshold
/// scanner calls this on the auth_data prefix it reads directly from the
/// secp256r1 precompile message.
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

/// Verify clientDataJSON type/challenge and return its SHA-256 digest.
///
/// The sidecar carries clientDataJSON, while the secp256r1 precompile message
/// carries only SHA256(clientDataJSON). Re-hashing here cryptographically binds
/// the parsed challenge to the exact bytes signed by the authenticator.
pub fn verify_client_data_json(
    client_data_json: &[u8],
    expected_challenge_b64: &[u8; CHALLENGE_B64_LEN],
) -> Result<[u8; 32], ProgramError> {
    if client_data_json.is_empty() || client_data_json.len() > MAX_CLIENT_DATA_JSON_SIZE {
        return Err(MachineWalletError::InvalidWebAuthnClientDataJson.into());
    }

    let (auth_type, challenge) = extract_type_and_challenge(client_data_json)?
        .ok_or(MachineWalletError::InvalidWebAuthnClientDataJson)?;
    if auth_type != b"webauthn.get" {
        return Err(MachineWalletError::WebAuthnInvalidType.into());
    }
    if challenge != expected_challenge_b64 {
        return Err(MachineWalletError::WebAuthnChallengeMismatch.into());
    }

    Ok(solana_program::hash::hashv(&[client_data_json])
        .as_ref()
        .try_into()
        .expect("SHA-256 hash is 32 bytes"))
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

/// Extract top-level "type" and "challenge" string fields from clientDataJSON.
///
/// Canonical policy: keys and wanted values reject backslash escapes, so a
/// byte-equal challenge comparison cannot be confused by alternate JSON
/// spellings. Other fields are skipped with string-escape awareness.
fn extract_type_and_challenge(json: &[u8]) -> Result<Option<(&[u8], &[u8])>, ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);

    let mut cursor = 0;
    skip_ws(json, &mut cursor);
    if cursor >= json.len() || json[cursor] != b'{' {
        return Err(err());
    }
    cursor += 1;

    let mut auth_type: Option<&[u8]> = None;
    let mut challenge: Option<&[u8]> = None;

    loop {
        skip_ws(json, &mut cursor);
        if cursor >= json.len() {
            return Err(err());
        }
        if json[cursor] == b'}' {
            cursor += 1;
            skip_ws(json, &mut cursor);
            if cursor != json.len() {
                return Err(err());
            }
            break;
        }

        let (key, next) = read_canonical_string(json, cursor)?;
        cursor = next;
        skip_ws(json, &mut cursor);
        if cursor >= json.len() || json[cursor] != b':' {
            return Err(err());
        }
        cursor += 1;
        skip_ws(json, &mut cursor);
        if cursor >= json.len() {
            return Err(err());
        }

        let want = matches!(key, b"type" | b"challenge");
        if want {
            let (value, next) = read_canonical_string(json, cursor)?;
            let slot = if key == b"type" {
                &mut auth_type
            } else {
                &mut challenge
            };
            if slot.is_some() {
                return Err(MachineWalletError::WebAuthnDuplicateField.into());
            }
            *slot = Some(value);
            cursor = next;
        } else {
            cursor = skip_json_value(json, cursor)?;
        }

        skip_ws(json, &mut cursor);
        if cursor >= json.len() {
            return Err(err());
        }
        match json[cursor] {
            b',' => {
                cursor += 1;
            }
            b'}' => {
                cursor += 1;
                skip_ws(json, &mut cursor);
                if cursor != json.len() {
                    return Err(err());
                }
                break;
            }
            _ => return Err(err()),
        }
    }

    Ok(match (auth_type, challenge) {
        (Some(t), Some(c)) => Some((t, c)),
        _ => None,
    })
}

fn skip_ws(json: &[u8], cursor: &mut usize) {
    while *cursor < json.len() && json[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
}

fn read_canonical_string(json: &[u8], cursor: usize) -> Result<(&[u8], usize), ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);
    if cursor >= json.len() || json[cursor] != b'"' {
        return Err(err());
    }
    let value_start = cursor + 1;
    let mut value_end = value_start;
    while value_end < json.len() {
        let b = json[value_end];
        if b == b'"' {
            return Ok((&json[value_start..value_end], value_end + 1));
        }
        if b == b'\\' {
            return Err(err());
        }
        value_end += 1;
    }
    Err(err())
}

fn skip_json_value(json: &[u8], mut cursor: usize) -> Result<usize, ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);
    skip_ws(json, &mut cursor);
    if cursor >= json.len() {
        return Err(err());
    }

    match json[cursor] {
        b'"' => skip_string_body(json, cursor),
        b'{' | b'[' => skip_json_container(json, cursor),
        b't' if json.get(cursor..cursor + 4) == Some(b"true") => Ok(cursor + 4),
        b'f' if json.get(cursor..cursor + 5) == Some(b"false") => Ok(cursor + 5),
        b'n' if json.get(cursor..cursor + 4) == Some(b"null") => Ok(cursor + 4),
        b'-' | b'0'..=b'9' => skip_json_number(json, cursor),
        _ => Err(err()),
    }
}

fn skip_json_container(json: &[u8], mut cursor: usize) -> Result<usize, ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);
    let mut stack = [0u8; 16];
    let mut depth = 0usize;

    loop {
        if cursor >= json.len() {
            return Err(err());
        }
        match json[cursor] {
            b'"' => cursor = skip_string_body(json, cursor)?,
            b'{' => {
                if depth >= stack.len() {
                    return Err(err());
                }
                stack[depth] = b'}';
                depth += 1;
                cursor += 1;
            }
            b'[' => {
                if depth >= stack.len() {
                    return Err(err());
                }
                stack[depth] = b']';
                depth += 1;
                cursor += 1;
            }
            b'}' | b']' => {
                if depth == 0 || json[cursor] != stack[depth - 1] {
                    return Err(err());
                }
                depth -= 1;
                cursor += 1;
                if depth == 0 {
                    return Ok(cursor);
                }
            }
            _ => cursor += 1,
        }
    }
}

fn skip_json_number(json: &[u8], mut cursor: usize) -> Result<usize, ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);
    if json[cursor] == b'-' {
        cursor += 1;
    }
    if cursor >= json.len() {
        return Err(err());
    }
    match json[cursor] {
        b'0' => cursor += 1,
        b'1'..=b'9' => {
            cursor += 1;
            while cursor < json.len() && json[cursor].is_ascii_digit() {
                cursor += 1;
            }
        }
        _ => return Err(err()),
    }

    if cursor < json.len() && json[cursor] == b'.' {
        cursor += 1;
        let frac_start = cursor;
        while cursor < json.len() && json[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor == frac_start {
            return Err(err());
        }
    }

    if cursor < json.len() && matches!(json[cursor], b'e' | b'E') {
        cursor += 1;
        if cursor < json.len() && matches!(json[cursor], b'+' | b'-') {
            cursor += 1;
        }
        let exp_start = cursor;
        while cursor < json.len() && json[cursor].is_ascii_digit() {
            cursor += 1;
        }
        if cursor == exp_start {
            return Err(err());
        }
    }
    Ok(cursor)
}

fn skip_string_body(json: &[u8], mut cursor: usize) -> Result<usize, ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);
    debug_assert!(cursor < json.len() && json[cursor] == b'"');
    cursor += 1;
    while cursor < json.len() {
        let b = json[cursor];
        if b == b'"' {
            return Ok(cursor + 1);
        }
        if b == b'\\' {
            if cursor + 1 >= json.len() {
                return Err(err());
            }
            cursor += 2;
            continue;
        }
        cursor += 1;
    }
    Err(err())
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

    fn client_data_for(challenge: &[u8; 32]) -> String {
        let b64 = encode_challenge(challenge);
        let challenge = core::str::from_utf8(&b64).unwrap();
        format!(r#"{{"type":"webauthn.get","challenge":"{}"}}"#, challenge)
    }

    #[test]
    fn test_verify_client_data_json_returns_signed_hash() {
        let challenge = [0x42u8; 32];
        let client_data = client_data_for(&challenge);
        let expected = hashv(&[client_data.as_bytes()]).to_bytes();
        let actual =
            verify_client_data_json(client_data.as_bytes(), &encode_challenge(&challenge)).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_verify_client_data_json_rejects_wrong_challenge() {
        let signed_challenge = [0x42u8; 32];
        let attempted_operation = [0x43u8; 32];
        let client_data = client_data_for(&signed_challenge);
        let err = verify_client_data_json(
            client_data.as_bytes(),
            &encode_challenge(&attempted_operation),
        )
        .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnChallengeMismatch as u32)
        );
    }

    #[test]
    fn test_verify_client_data_json_rejects_wrong_type() {
        let challenge = [0x42u8; 32];
        let b64 = encode_challenge(&challenge);
        let challenge_str = core::str::from_utf8(&b64).unwrap();
        let client_data = format!(
            r#"{{"type":"webauthn.create","challenge":"{}"}}"#,
            challenge_str
        );
        let err = verify_client_data_json(client_data.as_bytes(), &encode_challenge(&challenge))
            .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnInvalidType as u32)
        );
    }

    #[test]
    fn test_verify_client_data_json_rejects_duplicate_after_valid_fields() {
        let challenge = [0x42u8; 32];
        let b64 = encode_challenge(&challenge);
        let challenge_str = core::str::from_utf8(&b64).unwrap();
        let client_data = format!(
            r#"{{"type":"webauthn.get","challenge":"{}","challenge":"evil"}}"#,
            challenge_str
        );
        let err = verify_client_data_json(client_data.as_bytes(), &encode_challenge(&challenge))
            .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnDuplicateField as u32)
        );
    }

    #[test]
    fn test_verify_client_data_json_accepts_extra_fields() {
        let challenge = [0x42u8; 32];
        let b64 = encode_challenge(&challenge);
        let challenge_str = core::str::from_utf8(&b64).unwrap();
        let client_data = format!(
            r#"{{"type":"webauthn.get","challenge":"{}","origin":"https://soulpass.ai","crossOrigin":false}}"#,
            challenge_str
        );
        assert!(
            verify_client_data_json(client_data.as_bytes(), &encode_challenge(&challenge)).is_ok()
        );
    }

    #[test]
    fn test_verify_client_data_json_rejects_missing_comma() {
        let challenge = [0x42u8; 32];
        let b64 = encode_challenge(&challenge);
        let challenge_str = core::str::from_utf8(&b64).unwrap();
        let client_data = format!(
            r#"{{"type":"webauthn.get" "challenge":"{}"}}"#,
            challenge_str
        );
        let err = verify_client_data_json(client_data.as_bytes(), &encode_challenge(&challenge))
            .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::InvalidWebAuthnClientDataJson as u32)
        );
    }

    #[test]
    fn test_verify_client_data_json_rejects_trailing_garbage() {
        let challenge = [0x42u8; 32];
        let client_data = format!("{} trailing", client_data_for(&challenge));
        let err = verify_client_data_json(client_data.as_bytes(), &encode_challenge(&challenge))
            .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::InvalidWebAuthnClientDataJson as u32)
        );
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
