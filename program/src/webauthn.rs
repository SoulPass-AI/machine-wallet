use solana_program::{hash::hashv, program_error::ProgramError};

use crate::error::MachineWalletError;

/// Maximum authenticatorData size. Minimum is 37 (rpIdHash 32 + flags 1 +
/// signCount 4). Upper bound sized to cover assertions with practical extensions
/// such as `largeBlob` read outputs and `prf` eval results, plus margin.
pub const MAX_AUTH_DATA_SIZE: usize = 512;
/// Minimum authenticatorData size (rpIdHash 32 + flags 1 + signCount 4).
/// Re-exported so the wire-level parser enforces the same lower bound.
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
/// Maximum clientDataJSON size.
pub const MAX_CLIENT_DATA_JSON_SIZE: usize = 1024;

/// Build the WebAuthn signed message from authenticatorData + clientDataJSON.
///
/// Verifies that:
/// 1. authenticatorData length is within [37, MAX_AUTH_DATA_SIZE]
/// 2. authenticatorData rpIdHash matches SHA-256(EXPECTED_RP_ID)
/// 3. authenticatorData flags have UP and UV set
/// 4. clientDataJSON.type == "webauthn.get"
/// 5. clientDataJSON.challenge == `expected_challenge_b64`
///
/// `expected_challenge_b64` is the base64url-no-pad encoding of the 32-byte
/// operation hash. Callers that verify many sidecars per operation should
/// encode once via `encode_challenge` and reuse it across calls.
///
/// Returns the number of bytes written to `message_buf`.
/// The message is: authenticatorData || SHA256(clientDataJSON)
pub fn build_webauthn_message(
    auth_data: &[u8],
    client_data_json: &[u8],
    expected_challenge_b64: &[u8; 43],
    message_buf: &mut [u8],
) -> Result<usize, ProgramError> {
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

    // Validate clientDataJSON size
    if client_data_json.is_empty() || client_data_json.len() > MAX_CLIENT_DATA_JSON_SIZE {
        return Err(MachineWalletError::InvalidWebAuthnClientDataJson.into());
    }

    // Verify type == "webauthn.get" and challenge matches in a single JSON pass
    let (auth_type, challenge) =
        extract_type_and_challenge(client_data_json)?.ok_or(MachineWalletError::InvalidWebAuthnClientDataJson)?;
    if auth_type != b"webauthn.get" {
        return Err(MachineWalletError::WebAuthnInvalidType.into());
    }
    if challenge != expected_challenge_b64 {
        return Err(MachineWalletError::WebAuthnChallengeMismatch.into());
    }

    // Compute SHA256(clientDataJSON) via syscall
    let client_data_hash = hashv(&[client_data_json]);

    // Build message: auth_data || SHA256(clientDataJSON)
    let msg_len = auth_data.len() + 32;
    if message_buf.len() < msg_len {
        return Err(MachineWalletError::InvalidWebAuthnAuthData.into());
    }

    message_buf[..auth_data.len()].copy_from_slice(auth_data);
    message_buf[auth_data.len()..msg_len].copy_from_slice(client_data_hash.as_ref());

    Ok(msg_len)
}

/// Base64url-no-pad encoding of a 32-byte operation hash for use as the
/// WebAuthn challenge. Encode once per verification; reuse across sidecars.
pub fn encode_challenge(challenge: &[u8; 32]) -> [u8; 43] {
    base64url_encode_no_pad(challenge)
}

/// Base64url encode without padding (RFC 4648 section 5).
/// Input: 32 bytes. Output: 43 base64url chars (no padding).
fn base64url_encode_no_pad(data: &[u8; 32]) -> [u8; 43] {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = [0u8; 43];
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

/// Extract the top-level "type" and "challenge" string fields from clientDataJSON
/// in a single pass. Returns `Ok(Some((type, challenge)))` if both found,
/// `Ok(None)` if either is missing.
///
/// Canonical-form policy:
/// - Keys and the values of `type` / `challenge` reject any `\` escape. This is
///   an anti-confusion defense: it forces byte-equal representation so that
///   `"\u0074ype"` cannot masquerade as `"type"` under a lax decoder.
/// - Other string values (`origin`, `crossOrigin`, future extensions) are
///   skipped with escape awareness (`\X` consumes two bytes) so a legitimate
///   `origin` containing `\"`, `\\`, or `\uXXXX` does not break parsing.
///
/// Nested object/array contents are ignored. Strings at any depth are consumed
/// with escape awareness so that `{`, `}` inside string literals cannot corrupt
/// the depth counter.
fn extract_type_and_challenge(
    json: &[u8],
) -> Result<Option<(&[u8], &[u8])>, ProgramError> {
    let err = || ProgramError::from(MachineWalletError::InvalidWebAuthnClientDataJson);

    let mut i = 0;
    while i < json.len() && json[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= json.len() || json[i] != b'{' {
        return Err(err());
    }

    let mut depth: usize = 0;
    let mut cursor = i;
    let mut auth_type: Option<&[u8]> = None;
    let mut challenge: Option<&[u8]> = None;

    while cursor < json.len() {
        let byte = json[cursor];

        if byte == b'{' {
            depth += 1;
            cursor += 1;
            continue;
        }
        if byte == b'}' {
            if depth == 0 {
                return Err(err());
            }
            depth -= 1;
            cursor += 1;
            continue;
        }

        if byte != b'"' {
            cursor += 1;
            continue;
        }

        // At any depth > 1 we are inside a nested object/array. Skip the whole
        // string literal with escape awareness so embedded `{` / `}` cannot
        // desynchronize the depth counter.
        if depth != 1 {
            cursor = skip_string_body(json, cursor)?;
            continue;
        }

        // Top-level key. Keys forbid escapes (canonical form).
        let key_start = cursor + 1;
        let mut key_end = key_start;
        while key_end < json.len() {
            let b = json[key_end];
            if b == b'"' {
                break;
            }
            if b == b'\\' {
                return Err(err());
            }
            key_end += 1;
        }
        if key_end >= json.len() {
            return Err(err());
        }
        cursor = key_end + 1;

        while cursor < json.len() && json[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= json.len() || json[cursor] != b':' {
            return Err(err());
        }
        cursor += 1;
        while cursor < json.len() && json[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= json.len() {
            return Err(err());
        }

        let key = &json[key_start..key_end];
        let want = matches!(key, b"type" | b"challenge");

        if want {
            // Wanted values must be canonical strings (no escapes).
            if json[cursor] != b'"' {
                return Err(err());
            }
            let value_start = cursor + 1;
            let mut value_end = value_start;
            while value_end < json.len() {
                let b = json[value_end];
                if b == b'"' {
                    break;
                }
                if b == b'\\' {
                    return Err(err());
                }
                value_end += 1;
            }
            if value_end >= json.len() {
                return Err(err());
            }
            let value = &json[value_start..value_end];
            let slot = if key == b"type" {
                &mut auth_type
            } else {
                &mut challenge
            };
            if slot.is_some() {
                return Err(MachineWalletError::WebAuthnDuplicateField.into());
            }
            *slot = Some(value);
            cursor = value_end + 1;

            if let (Some(t), Some(c)) = (auth_type, challenge) {
                return Ok(Some((t, c)));
            }
            continue;
        }

        // Non-wanted value. For string values we must consume the whole literal
        // (escape-aware) so that `"` inside the string doesn't later be read as
        // a key quote. Non-string values (number / object / array / literal)
        // are handled by the main loop: `{` / `}` update depth, string literals
        // are caught by the `"` branch above at the appropriate depth.
        if json[cursor] == b'"' {
            cursor = skip_string_body(json, cursor)?;
        }
    }

    Ok(match (auth_type, challenge) {
        (Some(t), Some(c)) => Some((t, c)),
        _ => None,
    })
}

/// Consume a JSON string literal starting at the opening `"` and return the
/// position just past the closing `"`. Escape sequences `\X` consume two bytes;
/// the identity of the escaped character is irrelevant here because the content
/// is discarded — we only need to avoid treating `\"` as a string terminator.
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
    fn test_extract_type_and_challenge_basic() {
        let json = br#"{"type":"webauthn.get","challenge":"abc123"}"#;
        let (t, c) = extract_type_and_challenge(json).unwrap().unwrap();
        assert_eq!(t, b"webauthn.get");
        assert_eq!(c, b"abc123");
    }

    #[test]
    fn test_extract_type_and_challenge_missing() {
        let json = br#"{"type":"webauthn.get"}"#;
        assert!(extract_type_and_challenge(json).unwrap().is_none());
    }

    #[test]
    fn test_extract_json_rejects_escaped_chars() {
        let json = br#"{"type":"webauthn\.get","challenge":"x"}"#;
        assert!(extract_type_and_challenge(json).is_err());
    }

    #[test]
    fn test_extract_json_rejects_escaped_key() {
        let json = br#"{"ty\"pe":"value","challenge":"x"}"#;
        assert!(extract_type_and_challenge(json).is_err());
    }

    #[test]
    fn test_extract_type_and_challenge_rejects_duplicate_type() {
        let json = br#"{"type":"webauthn.create","type":"webauthn.get","challenge":"x"}"#;
        let err = extract_type_and_challenge(json).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnDuplicateField as u32)
        );
    }

    #[test]
    fn test_extract_type_and_challenge_rejects_duplicate_challenge() {
        let json = br#"{"challenge":"fake","challenge":"real","type":"webauthn.get"}"#;
        let err = extract_type_and_challenge(json).unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnDuplicateField as u32)
        );
    }

    #[test]
    fn test_build_webauthn_message_valid() {
        // Create a valid clientDataJSON
        let challenge_bytes = [0x42u8; 32];
        let b64_challenge = base64url_encode_no_pad(&challenge_bytes);
        let challenge_str = core::str::from_utf8(&b64_challenge).unwrap();

        let client_data = format!(
            r#"{{"type":"webauthn.get","challenge":"{}","origin":"https://example.com","crossOrigin":false}}"#,
            challenge_str
        );

        let auth_data: [u8; 37] = make_auth_data();
        let mut buf = [0u8; 256];

        let len = build_webauthn_message(
            &auth_data,
            client_data.as_bytes(),
            &b64_challenge,
            &mut buf,
        )
        .unwrap();

        assert_eq!(len, 37 + 32);
        assert_eq!(&buf[..37], &auth_data);
    }

    fn client_data_for(challenge: &[u8; 32]) -> String {
        let b64 = base64url_encode_no_pad(challenge);
        let s = core::str::from_utf8(&b64).unwrap();
        format!(r#"{{"type":"webauthn.get","challenge":"{}"}}"#, s)
    }

    #[test]
    fn test_build_webauthn_message_rejects_up_clear() {
        let challenge_bytes = [0x42u8; 32];
        let client_data = client_data_for(&challenge_bytes);
        let mut auth_data = [0xAAu8; 37];
        auth_data[..32].copy_from_slice(&EXPECTED_RP_ID_HASH);
        auth_data[AUTH_DATA_FLAGS_OFFSET] = FLAG_USER_VERIFIED; // UV=1, UP=0
        let mut buf = [0u8; 256];

        let err = build_webauthn_message(
            &auth_data,
            client_data.as_bytes(),
            &encode_challenge(&challenge_bytes),
            &mut buf,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnUserNotPresent as u32)
        );
    }

    #[test]
    fn test_build_webauthn_message_rejects_uv_clear() {
        let challenge_bytes = [0x42u8; 32];
        let client_data = client_data_for(&challenge_bytes);
        let mut auth_data = [0xAAu8; 37];
        auth_data[..32].copy_from_slice(&EXPECTED_RP_ID_HASH);
        auth_data[AUTH_DATA_FLAGS_OFFSET] = FLAG_USER_PRESENT; // UP=1, UV=0
        let mut buf = [0u8; 256];

        let err = build_webauthn_message(
            &auth_data,
            client_data.as_bytes(),
            &encode_challenge(&challenge_bytes),
            &mut buf,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnUserNotVerified as u32)
        );
    }

    #[test]
    fn test_build_webauthn_message_wrong_type() {
        let challenge_bytes = [0x42u8; 32];
        let b64 = base64url_encode_no_pad(&challenge_bytes);
        let challenge_str = core::str::from_utf8(&b64).unwrap();

        let client_data = format!(
            r#"{{"type":"webauthn.create","challenge":"{}"}}"#,
            challenge_str
        );

        let auth_data: [u8; 37] = make_auth_data();
        let mut buf = [0u8; 256];

        let err = build_webauthn_message(
            &auth_data,
            client_data.as_bytes(),
            &b64,
            &mut buf,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnInvalidType as u32)
        );
    }

    #[test]
    fn test_build_webauthn_message_wrong_challenge() {
        let challenge_bytes = [0x42u8; 32];
        let wrong_bytes = [0x43u8; 32];
        let b64 = base64url_encode_no_pad(&wrong_bytes);
        let challenge_str = core::str::from_utf8(&b64).unwrap();

        let client_data = format!(
            r#"{{"type":"webauthn.get","challenge":"{}"}}"#,
            challenge_str
        );

        let auth_data: [u8; 37] = make_auth_data();
        let mut buf = [0u8; 256];

        let err = build_webauthn_message(
            &auth_data,
            client_data.as_bytes(),
            &encode_challenge(&challenge_bytes),
            &mut buf,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnChallengeMismatch as u32)
        );
    }

    #[test]
    fn test_build_webauthn_message_auth_data_too_short() {
        let auth_data = [0xAA; 36]; // below minimum 37
        let client_data = br#"{"type":"webauthn.get","challenge":"x"}"#;
        let mut buf = [0u8; 256];

        let err = build_webauthn_message(
            &auth_data,
            client_data,
            &encode_challenge(&[0u8; 32]),
            &mut buf,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::InvalidWebAuthnAuthData as u32)
        );
    }

    #[test]
    fn test_build_webauthn_message_auth_data_too_long() {
        let auth_data = vec![0xAAu8; MAX_AUTH_DATA_SIZE + 1];
        let client_data = br#"{"type":"webauthn.get","challenge":"x"}"#;
        let mut buf = vec![0u8; MAX_AUTH_DATA_SIZE + 64];

        let err = build_webauthn_message(
            &auth_data,
            client_data,
            &encode_challenge(&[0u8; 32]),
            &mut buf,
        )
        .unwrap_err();

        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::InvalidWebAuthnAuthData as u32)
        );
    }

    /// Non-wanted string values (here: `origin`) may contain JSON escape
    /// sequences — these used to be rejected wholesale, which broke IDN /
    /// non-ASCII origins. The parser must now skip them with escape awareness
    /// while still returning the correct `type` and `challenge`.
    #[test]
    fn test_extract_allows_escapes_in_non_wanted_values() {
        let json = br#"{"type":"webauthn.get","origin":"https://w.test/\"path\"","challenge":"abc"}"#;
        let (t, c) = extract_type_and_challenge(json).unwrap().unwrap();
        assert_eq!(t, b"webauthn.get");
        assert_eq!(c, b"abc");
    }

    #[test]
    fn test_extract_allows_unicode_escape_in_origin() {
        let json = br#"{"origin":"https://\u4f60\u597d.test","type":"webauthn.get","challenge":"xyz"}"#;
        let (t, c) = extract_type_and_challenge(json).unwrap().unwrap();
        assert_eq!(t, b"webauthn.get");
        assert_eq!(c, b"xyz");
    }

    /// A nested object containing `{` or `}` inside a string literal must not
    /// desynchronize the depth counter.
    #[test]
    fn test_extract_nested_string_with_braces() {
        let json = br#"{"tokenBinding":{"status":"}{"},"type":"webauthn.get","challenge":"z"}"#;
        let (t, c) = extract_type_and_challenge(json).unwrap().unwrap();
        assert_eq!(t, b"webauthn.get");
        assert_eq!(c, b"z");
    }

    /// Escapes are still forbidden inside the canonical `type` / `challenge`
    /// values — this is the anti-confusion invariant.
    #[test]
    fn test_extract_still_rejects_escape_in_challenge() {
        let json = br#"{"type":"webauthn.get","challenge":"ab\u0063"}"#;
        assert!(extract_type_and_challenge(json).is_err());
    }

    /// A lone trailing backslash inside a string (malformed JSON) must error,
    /// not silently consume past end-of-input.
    #[test]
    fn test_extract_rejects_unterminated_escape() {
        let json = br#"{"origin":"x\"#;
        assert!(extract_type_and_challenge(json).is_err());
    }

    /// The hard-coded EXPECTED_RP_ID_HASH must equal SHA-256(EXPECTED_RP_ID).
    /// If either constant drifts the whole WebAuthn path breaks silently, so we
    /// lock them together with a compile-free runtime check.
    #[test]
    fn test_expected_rp_id_hash_matches() {
        let computed = hashv(&[EXPECTED_RP_ID]);
        assert_eq!(computed.as_ref(), &EXPECTED_RP_ID_HASH);
    }

    /// An assertion whose authenticatorData carries a rpIdHash for any other RP
    /// must be rejected, even when UP/UV/type/challenge all validate.
    #[test]
    fn test_build_webauthn_message_rejects_wrong_rpid() {
        let challenge_bytes = [0x42u8; 32];
        let client_data = client_data_for(&challenge_bytes);
        let mut auth_data: [u8; 37] = make_auth_data();
        // Flip one byte of the rpIdHash — simulates a credential from a
        // different Relying Party.
        auth_data[0] ^= 0x01;
        let mut buf = [0u8; 256];

        let err = build_webauthn_message(
            &auth_data,
            client_data.as_bytes(),
            &encode_challenge(&challenge_bytes),
            &mut buf,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ProgramError::Custom(MachineWalletError::WebAuthnRpIdMismatch as u32)
        );
    }
}
