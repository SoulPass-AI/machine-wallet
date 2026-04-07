use solana_program::{keccak, program_error::ProgramError};

/// SessionState version byte. Must remain 0 — wallet state uses version 1+,
/// so this value also serves as a discriminator between account types.
pub const SESSION_STATE_VERSION: u8 = 0;

/// System program ID (11111111111111111111111111111111).
/// Shared across all processors to avoid repeated definitions.
pub const SYSTEM_PROGRAM_ID: solana_program::pubkey::Pubkey =
    solana_program::pubkey!("11111111111111111111111111111111");

/// Signature scheme identifiers.
/// Phase 0 supports only Secp256r1; future versions may add Ed25519, Dilithium, etc.
pub const SIG_SCHEME_SECP256R1: u8 = 0;

/// Ed25519 signature scheme identifier.
pub const SIG_SCHEME_ED25519: u8 = 1;

/// Maximum number of authorities in a multi-authority wallet.
pub const MAX_AUTHORITIES: u8 = 16;

/// Size of a single AuthoritySlot: 1 (scheme) + 33 (pubkey).
pub const AUTHORITY_SLOT_SIZE: usize = 34;

/// V1 fixed header size: version(1) + bump(1) + wallet_id(32) + threshold(1) + authority_count(1) + nonce(8) + creation_slot(8) + vault_bump(1) = 53
pub const V1_HEADER_SIZE: usize = 53;

/// Single authority entry: signature scheme + compressed public key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthoritySlot {
    pub sig_scheme: u8,
    pub pubkey: [u8; 33],
}

impl AuthoritySlot {
    /// An empty/default authority slot. sig_scheme = 0xFF is an invalid sentinel
    /// that cannot match SIG_SCHEME_SECP256R1 (0) or SIG_SCHEME_ED25519 (1),
    /// preventing accidental matches during threshold scanning.
    pub const EMPTY: Self = Self {
        sig_scheme: 0xFF,
        pubkey: [0u8; 33],
    };

    /// Validate this authority slot based on its signature scheme.
    pub fn is_valid(&self) -> bool {
        match self.sig_scheme {
            SIG_SCHEME_SECP256R1 => MachineWallet::is_valid_authority(&self.pubkey),
            SIG_SCHEME_ED25519 => Self::is_valid_ed25519(&self.pubkey),
            _ => false,
        }
    }

    /// Ed25519 pubkey: 32 bytes + 0x00 padding.
    ///
    /// Lightweight format check only — full curve-point decompression via
    /// `curve25519-dalek` causes stack overflow / CU exhaustion in BPF.
    /// Security is preserved: the Ed25519 precompile rejects signatures from
    /// invalid keys at signing time, so an invalid key can never authorize
    /// a transaction. The worst case is a wasted authority slot that the
    /// existing owner can remove.
    fn is_valid_ed25519(pubkey: &[u8; 33]) -> bool {
        // Padding byte (byte 32) must be 0x00
        if pubkey[32] != 0 {
            return false;
        }
        // Key must not be all zeros (definitely invalid)
        pubkey[..32] != [0u8; 32]
    }
}

/// MachineWallet on-chain state.
///
/// Supports two on-chain layouts:
///
/// **V0 Layout (87 bytes, single authority):**
/// - version:         u8       (offset 0)   = 0
/// - bump:            u8       (offset 1)
/// - wallet_id:       [u8; 32] (offset 2)
/// - sig_scheme:      u8       (offset 34)  — header-level, single scheme
/// - threshold:       u8       (offset 35)
/// - authority_count: u8       (offset 36)
/// - authority:       [u8; 33] (offset 37)
/// - nonce:           u64      (offset 70)
/// - creation_slot:   u64      (offset 78)
/// - vault_bump:      u8       (offset 86)
///
/// **V1 Layout (53 + N*34 bytes, multi-authority):**
/// - version:         u8       (offset 0)   = 1
/// - bump:            u8       (offset 1)
/// - wallet_id:       [u8; 32] (offset 2)
/// - threshold:       u8       (offset 34)
/// - authority_count: u8       (offset 35)
/// - nonce:           u64      (offset 36)
/// - creation_slot:   u64      (offset 44)
/// - vault_bump:      u8       (offset 52)
/// - authorities:     [AuthoritySlot; N] (offset 53), N = authority_count
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineWallet {
    pub version: u8,
    pub bump: u8,
    pub wallet_id: [u8; 32],
    pub threshold: u8,
    pub authority_count: u8,
    pub authorities: [AuthoritySlot; MAX_AUTHORITIES as usize],
    pub nonce: u64,
    pub creation_slot: u64,
    pub vault_bump: u8,
}

impl MachineWallet {
    /// Total serialized size in bytes for v0 (fixed part, single authority).
    pub const LEN: usize = 1 + 1 + 32 + 1 + 1 + 1 + 33 + 8 + 8 + 1; // 87

    // --- V0 offsets (kept for backward compatibility and migration) ---

    /// Byte offset of the nonce field within v0 serialized state.
    pub const V0_NONCE_OFFSET: usize = 70;
    /// Byte offset of the creation_slot field within v0 serialized state.
    pub const V0_CREATION_SLOT_OFFSET: usize = 78;
    /// Byte offset of the vault_bump field within v0 serialized state.
    pub const V0_VAULT_BUMP_OFFSET: usize = 86;

    // --- V1 offsets ---

    /// Byte offset of the nonce field within v1 serialized state.
    pub const V1_NONCE_OFFSET: usize = 36;
    /// Byte offset of the creation_slot field within v1 serialized state.
    pub const V1_CREATION_SLOT_OFFSET: usize = 44;
    /// Byte offset of the vault_bump field within v1 serialized state.
    pub const V1_VAULT_BUMP_OFFSET: usize = 52;

    /// Wallet PDA seed prefix.
    pub const SEED_PREFIX: &'static [u8] = b"machine_wallet";

    /// Vault PDA seed prefix.
    pub const VAULT_SEED_PREFIX: &'static [u8] = b"machine_vault";

    /// Version-aware nonce offset.
    pub fn nonce_offset(&self) -> usize {
        match self.version {
            0 => Self::V0_NONCE_OFFSET,
            _ => Self::V1_NONCE_OFFSET,
        }
    }

    /// Version-aware creation_slot offset.
    pub fn creation_slot_offset(&self) -> usize {
        match self.version {
            0 => Self::V0_CREATION_SLOT_OFFSET,
            _ => Self::V1_CREATION_SLOT_OFFSET,
        }
    }

    /// Version-aware vault_bump offset.
    pub fn vault_bump_offset(&self) -> usize {
        match self.version {
            0 => Self::V0_VAULT_BUMP_OFFSET,
            _ => Self::V1_VAULT_BUMP_OFFSET,
        }
    }

    /// Return the stored wallet ID (set at creation, never changes).
    /// Used for PDA derivation.
    pub fn id(&self) -> [u8; 32] {
        self.wallet_id
    }

    /// Compute wallet ID from an authority pubkey: keccak256(authority).
    /// Used at creation time to derive and store the wallet_id.
    pub fn compute_id(authority: &[u8; 33]) -> [u8; 32] {
        keccak::hash(authority).to_bytes()
    }

    /// Calculate the required account size for a v1 wallet with the given authority count.
    pub fn v1_account_size(authority_count: u8) -> usize {
        V1_HEADER_SIZE + authority_count as usize * AUTHORITY_SLOT_SIZE
    }

    /// Serialize v0 layout into a byte buffer (must be at least LEN bytes).
    pub fn serialize(&self, dst: &mut [u8]) -> Result<(), ProgramError> {
        if dst.len() < Self::LEN {
            return Err(ProgramError::AccountDataTooSmall);
        }
        dst[0] = self.version;
        dst[1] = self.bump;
        dst[2..34].copy_from_slice(&self.wallet_id);
        // v0: sig_scheme at offset 34 from first authority slot
        dst[34] = self.authorities[0].sig_scheme;
        dst[35] = self.threshold;
        dst[36] = self.authority_count;
        dst[37..70].copy_from_slice(&self.authorities[0].pubkey);
        dst[Self::V0_NONCE_OFFSET..Self::V0_NONCE_OFFSET + 8]
            .copy_from_slice(&self.nonce.to_le_bytes());
        dst[Self::V0_CREATION_SLOT_OFFSET..Self::V0_CREATION_SLOT_OFFSET + 8]
            .copy_from_slice(&self.creation_slot.to_le_bytes());
        dst[Self::V0_VAULT_BUMP_OFFSET] = self.vault_bump;
        Ok(())
    }

    /// Serialize v1 layout into a byte buffer.
    pub fn serialize_v1(&self, dst: &mut [u8]) -> Result<(), ProgramError> {
        let required = Self::v1_account_size(self.authority_count);
        if dst.len() < required {
            return Err(ProgramError::AccountDataTooSmall);
        }
        dst[0] = self.version;
        dst[1] = self.bump;
        dst[2..34].copy_from_slice(&self.wallet_id);
        dst[34] = self.threshold;
        dst[35] = self.authority_count;
        dst[Self::V1_NONCE_OFFSET..Self::V1_NONCE_OFFSET + 8]
            .copy_from_slice(&self.nonce.to_le_bytes());
        dst[Self::V1_CREATION_SLOT_OFFSET..Self::V1_CREATION_SLOT_OFFSET + 8]
            .copy_from_slice(&self.creation_slot.to_le_bytes());
        dst[Self::V1_VAULT_BUMP_OFFSET] = self.vault_bump;
        // Write authority slots
        for i in 0..self.authority_count as usize {
            let offset = V1_HEADER_SIZE + i * AUTHORITY_SLOT_SIZE;
            dst[offset] = self.authorities[i].sig_scheme;
            dst[offset + 1..offset + 1 + 33].copy_from_slice(&self.authorities[i].pubkey);
        }
        Ok(())
    }

    /// Deserialize v0 layout (existing 87-byte single-authority format).
    fn deserialize_v0(src: &[u8], validate: bool) -> Result<Self, ProgramError> {
        if src.len() < Self::LEN {
            return Err(ProgramError::AccountDataTooSmall);
        }

        let version = src[0];
        // v0 must have version == 0
        if version != 0 {
            return Err(ProgramError::InvalidAccountData);
        }

        let bump = src[1];

        let mut wallet_id = [0u8; 32];
        wallet_id.copy_from_slice(&src[2..34]);

        let sig_scheme = src[34];
        if sig_scheme != SIG_SCHEME_SECP256R1 {
            return Err(ProgramError::InvalidAccountData);
        }
        let threshold = src[35];
        let authority_count = src[36];
        // V0 layout has exactly one authority slot (87 bytes fixed).
        // Enforce this invariant even though we are the only writer.
        if authority_count != 1 || threshold != 1 {
            return Err(ProgramError::InvalidAccountData);
        }

        let mut pubkey = [0u8; 33];
        pubkey.copy_from_slice(&src[37..70]);

        // Length already validated ≥ 87; these slices are guaranteed 8 bytes.
        let nonce = u64::from_le_bytes(
            src[Self::V0_NONCE_OFFSET..Self::V0_NONCE_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let creation_slot = u64::from_le_bytes(
            src[Self::V0_CREATION_SLOT_OFFSET..Self::V0_CREATION_SLOT_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let vault_bump = src[Self::V0_VAULT_BUMP_OFFSET];

        // Wrap single authority into AuthoritySlot
        let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        authorities[0] = AuthoritySlot { sig_scheme, pubkey };

        let wallet = Self {
            version,
            bump,
            wallet_id,
            threshold,
            authority_count,
            authorities,
            nonce,
            creation_slot,
            vault_bump,
        };

        if validate {
            // Validate the single authority (P-256 curve point decompression)
            if !authorities[0].is_valid() {
                return Err(ProgramError::InvalidAccountData);
            }
        }

        Ok(wallet)
    }

    /// Deserialize v1 layout (multi-authority format).
    fn deserialize_v1(src: &[u8], validate: bool) -> Result<Self, ProgramError> {
        if src.len() < V1_HEADER_SIZE {
            return Err(ProgramError::AccountDataTooSmall);
        }

        let version = src[0];
        if version != 1 {
            return Err(ProgramError::InvalidAccountData);
        }

        let bump = src[1];

        let mut wallet_id = [0u8; 32];
        wallet_id.copy_from_slice(&src[2..34]);

        let threshold = src[34];
        let authority_count = src[35];

        if threshold == 0 || authority_count == 0 {
            return Err(ProgramError::InvalidAccountData);
        }
        if threshold > authority_count {
            return Err(ProgramError::InvalidAccountData);
        }
        if authority_count > MAX_AUTHORITIES {
            return Err(ProgramError::InvalidAccountData);
        }

        let required_size = Self::v1_account_size(authority_count);
        if src.len() < required_size {
            return Err(ProgramError::AccountDataTooSmall);
        }

        // Length already validated ≥ required_size; these slices are guaranteed 8 bytes.
        let nonce = u64::from_le_bytes(
            src[Self::V1_NONCE_OFFSET..Self::V1_NONCE_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let creation_slot = u64::from_le_bytes(
            src[Self::V1_CREATION_SLOT_OFFSET..Self::V1_CREATION_SLOT_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let vault_bump = src[Self::V1_VAULT_BUMP_OFFSET];

        let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        for i in 0..authority_count as usize {
            let offset = V1_HEADER_SIZE + i * AUTHORITY_SLOT_SIZE;
            let sig_scheme = src[offset];
            let mut pubkey = [0u8; 33];
            pubkey.copy_from_slice(&src[offset + 1..offset + 1 + 33]);
            authorities[i] = AuthoritySlot { sig_scheme, pubkey };
        }

        if validate {
            for i in 0..authority_count as usize {
                if !authorities[i].is_valid() {
                    return Err(ProgramError::InvalidAccountData);
                }
            }
        }

        Ok(Self {
            version,
            bump,
            wallet_id,
            threshold,
            authority_count,
            authorities,
            nonce,
            creation_slot,
            vault_bump,
        })
    }

    /// Deserialize from a byte buffer with full integrity checks.
    ///
    /// Dispatches by version byte. Validates authority pubkeys.
    pub fn deserialize(src: &[u8]) -> Result<Self, ProgramError> {
        if src.is_empty() {
            return Err(ProgramError::AccountDataTooSmall);
        }
        match src[0] {
            0 => Self::deserialize_v0(src, true),
            1 => Self::deserialize_v1(src, true),
            _ => Err(ProgramError::InvalidAccountData),
        }
    }

    /// Deserialize for runtime hot paths.
    ///
    /// Execute/CloseWallet/AdvanceNonce already require `owner == program_id`
    /// and re-derive the wallet PDA from `(id, bump)`, so repeating P-256 point
    /// validation here only burns compute. The program itself is the sole writer
    /// of initialized wallet state after creation.
    pub fn deserialize_runtime(src: &[u8]) -> Result<Self, ProgramError> {
        if src.is_empty() {
            return Err(ProgramError::AccountDataTooSmall);
        }
        match src[0] {
            0 => Self::deserialize_v0(src, false),
            1 => Self::deserialize_v1(src, false),
            _ => Err(ProgramError::InvalidAccountData),
        }
    }

    /// Lightweight format check on the first authority pubkey (test-only).
    /// Processors use `AuthoritySlot::is_valid()` which handles all schemes.
    #[cfg(test)]
    pub fn validate_authority(&self) -> Result<(), ProgramError> {
        if !Self::is_valid_authority(&self.authorities[0].pubkey) {
            return Err(ProgramError::InvalidAccountData);
        }
        Ok(())
    }

    /// Lightweight P-256 compressed pubkey format check.
    ///
    /// Full curve-point decompression (`p256::PublicKey::from_sec1_bytes`) costs
    /// ~435K CU on BPF — removed in favor of this format check.
    /// Security is preserved: the Secp256r1 precompile rejects signatures from
    /// invalid keys at signing time, so an invalid key can never authorize
    /// a transaction. The worst case is a wasted wallet slot whose rent the
    /// attacker pays and cannot reclaim. Same strategy as `is_valid_ed25519`.
    pub fn is_valid_authority(authority: &[u8; 33]) -> bool {
        // Valid compressed SEC1 prefix: 0x02 (even y) or 0x03 (odd y)
        let prefix = authority[0];
        if prefix != 0x02 && prefix != 0x03 {
            return false;
        }
        // x-coordinate must not be all zeros (not a valid curve point)
        authority[1..] != [0u8; 32]
    }
}

/// Maximum number of allowed programs in a session whitelist.
pub const MAX_ALLOWED_PROGRAMS: usize = 8;

/// Session PDA seed prefix.
pub const SESSION_SEED_PREFIX: &[u8] = b"machine_session";

/// SessionState on-chain state.
///
/// Layout (420 bytes total):
/// - version:               u8          (offset 0)   — state version, must be 0
/// - bump:                  u8          (offset 1)   — session PDA bump seed
/// - wallet:                [u8; 32]    (offset 2)   — wallet PDA address
/// - authority:             [u8; 32]    (offset 34)  — session authority (Ed25519 pubkey)
/// - created_slot:          u64         (offset 66)  — slot at creation
/// - expiry_slot:           u64         (offset 74)  — slot at which session expires
/// - revoked:               bool/u8     (offset 82)  — non-zero means revoked
/// - wallet_creation_slot:  u64         (offset 83)  — wallet's creation_slot, prevents session resurrection
/// - max_lamports_per_ix:   u64         (offset 91)  — per-instruction lamport spend cap
/// - allowed_programs_count: u8         (offset 99)  — number of whitelisted programs (≤ MAX_ALLOWED_PROGRAMS)
/// - allowed_programs:      [[u8;32];8] (offset 100) — whitelisted program IDs (8 × 32 = 256 bytes)
/// - reserved:              [u8; 64]    (offset 356) — reserved, always zero
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionState {
    pub version: u8,
    pub bump: u8,
    pub wallet: [u8; 32],
    pub authority: [u8; 32],
    pub created_slot: u64,
    pub expiry_slot: u64,
    pub revoked: bool,
    pub wallet_creation_slot: u64,
    pub max_lamports_per_ix: u64,
    pub allowed_programs_count: u8,
    pub allowed_programs: [[u8; 32]; MAX_ALLOWED_PROGRAMS],
}

impl SessionState {
    /// Total serialized size in bytes.
    /// 1+1+32+32+8+8+1+8+8+1+256+64 = 420
    pub const LEN: usize = 420;

    /// Byte offset of the revoked field within serialized state.
    pub const REVOKED_OFFSET: usize = 82;

    /// Serialize into a byte buffer. Zeros reserved region explicitly.
    pub fn serialize(&self, dst: &mut [u8]) -> Result<(), ProgramError> {
        if dst.len() < Self::LEN {
            return Err(ProgramError::AccountDataTooSmall);
        }

        dst[0] = self.version;
        dst[1] = self.bump;
        dst[2..34].copy_from_slice(&self.wallet);
        dst[34..66].copy_from_slice(&self.authority);
        dst[66..74].copy_from_slice(&self.created_slot.to_le_bytes());
        dst[74..82].copy_from_slice(&self.expiry_slot.to_le_bytes());
        dst[Self::REVOKED_OFFSET] = self.revoked as u8;
        dst[83..91].copy_from_slice(&self.wallet_creation_slot.to_le_bytes());
        dst[91..99].copy_from_slice(&self.max_lamports_per_ix.to_le_bytes());
        dst[99] = self.allowed_programs_count;
        // Only write active program entries; zero the rest + reserved region in one fill
        let active_end = 100 + self.allowed_programs_count as usize * 32;
        for (i, prog) in self
            .allowed_programs
            .iter()
            .take(self.allowed_programs_count as usize)
            .enumerate()
        {
            let start = 100 + i * 32;
            dst[start..start + 32].copy_from_slice(prog);
        }
        // Zero unused program slots + reserved region (active_end..420) in one call
        dst[active_end..420].fill(0);
        Ok(())
    }

    /// Deserialize from a byte buffer. Validates version, allowed_programs_count,
    /// and checks for duplicate allowed programs.
    pub fn deserialize(src: &[u8]) -> Result<Self, ProgramError> {
        Self::deserialize_inner(src, true)
    }

    /// Deserialize for runtime hot paths (SessionExecute, CloseSession, etc.).
    ///
    /// Skips the O(n²) allowed_programs duplicate check. Session data is only
    /// written by our program (CreateSession already validates uniqueness), so
    /// re-checking on every SessionExecute only burns CU.
    pub fn deserialize_runtime(src: &[u8]) -> Result<Self, ProgramError> {
        Self::deserialize_inner(src, false)
    }

    fn deserialize_inner(src: &[u8], validate: bool) -> Result<Self, ProgramError> {
        if src.len() < Self::LEN {
            return Err(ProgramError::AccountDataTooSmall);
        }

        let version = src[0];
        if version != SESSION_STATE_VERSION {
            return Err(ProgramError::InvalidAccountData);
        }

        let bump = src[1];

        let mut wallet = [0u8; 32];
        wallet.copy_from_slice(&src[2..34]);

        let mut authority = [0u8; 32];
        authority.copy_from_slice(&src[34..66]);

        // Length already validated ≥ 420; these slices are guaranteed 8 bytes.
        let created_slot = u64::from_le_bytes(src[66..74].try_into().unwrap());
        let expiry_slot = u64::from_le_bytes(src[74..82].try_into().unwrap());

        let revoked = src[Self::REVOKED_OFFSET] != 0;

        let wallet_creation_slot = u64::from_le_bytes(src[83..91].try_into().unwrap());
        let max_lamports_per_ix = u64::from_le_bytes(src[91..99].try_into().unwrap());

        let allowed_programs_count = src[99];
        if allowed_programs_count == 0 || allowed_programs_count as usize > MAX_ALLOWED_PROGRAMS {
            return Err(ProgramError::InvalidAccountData);
        }
        if wallet == [0u8; 32] || authority == [0u8; 32] || created_slot > expiry_slot {
            return Err(ProgramError::InvalidAccountData);
        }

        // Only read `allowed_programs_count` entries, not all 8 — saves up to 224 bytes of reads
        let mut allowed_programs = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
        for i in 0..allowed_programs_count as usize {
            let start = 100 + i * 32;
            allowed_programs[i].copy_from_slice(&src[start..start + 32]);
            if validate {
                for prev in allowed_programs.iter().take(i) {
                    if *prev == allowed_programs[i] {
                        return Err(ProgramError::InvalidAccountData);
                    }
                }
            }
        }

        Ok(Self {
            version,
            bump,
            wallet,
            authority,
            created_slot,
            expiry_slot,
            revoked,
            wallet_creation_slot,
            max_lamports_per_ix,
            allowed_programs_count,
            allowed_programs,
        })
    }

    /// Returns true if `program_id` is in the allowed programs whitelist.
    /// Linear scan over allowed_programs_count entries.
    pub fn is_program_allowed(&self, program_id: &[u8; 32]) -> bool {
        for i in 0..self.allowed_programs_count as usize {
            if &self.allowed_programs[i] == program_id {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BASEPOINT_X: [u8; 32] = [
        0x6B, 0x17, 0xD1, 0xF2, 0xE1, 0x2C, 0x42, 0x47, 0xF8, 0xBC, 0xE6, 0xE5, 0x63, 0xA4, 0x40,
        0xF2, 0x77, 0x03, 0x7D, 0x81, 0x2D, 0xEB, 0x33, 0xA0, 0xF4, 0xA1, 0x39, 0x45, 0xD8, 0x98,
        0xC2, 0x96,
    ];

    fn make_p256_authority(prefix: u8) -> AuthoritySlot {
        let mut pubkey = [0u8; 33];
        pubkey[0] = prefix;
        pubkey[1..].copy_from_slice(&BASEPOINT_X);
        AuthoritySlot {
            sig_scheme: SIG_SCHEME_SECP256R1,
            pubkey,
        }
    }

    fn make_ed25519_authority(seed: u8) -> AuthoritySlot {
        let mut pubkey = [0u8; 33];
        pubkey[0] = seed | 1; // ensure non-zero; byte 32 stays 0x00 (valid padding)
        AuthoritySlot {
            sig_scheme: SIG_SCHEME_ED25519,
            pubkey,
        }
    }

    fn make_wallet(prefix: u8) -> MachineWallet {
        let auth = make_p256_authority(prefix);
        let wallet_id = MachineWallet::compute_id(&auth.pubkey);
        let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        authorities[0] = auth;
        MachineWallet {
            version: 0,
            bump: 254,
            wallet_id,
            threshold: 1,
            authority_count: 1,
            authorities,
            nonce: 0x0102030405060708,
            creation_slot: 42,
            vault_bump: 253,
        }
    }

    fn make_v1_wallet(authority_count: u8, authorities: &[AuthoritySlot]) -> MachineWallet {
        let wallet_id = MachineWallet::compute_id(&authorities[0].pubkey);
        let mut auth_arr = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        for (i, a) in authorities.iter().enumerate() {
            auth_arr[i] = *a;
        }
        MachineWallet {
            version: 1,
            bump: 254,
            wallet_id,
            threshold: 1,
            authority_count,
            authorities: auth_arr,
            nonce: 0x0102030405060708,
            creation_slot: 42,
            vault_bump: 253,
        }
    }

    // ==================== V0 Tests (unchanged behavior) ====================

    #[test]
    fn test_wallet_size() {
        assert_eq!(MachineWallet::LEN, 87);
    }

    #[test]
    fn test_wallet_roundtrip() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(wallet, decoded);
        assert_eq!(decoded.version, 0);
        assert_eq!(decoded.bump, 254);
        assert_eq!(
            decoded.wallet_id,
            MachineWallet::compute_id(&decoded.authorities[0].pubkey)
        );
        assert_eq!(decoded.authorities[0].sig_scheme, SIG_SCHEME_SECP256R1);
        assert_eq!(decoded.threshold, 1);
        assert_eq!(decoded.authority_count, 1);
        assert_eq!(decoded.authorities[0].pubkey[0], 0x02);
        assert_eq!(decoded.nonce, 0x0102030405060708);
        assert_eq!(decoded.creation_slot, 42);
        assert_eq!(decoded.vault_bump, 253);
    }

    #[test]
    fn test_wallet_id_stored() {
        let wallet = make_wallet(0x02);
        let expected_id = keccak::hash(&wallet.authorities[0].pubkey).to_bytes();
        assert_eq!(wallet.id(), expected_id);
        assert_eq!(wallet.wallet_id, expected_id);
    }

    #[test]
    fn test_compute_id() {
        let mut authority = [0u8; 33];
        authority[0] = 0x02;
        authority[1..].copy_from_slice(&BASEPOINT_X);
        let id = MachineWallet::compute_id(&authority);
        assert_eq!(id, keccak::hash(&authority).to_bytes());
    }

    #[test]
    fn test_wallet_roundtrip_prefix_03() {
        let wallet = make_wallet(0x03);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.authorities[0].pubkey[0], 0x03);
        decoded.validate_authority().unwrap();
    }

    #[test]
    fn test_wallet_invalid_version() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();

        buf[0] = 99; // invalid version (neither 0 nor 1)
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_too_short() {
        let buf = [0u8; 50];
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    #[test]
    fn test_wallet_serialize_too_short() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; 50];
        let result = wallet.serialize(&mut buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    #[test]
    fn test_wallet_invalid_pubkey_prefix() {
        let mut wallet = make_wallet(0x02);
        wallet.authorities[0].pubkey[0] = 0x04;
        let result = wallet.validate_authority();
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_invalid_pubkey_prefix_zero() {
        let mut wallet = make_wallet(0x02);
        wallet.authorities[0].pubkey[0] = 0x00;
        let result = wallet.validate_authority();
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_off_curve_point_accepted_by_format_check() {
        // Lightweight format check accepts valid-format but off-curve points.
        // Full curve validation is deferred to the Secp256r1 precompile at signing time.
        let mut wallet = make_wallet(0x02);
        let mut authority = [0xFFu8; 33];
        authority[0] = 0x02;
        wallet.authorities[0].pubkey = authority;
        wallet.validate_authority().unwrap(); // format is valid → accepted
    }

    #[test]
    fn test_wallet_rejects_zero_x_coordinate() {
        let mut wallet = make_wallet(0x02);
        let mut authority = [0u8; 33];
        authority[0] = 0x02; // valid prefix but x = 0
        wallet.authorities[0].pubkey = authority;
        let result = wallet.validate_authority();
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_valid_pubkey_prefixes() {
        make_wallet(0x02).validate_authority().unwrap();
        make_wallet(0x03).validate_authority().unwrap();
    }

    #[test]
    fn test_wallet_runtime_deserialize_skips_expensive_integrity_checks() {
        // Runtime deserialize doesn't validate authority
        let mut wallet = make_wallet(0x02);
        wallet.authorities[0].pubkey = [0xFF; 33]; // invalid authority
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize_runtime(&buf).unwrap();
        assert_eq!(decoded.authorities[0].pubkey, wallet.authorities[0].pubkey);
    }

    #[test]
    fn test_wallet_rejects_invalid_sig_scheme() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        buf[34] = 5; // invalid sig_scheme
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_rejects_zero_threshold() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        buf[35] = 0; // zero threshold
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_rejects_zero_authority_count() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        buf[36] = 0; // zero authority_count
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_nonce_max_roundtrip() {
        let mut wallet = make_wallet(0x02);
        wallet.nonce = u64::MAX;
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.nonce, u64::MAX);
    }

    #[test]
    fn test_wallet_creation_slot_max_roundtrip() {
        let mut wallet = make_wallet(0x02);
        wallet.creation_slot = u64::MAX;
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.creation_slot, u64::MAX);
    }

    #[test]
    fn test_wallet_runtime_deserialize_still_rejects_invalid_sig_scheme() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        buf[34] = 1; // invalid sig_scheme for v0 (Ed25519 not allowed in v0 header)
                     // Runtime deserialize also validates sig_scheme for v0
        let result = MachineWallet::deserialize_runtime(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_runtime_deserialize_still_rejects_zero_threshold() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        buf[35] = 0; // zero threshold
        let result = MachineWallet::deserialize_runtime(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_wallet_runtime_deserialize_still_rejects_invalid_version() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        buf[0] = 9;

        let result = MachineWallet::deserialize_runtime(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    // ==================== V1 Tests ====================

    #[test]
    fn test_v1_roundtrip_single_authority() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        let size = MachineWallet::v1_account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.bump, 254);
        assert_eq!(decoded.threshold, 1);
        assert_eq!(decoded.authority_count, 1);
        assert_eq!(decoded.authorities[0], auth);
        assert_eq!(decoded.nonce, 0x0102030405060708);
        assert_eq!(decoded.creation_slot, 42);
        assert_eq!(decoded.vault_bump, 253);
    }

    #[test]
    fn test_v1_roundtrip_three_authorities_mixed() {
        let auth0 = make_p256_authority(0x02);
        let auth1 = make_ed25519_authority(0xAA);
        let auth2 = make_p256_authority(0x03);
        let mut wallet = make_v1_wallet(3, &[auth0, auth1, auth2]);
        wallet.threshold = 2;

        let size = MachineWallet::v1_account_size(3);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.threshold, 2);
        assert_eq!(decoded.authority_count, 3);
        assert_eq!(decoded.authorities[0], auth0);
        assert_eq!(decoded.authorities[1], auth1);
        assert_eq!(decoded.authorities[2], auth2);
        assert_eq!(decoded.nonce, wallet.nonce);
        assert_eq!(decoded.creation_slot, wallet.creation_slot);
        assert_eq!(decoded.vault_bump, wallet.vault_bump);
    }

    #[test]
    fn test_v1_roundtrip_max_authorities() {
        let mut auths = Vec::new();
        for i in 0..MAX_AUTHORITIES {
            if i % 2 == 0 {
                auths.push(make_p256_authority(if i % 4 == 0 { 0x02 } else { 0x03 }));
            } else {
                auths.push(make_ed25519_authority(i + 1));
            }
        }
        let mut wallet = make_v1_wallet(MAX_AUTHORITIES, &auths);
        wallet.threshold = MAX_AUTHORITIES;

        let size = MachineWallet::v1_account_size(MAX_AUTHORITIES);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.authority_count, MAX_AUTHORITIES);
        assert_eq!(decoded.threshold, MAX_AUTHORITIES);
        for i in 0..MAX_AUTHORITIES as usize {
            assert_eq!(decoded.authorities[i], auths[i]);
        }
    }

    #[test]
    fn test_v1_rejects_zero_authority_count() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);

        // Manually construct buffer and override authority_count to 0
        let size = MachineWallet::v1_account_size(1); // still need space
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();
        buf[35] = 0; // zero authority_count

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_v1_rejects_zero_threshold() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        let size = MachineWallet::v1_account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();
        buf[34] = 0; // zero threshold

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_v1_rejects_threshold_exceeds_authority_count() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        let size = MachineWallet::v1_account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();
        buf[34] = 2; // threshold > authority_count (1)

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_v1_account_size() {
        assert_eq!(MachineWallet::v1_account_size(1), 53 + 34); // 87
        assert_eq!(MachineWallet::v1_account_size(2), 53 + 68); // 121
        assert_eq!(MachineWallet::v1_account_size(3), 53 + 102); // 155
        assert_eq!(MachineWallet::v1_account_size(16), 53 + 544); // 597
    }

    #[test]
    fn test_v1_serialize_too_short() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        let mut buf = [0u8; 50]; // too small
        let result = wallet.serialize_v1(&mut buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    // ==================== AuthoritySlot Validation ====================

    #[test]
    fn test_authority_slot_valid_p256() {
        let auth = make_p256_authority(0x02);
        assert!(auth.is_valid());

        let auth = make_p256_authority(0x03);
        assert!(auth.is_valid());
    }

    #[test]
    fn test_authority_slot_valid_ed25519() {
        let auth = make_ed25519_authority(0xAA);
        assert!(auth.is_valid());
    }

    #[test]
    fn test_authority_slot_rejects_all_zero_ed25519() {
        let pubkey = [0u8; 33];
        // All 33 bytes are 0 — first 32 are zero key, byte 32 is valid padding
        let auth = AuthoritySlot {
            sig_scheme: SIG_SCHEME_ED25519,
            pubkey,
        };
        assert!(!auth.is_valid());
    }

    #[test]
    fn test_authority_slot_rejects_bad_padding_ed25519() {
        let mut pubkey = [0xAA; 33];
        pubkey[32] = 0x01; // bad padding byte
        let auth = AuthoritySlot {
            sig_scheme: SIG_SCHEME_ED25519,
            pubkey,
        };
        assert!(!auth.is_valid());
    }

    #[test]
    fn test_authority_slot_accepts_low_order_ed25519() {
        // Low-order points pass format check — see is_valid_ed25519 doc comment.
        let mut pubkey = [0u8; 33];
        pubkey[0] = 1;
        let auth = AuthoritySlot {
            sig_scheme: SIG_SCHEME_ED25519,
            pubkey,
        };
        assert!(auth.is_valid());
    }

    #[test]
    fn test_authority_slot_rejects_unknown_scheme() {
        let auth = AuthoritySlot {
            sig_scheme: 99,
            pubkey: [0xAA; 33],
        };
        assert!(!auth.is_valid());
    }

    // ==================== Version-Aware Offset Methods ====================

    #[test]
    fn test_version_aware_offsets_v0() {
        let wallet = make_wallet(0x02);
        assert_eq!(wallet.nonce_offset(), 70);
        assert_eq!(wallet.creation_slot_offset(), 78);
        assert_eq!(wallet.vault_bump_offset(), 86);
    }

    #[test]
    fn test_version_aware_offsets_v1() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        assert_eq!(wallet.nonce_offset(), 36);
        assert_eq!(wallet.creation_slot_offset(), 44);
        assert_eq!(wallet.vault_bump_offset(), 52);
    }

    // ==================== deserialize_runtime for both versions ====================

    #[test]
    fn test_v0_runtime_deserialize() {
        let wallet = make_wallet(0x02);
        let mut buf = [0u8; MachineWallet::LEN];
        wallet.serialize(&mut buf).unwrap();
        let decoded = MachineWallet::deserialize_runtime(&buf).unwrap();
        assert_eq!(decoded.version, 0);
        assert_eq!(decoded.nonce, wallet.nonce);
    }

    #[test]
    fn test_v1_runtime_deserialize() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        let size = MachineWallet::v1_account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize_runtime(&buf).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.authority_count, 1);
        assert_eq!(decoded.nonce, wallet.nonce);
    }

    #[test]
    fn test_v1_runtime_deserialize_skips_validation() {
        // Invalid P-256 key should pass runtime deserialize (no validation)
        let mut auth = make_p256_authority(0x02);
        auth.pubkey = [0xFF; 33]; // invalid
        let wallet = make_v1_wallet(1, &[auth]);
        let size = MachineWallet::v1_account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();

        // Runtime should succeed (skips validation)
        let decoded = MachineWallet::deserialize_runtime(&buf).unwrap();
        assert_eq!(decoded.authorities[0].pubkey, [0xFF; 33]);

        // Full deserialize should fail (validates)
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_v1_deserialize_too_short_for_authorities() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        let size = MachineWallet::v1_account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize_v1(&mut buf).unwrap();
        // Claim 3 authorities but buffer is only sized for 1
        buf[35] = 3;

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    #[test]
    fn test_v1_rejects_authority_count_exceeds_max() {
        let auth = make_p256_authority(0x02);
        let wallet = make_v1_wallet(1, &[auth]);
        // Create a big enough buffer
        let mut buf = vec![0u8; V1_HEADER_SIZE + 17 * AUTHORITY_SLOT_SIZE];
        wallet.serialize_v1(&mut buf).unwrap();
        buf[35] = 17; // exceeds MAX_AUTHORITIES (16)

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_empty_buffer_rejected() {
        let buf: [u8; 0] = [];
        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);

        let result = MachineWallet::deserialize_runtime(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    // --- SessionState tests ---

    fn make_session() -> SessionState {
        let mut wallet = [0u8; 32];
        wallet[0] = 0xAA;
        let mut authority = [0u8; 32];
        authority[0] = 0xBB;
        let mut prog0 = [0u8; 32];
        prog0[0] = 0x11;
        let mut prog1 = [0u8; 32];
        prog1[0] = 0x22;
        let mut allowed_programs = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
        allowed_programs[0] = prog0;
        allowed_programs[1] = prog1;
        SessionState {
            version: SESSION_STATE_VERSION,
            bump: 253,
            wallet,
            authority,
            created_slot: 100,
            expiry_slot: 200,
            revoked: false,
            wallet_creation_slot: 50,
            max_lamports_per_ix: 1_000_000,
            allowed_programs_count: 2,
            allowed_programs,
        }
    }

    #[test]
    fn test_session_state_size() {
        assert_eq!(SessionState::LEN, 420);
    }

    #[test]
    fn test_session_state_roundtrip() {
        let session = make_session();
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();

        let decoded = SessionState::deserialize(&buf).unwrap();
        assert_eq!(session, decoded);
        assert_eq!(decoded.version, SESSION_STATE_VERSION);
        assert_eq!(decoded.bump, 253);
        assert_eq!(decoded.created_slot, 100);
        assert_eq!(decoded.expiry_slot, 200);
        assert!(!decoded.revoked);
        assert_eq!(decoded.wallet_creation_slot, 50);
        assert_eq!(decoded.max_lamports_per_ix, 1_000_000);
        assert_eq!(decoded.allowed_programs_count, 2);
    }

    #[test]
    fn test_session_is_program_allowed() {
        let session = make_session();
        let mut allowed = [0u8; 32];
        allowed[0] = 0x11;
        let mut not_allowed = [0u8; 32];
        not_allowed[0] = 0xFF;

        assert!(session.is_program_allowed(&allowed));
        assert!(!session.is_program_allowed(&not_allowed));
    }

    #[test]
    fn test_session_rejects_too_many_programs() {
        let session = make_session();
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        // Set allowed_programs_count to 9 (> MAX_ALLOWED_PROGRAMS = 8)
        buf[99] = 9;
        let result = SessionState::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_session_revoked_flag() {
        // revoked = false roundtrip
        let session = make_session();
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        let decoded = SessionState::deserialize(&buf).unwrap();
        assert!(!decoded.revoked);

        // revoked = true roundtrip
        let mut revoked_session = session.clone();
        revoked_session.revoked = true;
        let mut buf2 = [0u8; SessionState::LEN];
        revoked_session.serialize(&mut buf2).unwrap();
        assert_eq!(buf2[SessionState::REVOKED_OFFSET], 1);
        let decoded2 = SessionState::deserialize(&buf2).unwrap();
        assert!(decoded2.revoked);
    }

    #[test]
    fn test_session_rejects_zero_allowed_programs_count() {
        let session = make_session();
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        buf[99] = 0;
        let result = SessionState::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_session_rejects_zero_wallet_or_authority() {
        let mut session = make_session();
        session.wallet = [0u8; 32];
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );

        let mut session = make_session();
        session.authority = [0u8; 32];
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );
    }

    #[test]
    fn test_session_rejects_duplicate_program_ids() {
        let mut session = make_session();
        session.allowed_programs[1] = session.allowed_programs[0];
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );
    }

    #[test]
    fn test_session_accepts_system_program() {
        // System Program address is [0u8; 32] — must be accepted as a valid allowed program
        let mut session = make_session();
        session.allowed_programs[0] = [0u8; 32]; // System Program
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        let decoded = SessionState::deserialize(&buf).unwrap();
        assert_eq!(decoded.allowed_programs[0], [0u8; 32]);
    }

    #[test]
    fn test_session_rejects_created_slot_after_expiry() {
        let mut session = make_session();
        session.created_slot = 201;
        session.expiry_slot = 200;
        let mut buf = [0u8; SessionState::LEN];
        session.serialize(&mut buf).unwrap();
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );
    }
}
