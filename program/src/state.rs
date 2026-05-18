use solana_program::{keccak, program_error::ProgramError};

/// SessionState version byte. Must remain 0 — wallet state uses
/// `MachineWallet::LAYOUT_VERSION` (≥1), so this value also serves as a
/// discriminator between account types.
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

/// WebAuthn Passkey signature scheme identifier.
/// Uses the same P-256 key format as SECP256R1, but with WebAuthn message wrapping.
pub const SIG_SCHEME_WEBAUTHN: u8 = 2;

/// Maximum number of authorities per wallet.
pub const MAX_AUTHORITIES: u8 = 16;

/// Size of a single AuthoritySlot: 1 (scheme) + 33 (pubkey).
pub const AUTHORITY_SLOT_SIZE: usize = 34;

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
            SIG_SCHEME_SECP256R1 | SIG_SCHEME_WEBAUTHN => {
                MachineWallet::is_valid_authority(&self.pubkey)
            }
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
/// Layout (53 + N×34 bytes, where N = authority_count):
/// - version:         u8       (offset 0)   = `LAYOUT_VERSION`
/// - bump:            u8       (offset 1)
/// - wallet_id:       [u8; 32] (offset 2)
/// - threshold:       u8       (offset 34)
/// - authority_count: u8       (offset 35)
/// - nonce:           u64      (offset 36)
/// - creation_slot:   u64      (offset 44)
/// - vault_bump:      u8       (offset 52)
/// - authorities:     [AuthoritySlot; N] (offset 53)
///
/// Each `AuthoritySlot` carries its own `sig_scheme`, so one wallet can mix
/// Secp256r1, Ed25519, and WebAuthn signers.
///
/// The version byte lives only on disk; `deserialize` enforces
/// `data[0] == LAYOUT_VERSION`, so the in-memory struct carries no copy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineWallet {
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
    /// On-chain layout version byte. `serialize` writes it into `data[0]`,
    /// `deserialize_*` requires equality, and any future migration bumps this
    /// constant in one place.
    pub const LAYOUT_VERSION: u8 = 1;

    /// Fixed header size before the authority slots:
    /// version(1) + bump(1) + wallet_id(32) + threshold(1) + authority_count(1)
    /// + nonce(8) + creation_slot(8) + vault_bump(1) = 53.
    pub const HEADER_SIZE: usize = 53;

    /// Byte offset of the threshold field within the serialized layout.
    pub const THRESHOLD_OFFSET: usize = 34;
    /// Byte offset of the authority_count field within the serialized layout.
    pub const AUTHORITY_COUNT_OFFSET: usize = 35;
    /// Byte offset of the nonce field within the serialized layout.
    pub const NONCE_OFFSET: usize = 36;
    /// Byte offset of the creation_slot field within the serialized layout.
    pub const CREATION_SLOT_OFFSET: usize = 44;
    /// Byte offset of the vault_bump field within the serialized layout.
    pub const VAULT_BUMP_OFFSET: usize = 52;

    /// Wallet PDA seed prefix.
    pub const SEED_PREFIX: &'static [u8] = b"machine_wallet";

    /// Vault PDA seed prefix.
    pub const VAULT_SEED_PREFIX: &'static [u8] = b"machine_vault";

    /// Per-Execute ephemeral signer PDA seed prefix.
    ///
    /// Seeds: `[EPHEMERAL_SIGNER_SEED_PREFIX, wallet_account_key, wallet.nonce.le_bytes(), index_u8, bump]`.
    ///
    /// Wallet nonce is monotonic and bumps on every Execute, so each Execute call
    /// produces a fresh set of PDAs; a relay can't replay a prior tx's ephemeral
    /// signer slot because the seeds (and thus the derived PDA) change. The
    /// PDA's only role is to provide `invoke_signed` signer privilege to one
    /// inner instruction — it never holds long-lived state.
    ///
    /// Squads v4 uses the same pattern but seeded on the multisig transaction
    /// PDA; machine-wallet has no per-tx state account, so we use the wallet's
    /// own monotonic nonce instead.
    pub const EPHEMERAL_SIGNER_SEED_PREFIX: &'static [u8] = b"machine_ephemeral";

    /// Increment the nonce in a serialized wallet buffer. Single point of
    /// change for nonce-bump semantics across every state-mutating processor.
    pub fn write_incremented_nonce(&self, data: &mut [u8]) -> Result<(), ProgramError> {
        let new_nonce = self
            .nonce
            .checked_add(1)
            .ok_or(crate::error::MachineWalletError::InvalidNonce)?;
        if data.len() < Self::NONCE_OFFSET + 8 {
            return Err(ProgramError::AccountDataTooSmall);
        }
        data[Self::NONCE_OFFSET..Self::NONCE_OFFSET + 8]
            .copy_from_slice(&new_nonce.to_le_bytes());
        Ok(())
    }

    /// Increment the creation_slot field. CloseWallet bumps this so any
    /// pre-existing session — whose `wallet_creation_slot` snapshot now
    /// mismatches — fails the SessionExecute check and cannot drain residual
    /// assets after a close.
    pub fn write_incremented_creation_slot(
        &self,
        data: &mut [u8],
    ) -> Result<(), ProgramError> {
        let new_creation_slot = self
            .creation_slot
            .checked_add(1)
            .ok_or(ProgramError::ArithmeticOverflow)?;
        if data.len() < Self::CREATION_SLOT_OFFSET + 8 {
            return Err(ProgramError::AccountDataTooSmall);
        }
        data[Self::CREATION_SLOT_OFFSET..Self::CREATION_SLOT_OFFSET + 8]
            .copy_from_slice(&new_creation_slot.to_le_bytes());
        Ok(())
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

    /// Calculate the required account size for a wallet with the given authority count.
    pub fn account_size(authority_count: u8) -> usize {
        Self::HEADER_SIZE + authority_count as usize * AUTHORITY_SLOT_SIZE
    }

    /// Byte offset of the i-th authority slot within the serialized layout.
    #[inline(always)]
    pub const fn slot_offset(i: usize) -> usize {
        Self::HEADER_SIZE + i * AUTHORITY_SLOT_SIZE
    }

    /// Serialize wallet state into a byte buffer.
    pub fn serialize(&self, dst: &mut [u8]) -> Result<(), ProgramError> {
        let required = Self::account_size(self.authority_count);
        if dst.len() < required {
            return Err(ProgramError::AccountDataTooSmall);
        }
        dst[0] = Self::LAYOUT_VERSION;
        dst[1] = self.bump;
        dst[2..34].copy_from_slice(&self.wallet_id);
        dst[Self::THRESHOLD_OFFSET] = self.threshold;
        dst[Self::AUTHORITY_COUNT_OFFSET] = self.authority_count;
        dst[Self::NONCE_OFFSET..Self::NONCE_OFFSET + 8]
            .copy_from_slice(&self.nonce.to_le_bytes());
        dst[Self::CREATION_SLOT_OFFSET..Self::CREATION_SLOT_OFFSET + 8]
            .copy_from_slice(&self.creation_slot.to_le_bytes());
        dst[Self::VAULT_BUMP_OFFSET] = self.vault_bump;
        for i in 0..self.authority_count as usize {
            let offset = Self::slot_offset(i);
            dst[offset] = self.authorities[i].sig_scheme;
            dst[offset + 1..offset + 1 + 33].copy_from_slice(&self.authorities[i].pubkey);
        }
        Ok(())
    }

    fn deserialize_inner(src: &[u8], validate: bool) -> Result<Self, ProgramError> {
        if src.len() < Self::HEADER_SIZE {
            return Err(ProgramError::AccountDataTooSmall);
        }

        if src[0] != Self::LAYOUT_VERSION {
            return Err(ProgramError::InvalidAccountData);
        }

        let bump = src[1];

        let mut wallet_id = [0u8; 32];
        wallet_id.copy_from_slice(&src[2..34]);

        let threshold = src[Self::THRESHOLD_OFFSET];
        let authority_count = src[Self::AUTHORITY_COUNT_OFFSET];

        if threshold == 0 || authority_count == 0 {
            return Err(ProgramError::InvalidAccountData);
        }
        if threshold > authority_count {
            return Err(ProgramError::InvalidAccountData);
        }
        if authority_count > MAX_AUTHORITIES {
            return Err(ProgramError::InvalidAccountData);
        }

        // Exact-length equality rejects trailing bytes. Combined with the
        // realloc path that always sizes to `account_size(authority_count)`,
        // this means the on-chain account byte-length must mirror the
        // declared `authority_count` — no stale-slot shadowing possible.
        let required_size = Self::account_size(authority_count);
        if src.len() != required_size {
            return Err(ProgramError::AccountDataTooSmall);
        }

        // Length already validated ≥ required_size; these slices are guaranteed 8 bytes.
        let nonce = u64::from_le_bytes(
            src[Self::NONCE_OFFSET..Self::NONCE_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let creation_slot = u64::from_le_bytes(
            src[Self::CREATION_SLOT_OFFSET..Self::CREATION_SLOT_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let vault_bump = src[Self::VAULT_BUMP_OFFSET];

        let mut authorities = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        for (i, slot) in authorities
            .iter_mut()
            .take(authority_count as usize)
            .enumerate()
        {
            let offset = Self::slot_offset(i);
            let sig_scheme = src[offset];
            // Runtime-path validation of sig_scheme: only the three declared
            // schemes are accepted for active slots. An unknown value would
            // silently skip threshold matching (the scanner compares against
            // explicit constants), allowing a corrupted-storage bug to lock
            // users out of their wallet. Reject at deserialize rather than
            // letting the invariant leak into match-time code.
            if sig_scheme != SIG_SCHEME_SECP256R1
                && sig_scheme != SIG_SCHEME_ED25519
                && sig_scheme != SIG_SCHEME_WEBAUTHN
            {
                return Err(ProgramError::InvalidAccountData);
            }
            slot.sig_scheme = sig_scheme;
            slot.pubkey.copy_from_slice(&src[offset + 1..offset + 1 + 33]);
            if validate && !slot.is_valid() {
                return Err(ProgramError::InvalidAccountData);
            }
        }

        Ok(Self {
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

    /// Deserialize from a byte buffer with full integrity checks (validates
    /// authority pubkeys). `deserialize_inner` enforces version byte equality
    /// against `LAYOUT_VERSION`, so any unknown byte fails closed.
    pub fn deserialize(src: &[u8]) -> Result<Self, ProgramError> {
        if src.is_empty() {
            return Err(ProgramError::AccountDataTooSmall);
        }
        Self::deserialize_inner(src, true)
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
        Self::deserialize_inner(src, false)
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

/// Maximum number of ephemeral signer PDAs an Execute call may request.
///
/// Each adds 32 bytes to outer-tx account_keys (or 1 byte if covered by an
/// ALT lookup) plus one entry in the `invoke_signed` signer-seeds array.
/// 4 covers every known use case (Switchboard Randomness.create needs 1,
/// SystemProgram.createAccount needs 1, token-metadata Master Edition needs
/// up to 2) with room to spare.
pub const MAX_EPHEMERAL_SIGNERS: usize = 4;

/// Session PDA seed prefix.
pub const SESSION_SEED_PREFIX: &[u8] = b"machine_session";

/// SessionState on-chain state — **variable size** based on `allowed_programs_count`.
///
/// Dynamic layout (116 + count*32 bytes total):
/// - version:                    u8          (offset 0)                 — state version, must be 0
/// - bump:                       u8          (offset 1)                 — session PDA bump seed
/// - wallet:                     [u8; 32]    (offset 2)                 — wallet PDA address
/// - authority:                  [u8; 32]    (offset 34)                — session authority (Ed25519 pubkey)
/// - created_slot:               u64         (offset 66)                — slot at creation
/// - expiry_slot:                u64         (offset 74)                — slot at which session expires
/// - revoked:                    bool/u8     (offset 82)                — non-zero means revoked
/// - wallet_creation_slot:       u64         (offset 83)                — wallet's creation_slot, prevents session resurrection
/// - max_lamports_per_call:      u64         (offset 91)                — net outflow cap per SessionExecute invocation (0 = unlimited)
/// - allowed_programs_count:     u8          (offset 99)                — number of whitelisted programs (1..=MAX_ALLOWED_PROGRAMS)
/// - allowed_programs:           [[u8;32];N] (offset 100)               — exactly `count` × 32 bytes
/// - max_total_spent_lamports:   u64         (offset 100 + count*32)    — lifetime cumulative outflow cap (0 = unlimited)
/// - total_spent_lamports:       u64         (offset 108 + count*32)    — cumulative lamports withdrawn via this session
///
/// **Rent optimization**: a 1-program session is 148 B (vs previous fixed 420 B),
/// saving ~272 B × rent_per_byte per session. 8-program session is 372 B.
///
/// **Length-as-integrity**: `deserialize_inner` enforces `src.len() == size(count)`.
/// Combined with program-owned-account writes only, this prevents any stale-byte
/// shadowing or trailing-byte malleability.
///
/// Spend-cap semantics:
/// - `max_lamports_per_call` bounds the net vault outflow per single SessionExecute
///   invocation. It does NOT aggregate across multiple SessionExecute instructions
///   packed in the same transaction — each call is independently capped.
/// - `max_total_spent_lamports` bounds the cumulative lifetime outflow across all
///   SessionExecute calls made under this session. Enforced only when > 0.
///   When enforced, `session_account` must be writable so the processor can
///   persist the updated `total_spent_lamports` counter.
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
    pub max_lamports_per_call: u64,
    pub allowed_programs_count: u8,
    pub allowed_programs: [[u8; 32]; MAX_ALLOWED_PROGRAMS],
    pub max_total_spent_lamports: u64,
    pub total_spent_lamports: u64,
}

impl SessionState {
    /// Size of the fixed header preceding the variable-length allowed_programs.
    /// Covers offsets 0..100 (version, bump, wallet, authority, created_slot,
    /// expiry_slot, revoked, wallet_creation_slot, max_lamports_per_call,
    /// allowed_programs_count).
    pub const HEADER_SIZE: usize = 100;

    /// Size of the tail following the variable-length allowed_programs region.
    /// Covers max_total_spent_lamports (8B) + total_spent_lamports (8B).
    pub const TAIL_SIZE: usize = 16;

    /// Byte offset of the revoked field within serialized state (static).
    pub const REVOKED_OFFSET: usize = 82;

    /// Minimum serialized size (a session with exactly 1 allowed program).
    /// 1-program session is the legal minimum because allowed_programs_count
    /// must be ≥ 1.
    pub const MIN_SIZE: usize = Self::HEADER_SIZE + 32 + Self::TAIL_SIZE;

    /// Maximum serialized size (a session with MAX_ALLOWED_PROGRAMS programs).
    pub const MAX_SIZE: usize = Self::HEADER_SIZE + MAX_ALLOWED_PROGRAMS * 32 + Self::TAIL_SIZE;

    /// Total serialized size for a session with the given `allowed_programs_count`.
    /// Callers (CreateSession) compute the exact rent requirement via this.
    #[inline]
    pub const fn size(allowed_programs_count: u8) -> usize {
        Self::HEADER_SIZE + (allowed_programs_count as usize) * 32 + Self::TAIL_SIZE
    }

    /// Byte offset of `max_total_spent_lamports` within serialized state.
    /// Depends on count because the field sits after the variable-length
    /// allowed_programs region.
    #[inline]
    pub const fn max_total_spent_offset(allowed_programs_count: u8) -> usize {
        Self::HEADER_SIZE + (allowed_programs_count as usize) * 32
    }

    /// Byte offset of `total_spent_lamports` within serialized state.
    /// Hot-path writers (session_execute) update this in place without
    /// re-serializing the whole struct.
    #[inline]
    pub const fn total_spent_offset(allowed_programs_count: u8) -> usize {
        Self::max_total_spent_offset(allowed_programs_count) + 8
    }

    /// In-place update of `total_spent_lamports` in a serialized session
    /// account buffer. Lets callers (session_execute hot path) persist the
    /// cumulative spend counter without re-serializing the whole struct and
    /// without depending on the raw byte layout.
    pub fn write_total_spent(
        data: &mut [u8],
        allowed_programs_count: u8,
        new_total: u64,
    ) -> Result<(), ProgramError> {
        let off = Self::total_spent_offset(allowed_programs_count);
        if data.len() < off + 8 {
            return Err(ProgramError::AccountDataTooSmall);
        }
        data[off..off + 8].copy_from_slice(&new_total.to_le_bytes());
        Ok(())
    }

    /// Convenience: serialized size for this instance.
    #[inline]
    pub fn serialized_size(&self) -> usize {
        Self::size(self.allowed_programs_count)
    }

    /// Serialize into a byte buffer sized exactly for this session's
    /// `allowed_programs_count`. No reserved tail — the buffer holds only
    /// active fields.
    pub fn serialize(&self, dst: &mut [u8]) -> Result<(), ProgramError> {
        let required = self.serialized_size();
        if dst.len() < required {
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
        dst[91..99].copy_from_slice(&self.max_lamports_per_call.to_le_bytes());
        dst[99] = self.allowed_programs_count;
        for (i, prog) in self
            .allowed_programs
            .iter()
            .take(self.allowed_programs_count as usize)
            .enumerate()
        {
            let start = Self::HEADER_SIZE + i * 32;
            dst[start..start + 32].copy_from_slice(prog);
        }
        let max_off = Self::max_total_spent_offset(self.allowed_programs_count);
        let total_off = Self::total_spent_offset(self.allowed_programs_count);
        dst[max_off..max_off + 8].copy_from_slice(&self.max_total_spent_lamports.to_le_bytes());
        dst[total_off..total_off + 8].copy_from_slice(&self.total_spent_lamports.to_le_bytes());
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
        // Minimum-size gate: must at least hold the fixed header so we can
        // safely read the count byte at offset 99.
        if src.len() < Self::HEADER_SIZE {
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

        // Header ≥ 100 already verified; these slices are guaranteed 8 bytes.
        let created_slot = u64::from_le_bytes(src[66..74].try_into().unwrap());
        let expiry_slot = u64::from_le_bytes(src[74..82].try_into().unwrap());

        let revoked = src[Self::REVOKED_OFFSET] != 0;

        let wallet_creation_slot = u64::from_le_bytes(src[83..91].try_into().unwrap());
        let max_lamports_per_call = u64::from_le_bytes(src[91..99].try_into().unwrap());

        let allowed_programs_count = src[99];
        if allowed_programs_count == 0 || allowed_programs_count as usize > MAX_ALLOWED_PROGRAMS {
            return Err(ProgramError::InvalidAccountData);
        }

        // Exact-length equality once count is known. Combined with program-
        // owned writes that always size to `size(count)`, this means the
        // on-chain byte length must mirror the declared count — no stale-slot
        // shadowing, no trailing-byte malleability.
        let required_size = Self::size(allowed_programs_count);
        if src.len() != required_size {
            return Err(ProgramError::AccountDataTooSmall);
        }

        if wallet == [0u8; 32] || authority == [0u8; 32] || created_slot > expiry_slot {
            return Err(ProgramError::InvalidAccountData);
        }

        let mut allowed_programs = [[0u8; 32]; MAX_ALLOWED_PROGRAMS];
        for i in 0..allowed_programs_count as usize {
            let start = Self::HEADER_SIZE + i * 32;
            allowed_programs[i].copy_from_slice(&src[start..start + 32]);
            if validate {
                for prev in allowed_programs.iter().take(i) {
                    if *prev == allowed_programs[i] {
                        return Err(ProgramError::InvalidAccountData);
                    }
                }
            }
        }

        let max_off = Self::max_total_spent_offset(allowed_programs_count);
        let total_off = Self::total_spent_offset(allowed_programs_count);
        let max_total_spent_lamports =
            u64::from_le_bytes(src[max_off..max_off + 8].try_into().unwrap());
        let total_spent_lamports =
            u64::from_le_bytes(src[total_off..total_off + 8].try_into().unwrap());
        // Invariant: total_spent must never exceed the cap (when cap > 0).
        // A violation implies a prior write bypassed the check — fail closed.
        if max_total_spent_lamports > 0 && total_spent_lamports > max_total_spent_lamports {
            return Err(ProgramError::InvalidAccountData);
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
            max_lamports_per_call,
            allowed_programs_count,
            allowed_programs,
            max_total_spent_lamports,
            total_spent_lamports,
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

    fn make_wallet(authority_count: u8, authorities: &[AuthoritySlot]) -> MachineWallet {
        let wallet_id = MachineWallet::compute_id(&authorities[0].pubkey);
        let mut auth_arr = [AuthoritySlot::EMPTY; MAX_AUTHORITIES as usize];
        for (i, a) in authorities.iter().enumerate() {
            auth_arr[i] = *a;
        }
        MachineWallet {
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

    #[test]
    fn test_compute_id() {
        let mut authority = [0u8; 33];
        authority[0] = 0x02;
        authority[1..].copy_from_slice(&BASEPOINT_X);
        let id = MachineWallet::compute_id(&authority);
        assert_eq!(id, keccak::hash(&authority).to_bytes());
    }

    #[test]
    fn test_roundtrip_single_authority() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let size = MachineWallet::account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(buf[0], MachineWallet::LAYOUT_VERSION);
        assert_eq!(decoded.bump, 254);
        assert_eq!(decoded.threshold, 1);
        assert_eq!(decoded.authority_count, 1);
        assert_eq!(decoded.authorities[0], auth);
        assert_eq!(decoded.nonce, 0x0102030405060708);
        assert_eq!(decoded.creation_slot, 42);
        assert_eq!(decoded.vault_bump, 253);
    }

    #[test]
    fn test_roundtrip_three_authorities_mixed() {
        let auth0 = make_p256_authority(0x02);
        let auth1 = make_ed25519_authority(0xAA);
        let auth2 = make_p256_authority(0x03);
        let mut wallet = make_wallet(3, &[auth0, auth1, auth2]);
        wallet.threshold = 2;

        let size = MachineWallet::account_size(3);
        let mut buf = vec![0u8; size];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(buf[0], MachineWallet::LAYOUT_VERSION);
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
    fn test_roundtrip_max_authorities() {
        let mut auths = Vec::new();
        for i in 0..MAX_AUTHORITIES {
            if i % 2 == 0 {
                auths.push(make_p256_authority(if i % 4 == 0 { 0x02 } else { 0x03 }));
            } else {
                auths.push(make_ed25519_authority(i + 1));
            }
        }
        let mut wallet = make_wallet(MAX_AUTHORITIES, &auths);
        wallet.threshold = MAX_AUTHORITIES;

        let size = MachineWallet::account_size(MAX_AUTHORITIES);
        let mut buf = vec![0u8; size];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize(&buf).unwrap();
        assert_eq!(decoded.authority_count, MAX_AUTHORITIES);
        assert_eq!(decoded.threshold, MAX_AUTHORITIES);
        for i in 0..MAX_AUTHORITIES as usize {
            assert_eq!(decoded.authorities[i], auths[i]);
        }
    }

    #[test]
    fn test_rejects_zero_authority_count() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let size = MachineWallet::account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize(&mut buf).unwrap();
        buf[35] = 0; // zero authority_count

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_rejects_zero_threshold() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let size = MachineWallet::account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize(&mut buf).unwrap();
        buf[34] = 0; // zero threshold

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_rejects_threshold_exceeds_authority_count() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let size = MachineWallet::account_size(1);
        let mut buf = vec![0u8; size];
        wallet.serialize(&mut buf).unwrap();
        buf[34] = 2; // threshold > authority_count (1)

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_account_size() {
        assert_eq!(MachineWallet::account_size(1), 53 + 34); // 87
        assert_eq!(MachineWallet::account_size(2), 53 + 68); // 121
        assert_eq!(MachineWallet::account_size(3), 53 + 102); // 155
        assert_eq!(MachineWallet::account_size(16), 53 + 544); // 597
    }

    #[test]
    fn test_serialize_too_short() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let mut buf = [0u8; 50]; // too small
        let result = wallet.serialize(&mut buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    #[test]
    fn test_wallet_id_stored() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let expected_id = keccak::hash(&wallet.authorities[0].pubkey).to_bytes();
        assert_eq!(wallet.id(), expected_id);
        assert_eq!(wallet.wallet_id, expected_id);
    }

    #[test]
    fn test_rejects_invalid_version() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let mut buf = vec![0u8; MachineWallet::account_size(1)];
        wallet.serialize(&mut buf).unwrap();
        buf[0] = 99; // unknown version
        assert_eq!(
            MachineWallet::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );
        assert_eq!(
            MachineWallet::deserialize_runtime(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );
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

    #[test]
    fn test_runtime_deserialize_skips_validation() {
        // Invalid P-256 key should pass runtime deserialize (no validation),
        // but full deserialize must still reject.
        let mut auth = make_p256_authority(0x02);
        auth.pubkey = [0xFF; 33];
        let wallet = make_wallet(1, &[auth]);
        let mut buf = vec![0u8; MachineWallet::account_size(1)];
        wallet.serialize(&mut buf).unwrap();

        let decoded = MachineWallet::deserialize_runtime(&buf).unwrap();
        assert_eq!(decoded.authorities[0].pubkey, [0xFF; 33]);

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_deserialize_too_short_for_authorities() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        let mut buf = vec![0u8; MachineWallet::account_size(1)];
        wallet.serialize(&mut buf).unwrap();
        // Claim 3 authorities but buffer is only sized for 1
        buf[35] = 3;

        let result = MachineWallet::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::AccountDataTooSmall);
    }

    #[test]
    fn test_rejects_authority_count_exceeds_max() {
        let auth = make_p256_authority(0x02);
        let wallet = make_wallet(1, &[auth]);
        // Buffer large enough to claim authority_count = 17
        let mut buf = vec![0u8; MachineWallet::HEADER_SIZE + 17 * AUTHORITY_SLOT_SIZE];
        wallet.serialize(&mut buf).unwrap();
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
            max_lamports_per_call: 1_000_000,
            allowed_programs_count: 2,
            allowed_programs,
            max_total_spent_lamports: 0,
            total_spent_lamports: 0,
        }
    }

    #[test]
    fn test_session_state_size() {
        // Dynamic layout: 116 + count*32.
        assert_eq!(SessionState::MIN_SIZE, 148); // 1 program
        assert_eq!(SessionState::MAX_SIZE, 372); // 8 programs
        assert_eq!(SessionState::size(1), 148);
        assert_eq!(SessionState::size(2), 180);
        assert_eq!(SessionState::size(8), 372);
        // 1-program session saves 272 bytes of rent vs the former fixed 420-byte layout.
    }

    #[test]
    fn test_session_offsets_are_count_dependent() {
        // Tail field offsets shift with allowed_programs_count.
        assert_eq!(SessionState::max_total_spent_offset(1), 132);
        assert_eq!(SessionState::total_spent_offset(1), 140);
        assert_eq!(SessionState::max_total_spent_offset(8), 356);
        assert_eq!(SessionState::total_spent_offset(8), 364);
    }

    #[test]
    fn test_session_rejects_length_mismatch() {
        // A byte buffer that passes count validation but is one byte shorter
        // than `size(count)` must be rejected — the length-as-integrity check
        // is load-bearing for preventing stale-slot shadowing.
        let session = make_session(); // count == 2 → size == 180
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        buf.pop(); // now 179 bytes, count byte still claims 2 → mismatch
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::AccountDataTooSmall
        );
    }

    #[test]
    fn test_session_state_roundtrip() {
        let session = make_session();
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();

        let decoded = SessionState::deserialize(&buf).unwrap();
        assert_eq!(session, decoded);
        assert_eq!(decoded.version, SESSION_STATE_VERSION);
        assert_eq!(decoded.bump, 253);
        assert_eq!(decoded.created_slot, 100);
        assert_eq!(decoded.expiry_slot, 200);
        assert!(!decoded.revoked);
        assert_eq!(decoded.wallet_creation_slot, 50);
        assert_eq!(decoded.max_lamports_per_call, 1_000_000);
        assert_eq!(decoded.max_total_spent_lamports, 0);
        assert_eq!(decoded.total_spent_lamports, 0);
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
        let mut buf = vec![0u8; session.serialized_size()];
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
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        let decoded = SessionState::deserialize(&buf).unwrap();
        assert!(!decoded.revoked);

        // revoked = true roundtrip
        let mut revoked_session = session.clone();
        revoked_session.revoked = true;
        let mut buf2 = vec![0u8; revoked_session.serialized_size()];
        revoked_session.serialize(&mut buf2).unwrap();
        assert_eq!(buf2[SessionState::REVOKED_OFFSET], 1);
        let decoded2 = SessionState::deserialize(&buf2).unwrap();
        assert!(decoded2.revoked);
    }

    #[test]
    fn test_session_rejects_zero_allowed_programs_count() {
        let session = make_session();
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        buf[99] = 0;
        let result = SessionState::deserialize(&buf);
        assert_eq!(result.unwrap_err(), ProgramError::InvalidAccountData);
    }

    #[test]
    fn test_session_rejects_zero_wallet_or_authority() {
        let mut session = make_session();
        session.wallet = [0u8; 32];
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );

        let mut session = make_session();
        session.authority = [0u8; 32];
        let mut buf = vec![0u8; session.serialized_size()];
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
        let mut buf = vec![0u8; session.serialized_size()];
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
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        let decoded = SessionState::deserialize(&buf).unwrap();
        assert_eq!(decoded.allowed_programs[0], [0u8; 32]);
    }

    #[test]
    fn test_session_rejects_created_slot_after_expiry() {
        let mut session = make_session();
        session.created_slot = 201;
        session.expiry_slot = 200;
        let mut buf = vec![0u8; session.serialized_size()];
        session.serialize(&mut buf).unwrap();
        assert_eq!(
            SessionState::deserialize(&buf).unwrap_err(),
            ProgramError::InvalidAccountData
        );
    }
}
