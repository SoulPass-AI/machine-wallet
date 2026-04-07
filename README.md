# MachineWallet

Solana smart wallet program with multi-sig threshold verification (Secp256r1 + Ed25519).

Hardware-secured key management for AI agents and autonomous systems.

## Overview

MachineWallet is an on-chain program that enables hardware-backed wallets on Solana using P-256 (Secp256r1) signatures from secure enclaves (Apple SE, Android StrongBox, YubiKey). It supports multi-authority threshold signing and delegated session keys for high-frequency operations.

**Program ID:** `7VD7mx5bYgmSJY7D1etvADEdDXijdp3UMz79M53vTdMo`

**Deployed:** Devnet + Mainnet (Agave v3.x)

## Architecture

Single program, three PDA types:

| PDA | Seeds | Owner | Purpose |
|-----|-------|-------|---------|
| Wallet | `["machine_wallet", id]` | Program | Stores authorities, threshold, nonce |
| Vault | `["machine_vault", wallet_address]` | System Program | Holds SOL, signs CPI via `invoke_signed` |
| Session | `["machine_session", wallet_address, session_authority]` | Program | Delegated session key config |

The vault stays system-owned so `system_program::transfer` CPI works naturally. Signature verification uses Solana's native Secp256r1/Ed25519 precompile instruction introspection.

## Instructions

| Instruction | Description |
|-------------|-------------|
| `CreateWallet` | Initialize wallet with P-256 or Ed25519 authority |
| `Execute` | Execute CPI calls, verified by threshold signatures |
| `SessionExecute` | Execute CPI calls using a delegated session key |
| `CreateSession` | Create a session key with expiry, spend cap, and program whitelist |
| `RevokeSession` | Owner revokes a session (threshold-signed) |
| `SelfRevokeSession` | Session authority revokes its own session |
| `CloseSession` | Close expired/revoked session, reclaim rent |
| `OwnerCloseSession` | Owner closes a session, reclaim rent |
| `AddAuthority` | Add a new authority (P-256 or Ed25519) to the wallet |
| `RemoveAuthority` | Remove an authority from the wallet |
| `SetThreshold` | Change the signature threshold |
| `CloseWallet` | Close wallet and vault, recover all funds |
| `AdvanceNonce` | Advance nonce to cancel a pending signed operation |

## Security Properties

- **Anti-reentry:** `get_stack_height()` rejects all CPI invocations of Execute/SessionExecute
- **Anti-self-call:** CPI to own program_id is denied
- **CEI pattern:** Nonce incremented before CPI loop
- **Signature expiry:** `max_slot` enforces time-bound signatures
- **Replay protection:** `nonce` (u64) + `creation_slot` (prevents replay after close+recreate)
- **Domain separation:** Distinct `machine_wallet_*_v0` tags per instruction type
- **Message binding:** `keccak256(domain_tag || wallet || creation_slot || nonce || max_slot || payload)`
- **Account permission commitment:** `is_writable` flags included in signed inner instruction hash
- **Session isolation:** Program whitelist, per-instruction lamport cap, expiry slot, `wallet_creation_slot` binding
- **Authority validation:** P-256 compressed-point format check + Ed25519 validation at creation
- **Threshold verification:** Bitmap dedup prevents double-counting; each authority counted at most once

## Quantum-Resistant Ready

The multi-authority architecture is designed for post-quantum migration with zero wallet disruption:

1. `sig_scheme` field in each `AuthoritySlot` supports arbitrary signature algorithms (currently `0 = Secp256r1`, `1 = Ed25519`)
2. Adding a post-quantum authority (e.g., ML-DSA / Dilithium) is a single `AddAuthority` call — no wallet migration, no asset movement
3. `wallet_id` is fixed at creation (keccak256 of initial authority) and never changes, so the wallet PDA address remains stable across authority and algorithm changes
4. Threshold signing allows mixed algorithm sets (e.g., 2-of-3 with one P-256 + one Ed25519 + one ML-DSA), enabling gradual transition without downtime

When Solana adds a post-quantum precompile, this program only needs a new `sig_scheme` constant and corresponding precompile introspection — the core architecture, state layout, and all existing wallets remain unchanged.

## Build

Requires [Agave CLI v3.x](https://github.com/anza-xyz/agave) (solana-program v3.0).

```bash
cargo build-sbf          # Build for Solana BPF
cargo test-sbf           # Run BPF tests
cargo test               # Run native unit tests
cargo clippy --all-targets  # Lint
```

## Trust Assumptions

- **Relay trust:** Inner instruction account indices (not pubkeys) are signed. The relay resolves indices to concrete accounts. This is safe when the relay is the machine itself. If the relay is untrusted, encode critical pubkeys in CPI calldata for target-program verification.
- **Session `max_lamports_per_ix`:** This is a per-instruction cap, not per-transaction. A single transaction may contain multiple `SessionExecute` instructions, each independently capped.
- **Runtime GC:** Close+recreate prevention relies on Solana not GC'ing zero-lamport accounts mid-transaction.

## License

[AGPL-3.0-or-later](LICENSE)

Copyright 2025-2026 SoulPass AI
