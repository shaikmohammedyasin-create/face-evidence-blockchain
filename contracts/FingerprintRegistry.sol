// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title FingerprintRegistry
/// @notice Stores SHA-256 fingerprints of discovered web/social posts.
///         NO personal or biometric data is ever stored on-chain.
///
/// Deploy once to Ethereum Sepolia (or any EVM chain) and reuse the
/// contract address in BLOCKCHAIN_CONTRACT_ADDRESS.
///
/// Deployment (Remix IDE — https://remix.ethereum.org):
///   1. Paste this file into Remix.
///   2. Compile with Solidity 0.8.20+.
///   3. Connect MetaMask (Sepolia network, funded with test-ETH).
///   4. Deploy.  Copy the contract address into .env.
contract FingerprintRegistry {
    /// @dev fingerprint (bytes32) → unix timestamp of first store
    mapping(bytes32 => uint256) public storedAt;

    event Stored(
        bytes32 indexed fingerprint,
        uint256 timestamp,
        string  projectId
    );

    /// @notice Record a fingerprint on-chain.  Idempotent: re-storing the
    ///         same fingerprint is a no-op (the original timestamp is kept).
    function store(bytes32 fingerprint, string calldata projectId) external {
        if (storedAt[fingerprint] == 0) {
            storedAt[fingerprint] = block.timestamp;
            emit Stored(fingerprint, block.timestamp, projectId);
        }
    }

    /// @notice Return true if *fingerprint* has been recorded.
    function exists(bytes32 fingerprint) external view returns (bool) {
        return storedAt[fingerprint] != 0;
    }
}
