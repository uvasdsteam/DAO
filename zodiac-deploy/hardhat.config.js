/** @type import('hardhat/config').HardhatUserConfig */
require("dotenv").config();
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
    solidity: "0.8.19",  // Ensure this matches Zodiac's Solidity version
    networks: {
        sepolia: {
            url: process.env.SEPOLIA_RPC_URL,  // Your Sepolia RPC URL
            accounts: [process.env.PRIVATE_KEY],  // Your wallet's private key
        },
    },
    etherscan: {
        apiKey: process.env.ETHERSCAN_API_KEY,  // Optional: For contract verification
    },
};
