require("dotenv").config();
const { ethers } = require("hardhat");

async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("Deploying Roles Modifier with account:", deployer.address);

    // Replace with your actual Gnosis Safe address
    const SAFE_ADDRESS = "0x9859B8A9A7Ff0f7abCe45b03Fbd6Dc8f78AD94e0";

    // Deploy Roles Modifier contract
    const Roles = await ethers.getContractFactory(
  "contracts/modules/roles/Roles.sol:Roles"
);
    const roles = await Roles.deploy(SAFE_ADDRESS, SAFE_ADDRESS);
    
    await roles.waitForDeployment();

    console.log("Roles Modifier deployed to:", await roles.getAddress());
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    })
