"""Weir on Bittensor mainnet (finney), Subnet 36. These are the defaults of
weir-node and weir-validator; every one can be overridden on the command line."""

NETWORK = "finney"
NETUID = 36
# Hotkeys allowed to challenge nodes.
VALIDATORS = "5Ec48Jz5mS5kY7198cJFSDHDNPEAvDHnsCNaiB9bXq1Ux2iX"
# The Weir platform: places and releases orders, serves miner status.
PLATFORM = "5ERf3azzYiJEdUdgufdM9bnz4UU94igLh362gBSBXWVdJehe"
PLATFORM_URL = "https://weircompute.com"
# Coldkey that receives activation fees.
FEE_ADDRESS = "5Dk1ftRx4wpwZsiufim8KvHPaeY7k9RdkXJmii2ADv9uHfZ9"
# Tenant images: Weir image id = Docker image.
IMAGES = ("ubuntu-2204-cuda124=nvidia/cuda:12.4.1-runtime-ubuntu22.04,"
          "ubuntu-2404-cuda130=nvidia/cuda:13.0.0-runtime-ubuntu24.04")
