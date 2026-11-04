# weir-subnet

Miner node agent (`weir-node`) and validator (`weir-validator`) for Weir, Bittensor Subnet 36 (mainnet, netuid 36). Weir sells the GPUs that miners list here at [weircompute.com](https://weircompute.com).

## How verification works

1. A miner registers a hotkey on the subnet and publishes its node address with `weir-node announce` (hotkey-signed `serve_axon`).
2. Each round, the validator reads the metagraph and calls every published node. Requests use Bittensor signed requests (btauth/1), and every answer is signed with the miner's hotkey.
3. GPU check: the validator sends a random seed. The node builds matrices from the seed, fills about 70% of the card's memory, multiplies them in fp16, and returns a commitment. The validator then picks random rows and recomputes them on its CPU. A wrong answer zeroes the miner for 24 hours; a slow answer means the GPU is not verified. All GPUs of a node are challenged at the same time.
4. Weights, in three pools: 20% to rented GPUs (settled GPU-hours x model weight), up to 80% to idle verified GPUs that pass the idle gates, and the rest to the reserve hotkey. See `weir_subnet/validator/scoring.py`.

   **Network fund.** The reserve hotkey is Weir's network fund. It receives only what the rented and idle pools leave, and pays for standby GPUs kept ready so popular cards stay rentable when sold out, renter credits and promotions that bring demand to providers, and infrastructure and development (validators, payment processing, servers, this code). As providers fill the idle pool, its share shrinks toward zero.

## Rules for miners

- **Accepted cards:** H100 80GB, A100 80GB, A100 40GB, A800 80GB, A800 40GB, RTX 5090, RTX 4090 48GB, RTX 4090. Other cards earn nothing (`UNKNOWN_MODEL`).
- **Activation fee:** paid once per GPU, by model. One hotkey can list many machines and GPUs; each GPU needs its own fee.

  | Card | Fee (TAO per GPU) | Weight |
  | --- | --- | --- |
  | H100 80GB | 10.0 | 4.0 |
  | A100 80GB | 7.8 | 2.5 |
  | A800 80GB | 7.6 | 2.4 |
  | A100 40GB | 6.1 | 1.8 |
  | A800 40GB | 5.9 | 1.7 |
  | RTX 5090 | 4.7 | 1.3 |
  | RTX 4090 48GB | 4.4 | 1.2 |
  | RTX 4090 | 3.7 | 1.0 |

  These are the defaults. Weir may change fees and weights per model (for example to attract a card that is short), through the platform file the validator reads; `weir-node status` shows the fees in force. A GPU already activated stays activated when a fee changes. Activation fees are not refunded.

  Pay either way; both add to your activation credit:
  - **TAO:** send it from the hotkey's coldkey to the fee address `5Dk1ftRx4wpwZsiufim8KvHPaeY7k9RdkXJmii2ADv9uHfZ9`. The validator reads the transfers from the chain. Credit belongs to the coldkey and is shared by its hotkeys.
  - **USDT or USDC:** pay the same amount through Weir (support@weircompute.com) and give the hotkey.

  When a GPU passes its first check, credit activates it, most valuable model first. An activated GPU stays activated for good, also if it moves to another hotkey of the same coldkey. A GPU without its fee earns no idle pay and is not sold (`NOT_ACTIVATED`); a hotkey with no activated GPU and no credit earns nothing.
- **Price cap:** each model has a highest price per GPU-hour. Above it a machine earns no idle pay and is not sold (`PRICE_OVER_CAP`). Rentals already settled still count.
- **Waiting period:** a GPU earns idle pay after 14 days of passed checks without a gap (`PROBATION`; `weir-node status` shows the date). A gap, a changed GPU set or a wrong answer starts it again.
- **Idle pay:** model weight x 80% / 1134 per idle GPU and weight set (`--target-capacity`), until the idle pool is full. Disk >= 1.5x VRAM, Sysbox and a reachable node are required.

## Network values

| | |
| --- | --- |
| Network | `finney` (mainnet), netuid `36` |
| Validator hotkey | `5Ec48Jz5mS5kY7198cJFSDHDNPEAvDHnsCNaiB9bXq1Ux2iX` |
| Platform hotkey | `5ERf3azzYiJEdUdgufdM9bnz4UU94igLh362gBSBXWVdJehe` |
| Platform address | `https://weircompute.com` |
| Activation fee address | `5Dk1ftRx4wpwZsiufim8KvHPaeY7k9RdkXJmii2ADv9uHfZ9` |

These are the defaults of `weir-node`, so the commands below leave them out. They are in `weir_subnet/network.py`.

## Become a miner

You need: Ubuntu 22.04 or 24.04 with an NVIDIA driver, one of the accepted cards, Docker, a public address, and a Bittensor wallet with a hotkey registered on netuid 36 (`btcli subnet register --netuid 36`). Keep the coldkey on your own computer; the GPU machine needs the hotkey only.

**1. NVIDIA container toolkit** (lets Docker pass GPUs into containers):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**2. Sysbox** (tenants get root and can run Docker inside, without a privileged container; needed for idle pay). Installing it restarts Docker, so stop your containers first:

```bash
curl -fsSL -o sysbox.deb https://github.com/nestybox/sysbox/releases/download/v0.6.6/sysbox-ce_0.6.6-0.linux_amd64.deb
sudo apt-get install -y ./sysbox.deb
```

**3. weir-node:**

```bash
git clone https://github.com/weircompute/weir-subnet && cd weir-subnet
python3 -m venv .venv && .venv/bin/pip install -e '.[gpu]'
```

PyTorch must match your NVIDIA driver. `pip` installs the newest PyTorch, which needs a recent driver; on an older one the GPU check fails with "The NVIDIA driver on your system is too old". Check the CUDA version your driver supports (top right of `nvidia-smi`) and install the matching build, for example for CUDA 12.4 (driver 550):

```bash
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

The self-test in step 4 runs the same GPU check as the validator, so it shows this before you go live.

**4. Self-test.** Runs the validator's own GPU check on every card, with the same sizes and time limits, and checks Docker, GPUs inside a container, Sysbox and your payout address. Fix every `FAIL` before going on:

```bash
.venv/bin/weir-node selftest --payout-address <your USDT/USDC address>
```

**5. Ports.** The node listens on one port (default 8091) for the validator and the platform, and tenants SSH in on ports from `--ssh-ports` (default 42000-42999). All of them must be reachable from the internet. If your provider maps external ports to internal ones, run the node on the internal port, announce the external port, and give `--ssh-ports` a range that is mapped one to one, or as many single ports as you have.

**6. Run the node** (as a service, so it survives reboots):

```bash
.venv/bin/weir-node run --wallet miner --hotkey default --price 0.35 \
    --public-host <public ip> --payout-address <your USDT/USDC address>
```

`--price` is your ask in USD per GPU-hour; keep it under the price cap for your card.

**7. Publish the address** on chain (signed with the hotkey, once, and again if the address changes):

```bash
.venv/bin/weir-node announce --wallet miner --hotkey default --ip <public ip> --port 8091
```

**8. Pay the activation fee** for your GPUs (see the rules above). Credit activates each GPU once it passes its first check.

**9. Check your status:**

```bash
.venv/bin/weir-node status --wallet miner --hotkey default
```

It shows your activation credit, the fee per card, and a reason code for every machine and GPU with what to do about it.

## Run a validator

```bash
weir-validator --wallet validator --hotkey default \
    --platform-file <platform file> --reserve-hotkey <hotkey>
```

`--platform-file` is written by the platform: the highest price per GPU-hour for each model, any per-model fee or weight that differs from the defaults (`fees`, `weights`), and activation credit paid in USDT or USDC (`activation_credit`, TAO per hotkey).

 on a port from `--ssh-ports`. Release deletes the container, network and volume.

## Selling through weir-api

The validator writes `sellable.json` (`--sellable-file`): miners in good standing and their GPUs that passed a challenge in the last 6 hours. weir-api reads it (`SUBNET_NODES_FILE`) and sells only those GPUs, asking each node live for free capacity. weir-api signs node calls with its platform key (`SUBNET_KEY_FILE`, made by `cargo run --example platform_key`); nodes accept orders only from that address (`--platform`). `scripts/dev_node.py` runs a node plus one validator round for weir-api's `tests/subnet_nodes.rs`.

## Several machines

Run `weir-node` on each machine with the same hotkey and its own `--name`. On the machine at your chain address, list them all with `--fleet main=http://1.2.3.4:8091,box2=http://1.2.3.5:8091`. Every machine signs with your hotkey, so a listed machine that is not yours is refused. One machine can serve several tenants at once: each order gets its own GPUs.

## Your status and earnings

    weir-node status   --wallet miner
    weir-node summary  --wallet miner
    weir-node earnings ... [--csv]
    weir-node payouts  ...

Requests are signed with your hotkey; you only ever see your own data. `status` shows a reason code for every machine and GPU (`OK`, `UNREACHABLE`, `CHALLENGE_SLOW`, `DUPLICATE_GPU`, ...) with what to do about it. `--metrics-port 9100` serves Prometheus metrics on 127.0.0.1 for your own Grafana.

## Why the check is open source

The challenge does not depend on its code being secret: each challenge has a fresh random seed and size, all GPUs of a machine are checked at once against a time limit, and the rows the validator checks are chosen only after the node has committed to all of its results. Knowing the code does not help a node that skipped work. What a time-limited check cannot prove is the amount of memory on a card; a card sold with less memory than claimed fails its tenants, which costs the miner its earnings and a 48-hour penalty per order.

## Getting paid

Tenants pay your ask (`--price`, USD per GPU-hour). After each rental ends you earn what the tenant paid minus refunds, less the 5% platform fee, in stablecoin to `--payout-address`. Earnings become payable 48 hours after release and are settled weekly once they reach $5. A rental that fails because of your machine earns nothing and costs 48 hours of your share. Billed GPU-hours and delivery success decide your share of the 20% rented pool (`--rentals-file` on the validator, written by weir-api's `weir-payouts rentals`).

## Local chain

For development, `scripts/localnet_e2e.py` runs the whole lifecycle on a local chain with throwaway keys: register and activate a subnet, register a miner, publish its address, one validator round, weights on chain, incentive after the epoch.

## Known limits

- Timing limits are measured on one model so far (A800 40GB: 6.6-7.1 s against a 9.4 s limit). Other models are scaled by memory bandwidth and get 1.5x extra slack until measured with `selftest --json`.
- Tested on A800 and V100 machines (Ubuntu 22.04): GPUs inside containers with nvidia-container-toolkit, and inside Sysbox 0.6.6 containers.
- Rental stats (R, D) come from a platform file and are not signed yet.
- Fee history before the validator's first round needs an archive endpoint (`--fee-start-block`).

## Tests

```bash
pytest
WEIR_DOCKER_TESTS=1 pytest tests/test_docker_runtime.py   # needs Docker and the weir-test-sshd image
```
