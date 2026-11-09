# Weir · Bittensor Subnet 36

Weir is a GPU rental marketplace on Bittensor. Miners bring GPU machines, validators check that the cards are real and fast, and renters on [weircompute.com](https://weircompute.com) rent them by the hour.

This repo has both sides:

| Program | Who runs it | What it does |
| --- | --- | --- |
| `weir-node` | Miners | Answers GPU checks and starts rental containers on your machine |
| `weir-validator` | Validators | Checks every listed GPU and sets weights |

**At a glance**

| | |
| --- | --- |
| Network | `finney`, netuid `36` |
| Validator hotkey | `5Ec48Jz5mS5kY7198cJFSDHDNPEAvDHnsCNaiB9bXq1Ux2iX` |
| Platform hotkey | `5ERf3azzYiJEdUdgufdM9bnz4UU94igLh362gBSBXWVdJehe` |
| Cards that earn | H100 80GB · A100 80/40GB · A800 80/40GB · RTX 5090 · RTX 4090 48GB · RTX 4090 |
| Rent is paid in | USDT or USDC, weekly |
| Questions | [Telegram](https://t.me/+qmO92A3EUt5hYWMx) · support@weircompute.com |

`weir-node` already knows these values (`weir_subnet/network.py`), so you don't type them.

---

## Quick start for miners

Ubuntu 22.04 or 24.04 with an NVIDIA driver and Docker. Register a hotkey first (`btcli subnet register --netuid 36`). Only the hotkey goes on the GPU machine; leave your coldkey on your own computer.

```bash
# 1. GPUs inside Docker: NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# 2. Sysbox, so renters get root and Docker inside their container (restarts Docker)
curl -fsSL -o sysbox.deb https://github.com/nestybox/sysbox/releases/download/v0.6.6/sysbox-ce_0.6.6-0.linux_amd64.deb
sudo apt-get install -y ./sysbox.deb

# 3. weir-node
git clone https://github.com/weircompute/weir-subnet && cd weir-subnet
python3 -m venv .venv && .venv/bin/pip install -e '.[gpu]'

# 4. Self-test: the same GPU check the validator runs
.venv/bin/weir-node selftest --payout-address <USDT/USDC address>

# 5. Run, then publish your address on chain
.venv/bin/weir-node run --wallet miner --hotkey default --price 0.35 \
    --public-host <public ip> --payout-address <USDT/USDC address>
.venv/bin/weir-node announce --wallet miner --hotkey default --ip <public ip> --port 8091

# 6. See how your machines are doing
.venv/bin/weir-node status --wallet miner --hotkey default
```

Run step 5 as a systemd service so it comes back after a reboot.

> **Self-test says "The NVIDIA driver on your system is too old"?** pip installed a PyTorch built for a newer CUDA than your driver. Look at the CUDA version in the top right of `nvidia-smi` and install the matching build. For CUDA 12.4 (driver 550):
>
> ```bash
> .venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```

### Ports

Open the node port (default `8091`) and the renter SSH range (`--ssh-ports`, default `42000-42999`) to the internet.

Behind a provider that maps ports (external 30282 → internal 8800, for example), run the node on the internal port and announce the external one. For `--ssh-ports`, use ports that map one to one.

### More than one machine

Use the same hotkey everywhere and give each machine its own `--name`. On the machine whose address is on chain, add `--fleet main=http://1.2.3.4:8091,box2=http://1.2.3.5:8091`. A machine in the list that doesn't sign with your hotkey is ignored.

---

## Activation fee

Each GPU pays a one-time activation fee, set per card. Fees change with demand, so they are not listed here. Get the current ones, with the price range each card may be listed at:

```bash
.venv/bin/weir-node quote --wallet miner --hotkey default
```

A quote is held for that hotkey for 7 days: GPUs it activates in that time pay at most the quoted fee. No hotkey yet? Sign in at [weircompute.com](https://weircompute.com), create an API key under **API keys**, and ask with it:

```bash
curl -H "Authorization: Bearer <API key>" "https://weircompute.com/v1/miner/quote?hotkey=<your hotkey>"
```

Pay from the hotkey's coldkey to `5Dk1ftRx4wpwZsiufim8KvHPaeY7k9RdkXJmii2ADv9uHfZ9`. Payments add up as credit, and each GPU is activated once it passes its first check. A GPU that later reports a dearer card owes the difference within 7 days. Activation fees are not refunded.

---

## How you earn

Every weight-setting round, the miner share of emissions is split three ways:

| Pool | Share | Who gets it |
| --- | --- | --- |
| Rented | 20% | GPUs rented on Weir, by settled GPU-hours × card weight |
| Idle | up to 80% | Verified GPUs waiting for a renter: card weight × 80% ÷ 1134 each, until the pool is full |
| Network fund | what is left | The reserve hotkey (below) |

On top of emissions, **rent goes to you**: what the renter paid, minus refunds and Weir's 5% fee. It becomes payable 48 hours after the rental ends and is sent weekly in USDT or USDC once it reaches $5.

**Card weights:** H100 4.0 · A100 80GB 2.5 · A800 80GB 2.4 · A100 40GB 1.8 · A800 40GB 1.7 · RTX 5090 1.3 · RTX 4090 48GB 1.2 · RTX 4090 1.0

**What an idle GPU needs before it earns:**

- 14 days of passed checks in a row. A missed check, a changed GPU set or a wrong answer starts the count again.
- Your price between half the cap and the cap for that card.
- Disk at least 1.5× the machine's total VRAM.
- Sysbox as the container runtime.
- A node that answers at least half the time.

Weir may review GPUs periodically. A reviewed GPU can earn before its probation ends, or while a check result is disputed; disputed GPUs are not listed for rent.

**Network fund.** The reserve hotkey only receives what the rented and idle pools don't pay out. It pays for standby GPUs that keep popular cards available, renter credits and promotions that bring rentals to miners, and running the network (validators, payments, servers, this code). The more GPUs miners list, the smaller it gets.

---

## How a GPU is checked

1. The validator reads the metagraph and calls every node that has an address on chain. Requests and answers are signed with Bittensor hotkeys.
2. It sends a random seed. The node builds matrices from it, fills about 70% of each card's memory, multiplies them in fp16 and returns a commitment.
3. The validator picks random rows and recomputes them on its CPU. All cards on a machine are checked at the same time, against a time limit.
4. A wrong answer zeroes the miner for 24 hours. A slow answer means that card isn't verified this round.

The code is public on purpose. Each check has a new seed and size, and the rows to verify are picked only after the node has committed to its results, so reading the code doesn't help a node that skips the work. A check can't measure memory size exactly; a card sold with less memory than claimed fails its renters, which costs the miner that rent and a 48-hour penalty per order.

---

## Status codes

`weir-node status` gives every machine and GPU a code:

| Code | Meaning | What to do |
| --- | --- | --- |
| `OK` | Verified and earning | Nothing |
| `PROBATION` | Passing, within the first 14 days | Wait; the date is shown |
| `PENDING` | Not checked yet, or its last pass expired | Wait for the next round |
| `UNREACHABLE` | The validator couldn't reach the node | Check the port and the announced address |
| `CHALLENGE_SLOW` | Answer came in over the time limit | Check GPU load, clocks and PCIe |
| `UNKNOWN_MODEL` | Card isn't on the list | Only listed cards earn |
| `PRICE_OVER_CAP` | Your price is above the cap | Lower `--price` |
| `PRICE_UNDER_FLOOR` | Your price is below half the cap | Raise `--price` |
| `DISK_TOO_SMALL` | Disk under 1.5× VRAM | Add disk |
| `NO_SYSBOX` | Containers don't run under Sysbox | Install Sysbox, keep `--docker-runtime sysbox-runc` |
| `DUPLICATE_GPU` | The same GPU shows up under two hotkeys | Run each card under one hotkey only |
| `RENTED` | A renter is using it | Nothing; it counts toward the rented pool |

`weir-node summary`, `earnings` (add `--csv` for a spreadsheet) and `payouts` show your rent. You only see your own hotkey's data. `--metrics-port 9100` serves Prometheus metrics on 127.0.0.1.

---

## Running a validator

```bash
weir-validator --wallet validator --hotkey default \
    --platform-file <platform file> --rentals-file <rentals file> --reserve-hotkey <hotkey>
```

The platform file holds the price cap for each card. The rentals file holds settled GPU-hours and delivery rates per hotkey. Both are written by the Weir platform. The validator writes the list of verified GPUs that Weir is allowed to sell.

---

## FAQ

**Do I need a GPU on the list?** Yes. Other cards answer checks but don't earn.

**Can renters break out of the container?** Each rental gets its own container, network and `/data` volume, only the GPUs it ordered, and key-only SSH. Sysbox gives renters root inside the container without giving them the host. Releasing a rental deletes all of it.

**What if my machine fails a renter?** That rental earns nothing and costs 48 hours of your rented-pool share.

**Where do I ask?** [Telegram](https://t.me/+qmO92A3EUt5hYWMx) or support@weircompute.com.

---

## Development

```bash
pip install -e '.[gpu]' && pytest
WEIR_DOCKER_TESTS=1 pytest tests/test_docker_runtime.py   # needs Docker
```

`scripts/localnet_e2e.py` runs the full cycle on a local chain with throwaway keys.

Known limits: the time limits are measured on A800 40GB so far (6.6–7.1 s against 9.4 s). Other cards are scaled by memory bandwidth, with 1.5× extra slack until measured with `selftest --json`. Rental stats are not signed yet.
