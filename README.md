# Weir · Bittensor Subnet 36

Weir is a GPU rental marketplace on Bittensor. Miners bring GPU machines, validators check that the cards are real and fast, and renters on [weircompute.com](https://weircompute.com) rent them by the hour.

This repository contains the Weir miner agent.

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
git clone https://github.com/weircompute/weir-subnet.git
cd weir-subnet
python3 -m venv .venv && .venv/bin/pip install -e '.[gpu]'

# 4. Local hardware diagnostics
.venv/bin/weir-node selftest --payout-address <USDT/USDC address>

# 5. Run and register your control endpoint with Weir support
.venv/bin/weir-node run --wallet miner --hotkey default --price 0.35 \
    --host <control interface IP> --public-host <rental SSH IP> --payout-address <USDT/USDC address>

# 6. See your rental earnings
.venv/bin/weir-node summary --wallet miner --hotkey default
```

Run step 5 as a systemd service so it comes back after a reboot.

> **Self-test says "The NVIDIA driver on your system is too old"?** pip installed a PyTorch built for a newer CUDA than your driver. Look at the CUDA version in the top right of `nvidia-smi` and install the matching build. For CUDA 12.4 (driver 550):
>
> ```bash
> .venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```

### Connection setup

Provide your hotkey, machine name and control endpoint to Weir support. Every external miner must run the node agent and pass GPU checks. Control endpoints are registered privately, not announced on chain.

Bind `--host` to the intended control interface and restrict its port to Weir's authorized validator and platform sources. Renters connect to their assigned SSH ports (`--ssh-ports`, default `42000-42999`).

For multiple machines, use the same hotkey, assign each a unique `--name`, and register each endpoint with Weir.

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
| Rented | 10% | GPUs rented on Weir, by GPU-hours actually used × card weight. Rentals under 10 minutes don't count, and a miner gets at most the platform fee its rentals paid |
| Idle | up to 90% | Verified GPUs waiting for a renter: card weight × 90% ÷ 1134 each, until the pool is full |
| Network fund | what is left | The reserve hotkey (below) |

On top of emissions, **rent goes to you**: what the renter paid, minus refunds and Weir's 5% fee. It becomes payable 48 hours after the rental ends and is sent weekly in USDT or USDC once it reaches $5.

**Card weights:** H100 4.0 · A100 80GB 2.5 · A800 80GB 2.4 · A100 40GB 1.8 · A800 40GB 1.7 · RTX 5090 1.3 · RTX 4090 48GB 1.2 · RTX 4090 1.0

**Requirements for external miners to earn idle rewards:**

- 14 days of passed checks in a row. A missed check, a changed GPU set or a wrong answer starts the count again.
- Your price between half the cap and the cap for that card.
- Disk at least 1.5× the machine's total VRAM.
- Sysbox as the container runtime.
- A node that answers at least half the time.

External miners must operate the node agent and pass GPU checks.

**Network fund.** The reserve hotkey only receives what the rented and idle pools don't pay out. It pays for standby GPUs that keep popular cards available, renter credits and promotions that bring rentals to miners, and running the network (validators, payments, servers, this code). The more GPUs miners list, the smaller it gets.

---

## FAQ

**Do I need a GPU on the list?** Yes. Other cards answer checks but don't earn.

**Can renters break out of the container?** Each rental gets its own container, network and `/data` volume, only the GPUs it ordered, and key-only SSH. Sysbox gives renters root inside the container without giving them the host. Releasing a rental deletes all of it.

**What if my machine fails a renter?** That rental earns nothing and costs 48 hours of your rented-pool share.

**Where do I ask?** [Telegram](https://t.me/+qmO92A3EUt5hYWMx) or support@weircompute.com.
