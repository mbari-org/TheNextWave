# Production setup: SWIFT SBG over Digi Ethernet Bridge

This is a field checklist for getting raw SBG Ellipse data from multiple SWIFT buoys onto a PC over a Digi 900 MHz Ethernet bridge, then feeding those streams into `the_next_wave` through the built-in SBG TCP bridge server.

No credentials are included here (passwords, encryption keys, license keys, or device-specific IDs). Keep those in an operations runbook or password manager, not in git.

## System overview

- **On each SWIFT**: Sutron Xpert (logger) + SBG Ellipse block emits SBG ECom binary frames over TCP/IP.
- **Radio link**: Digi Ethernet bridge pair (Access Point at the ship/shore PC, Subscriber Unit on each SWIFT).
- **On the PC**:
  - Option A: `the_next_wave` node binds TCP servers (ports per buoy) and ingests frames directly.
  - Option B (debugging): AutoPoll “Capture” task listens on ports and logs raw binary to disk.

Key point: the SWIFT/Xpert is configured to **connect to the PC’s IP + port** and stream data. The PC side must be listening.

## Network/IP plan (example)

Pick a consistent scheme and stick to it; the numbers below mirror what worked in field notes.

- PC NIC:
  - IP: `192.168.1.99`
  - Netmask: `255.255.255.0`
  - Gateway: `192.168.1.17` (often the AP)

- Digi Access Point (ship/shore):
  - IP: `192.168.1.17`

- Digi Subscriber Units (on SWIFTs):
  - Example IPs: `192.168.1.18`, `192.168.1.19`, `192.168.1.20`, …

- Sutron Xpert LAN IPs (one per SWIFT):
  - Example IPs: `192.168.1.28`, `.29`, `.30`, `.31`
  - Xpert default gateway: set to the **Subscriber Unit IP** that the Xpert is physically connected to.

### Digi radio parameters (do not commit)

You will need:
- Network name / PAN ID (or equivalent)
- Encryption key
- Subscriber count expected by AP
- Unique subscriber ID per SWIFT

Store these as:
- `<DIGI_NETWORK_NAME>`
- `<DIGI_ENCRYPTION_KEY>`
- `<SUBSCRIBER_UNIT_ID>`

## Port mapping (important)

The deployed system uses a **non-linear** per-SWIFT port mapping (from field notes). Configure each Xpert/SBG Ellipse block to connect to the PC IP using:

- SWIFT24 → **3001**
- SWIFT25 → **3002**
- SWIFT22 → **3003**
- SWIFT23 → **3004**

On the PC, `the_next_wave` will listen on those ports when you set the ROS parameter block:

- `swifts.swift22: 3003`
- `swifts.swift23: 3004`
- `swifts.swift24: 3001`
- `swifts.swift25: 3002`

This reuses the existing `swifts.*` config block:
- In Gazebo/latent runs, `swifts.*` are indices into `inc_wave_heights[]`.
- In deployment / SBG TCP bridge runs, `swifts.*` are TCP ports.

## Configure the PC

1. Set the PC NIC to the static IP (example above).
2. Ensure firewall allows inbound TCP on the chosen port range (e.g., 3001–3004).
3. Confirm you can reach Digi/Xpert devices:
   - `ping 192.168.1.17` (AP)
   - `ping 192.168.1.28` (an Xpert)

## Configure Digi Ethernet bridge

Using Digi’s discovery/config utility:

1. Set one unit as **Access Point** (ship/shore side).
2. Set each SWIFT unit as **Subscriber Unit**.
3. Ensure:
   - Same network name and encryption key across AP/SUs
   - Each SU has a unique subscriber ID
   - AP is configured to expect the correct number of subscribers
   - Channel set to Auto (or a fixed channel if you prefer, but then set it consistently)

## Configure Sutron Xpert LAN + Ellipse block

On each SWIFT’s Xpert:

1. Configure LAN settings (no wizard) with:
   - Unique Xpert IP (e.g., `192.168.1.28`)
   - Netmask `255.255.255.0`
   - Default gateway = that SWIFT’s Digi Subscriber Unit IP

2. Add/configure the **SBG Ellipse** block:
   - “IP” / destination host: the PC IP (e.g., `192.168.1.99`)
   - “IP Port”: the port assigned for that SWIFT (see Port mapping above)
   - Com port: as required by your hardware wiring (field notes referenced Com 6)

3. Verify the stream reaches the PC:
   - If using AutoPoll, you should see a growing capture file.
   - If using `the_next_wave`, you should see the node log “listening” and then “connection from …”.

## Running `the_next_wave` in production

You typically want **bridge enabled** and **replay disabled**.

- Launch:
  - `enable_sbg_bridge:=true`
  - `enable_sbg_tcp_replay:=false`

The predictor node will:
- listen on the configured TCP ports,
- fill rolling SBG windows per SWIFT,
- and run the prediction processing on its timer schedule.

### WEC target position

In production you expect the WEC target lat/lon to come from the AHRS topic (via `ahrs_callback`).

If you are running without a WEC pose source (lab/replay), you can optionally enable:
- `sbg_bridge_use_example_frame: true`

in the ROS params YAML to force the same origin/rotation/target used by `example.py`.

## Debugging checklist

- “Address already in use” on ports:
  - Another process is still bound; stop it or change the per-SWIFT port in `swifts.swiftNN`.

- No “connection from …”:
  - Wrong PC IP configured in Xpert, wrong port mapping, firewall blocking, or Digi link down.

- “window ready?” stays False:
  - You are not receiving all required message types (ShipMotion + GpsVel + GpsPos) at a healthy rate.

- Data looks alive but predictions unstable:
  - Check time alignment (timestamp jitter, missing samples) and confirm the per-buoy stream rates.

## References

- Ocean Engineering paper (method): https://doi.org/10.1016/j.oceaneng.2021.108871
- Associated code archive: http://hdl.handle.net/1773/46928
