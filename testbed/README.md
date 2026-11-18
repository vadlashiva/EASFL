# EASFL on FIT IoT-LAB: measuring the real radio energy (reviewer comments AE-4, R1-4, R2-2)

The reviewers asked whether the first-order radio model hides MAC contention, retransmissions,
idle listening and control traffic. This kit replays one federated round of each scheme
(EASFL, FedAvg-LEACH, centralized collection) on real IEEE 802.15.4 nodes (FIT IoT-LAB M3,
AT86RF231 radio, CSMA/CA with ACKs and retries) while the testbed measures every node's power.
The result calibrates the simulator, which then recomputes lifetimes and checks the ranking.

Files: `easfl_traffic.elf` (prebuilt firmware), `firmware/` (its source, RIOT 2024.10),
`easfl_testbed.py` (runs the rounds), `analyze_testbed.py` (energy per role and round,
calibration constants), `../simulator/calibrate_sim.py` (lifetimes with the measured radio).

## Steps (run in Windows PowerShell; replace LOGIN with your IoT-LAB login)

1. Copy the kit to the IoT-LAB frontend:

       scp easfl_traffic.elf easfl_testbed.py analyze_testbed.py LOGIN@grenoble.iot-lab.info:~/

2. Log in and authenticate the command-line tools once (asks for your IoT-LAB password, typed by you):

       ssh LOGIN@grenoble.iot-lab.info
       iotlab-auth -u LOGIN

3. Create the power-monitoring profile (once):

       iotlab-profile addm3 -n easfl_pwr -p dc -current -voltage -power -period 1100 -avg 4

4. Smoke test with 4 nodes for 10 minutes:

       iotlab-experiment submit -n easfl_smoke -d 10 -l 4,archi=m3:at86rf231+site=grenoble,easfl_traffic.elf,easfl_pwr
       iotlab-experiment wait
       python3 easfl_testbed.py --rounds 3 --smoke

   It should print the node names, their addresses and "done". Send the printed output to Claude.

5. Full run with 20 nodes (about 30 minutes; the experiment is booked for 45):

       iotlab-experiment submit -n easfl_full -d 45 -l 20,archi=m3:at86rf231+site=grenoble,easfl_traffic.elf,easfl_pwr
       iotlab-experiment wait
       python3 easfl_testbed.py --rounds 100

6. Analyse (the experiment id is printed by `submit`; `iotlab-experiment get -i` also shows it):

       python3 analyze_testbed.py EXP_ID

   Copy `testbed_results.json`, `radio_measured.json`, `roles.json`, `phases.csv` and the
   `serial_*.log` files back to the PC:

       exit
       scp LOGIN@grenoble.iot-lab.info:~/{testbed_results.json,radio_measured.json,roles.json,phases.csv} .

   and send them to Claude (or put them in E:\EASFL HEAP\testbed).

7. Claude runs `python simulator/calibrate_sim.py radio_measured.json` and writes the result into the
   paper (energy per role and round, measured ratio FedAvg/EASFL, lifetimes and ranking with the
   measured radio).

## What is measured

* Idle power with every radio listening and no traffic.
* For each scheme and round: control (CH advertisement, joins), uplink (EASFL: 50% of members send
  94 B; FedAvg: all members send 1,668 B; centralized: all members send 480 B), CH to BS, and the
  model broadcast (EASFL 834 B, FedAvg 1,668 B). Payloads go out as frames of up to 100 bytes, so
  the real PHY/MAC headers, ACKs, CSMA/CA backoff and retransmissions are included, and members
  send at the same time, so contention is real.
* Energy above idle per role, the time each node's radio has to be on, and the energy per round
  if the radio is switched on only while the node sends or receives (duty cycling).

Rebuilding the firmware (optional): clone RIOT 2024.10 next to `firmware/` and run
`make BOARD=iotlab-m3` in `firmware/` with arm-none-eabi-gcc on the PATH.
