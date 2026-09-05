# Getting started guide

This guide assumes that you're on linux with the Isaac simulator already installed.

These instructions were tested with:
- nvidia-open driver 580.178.04
- dual RTX 5060 Ti
- isaac sim 5.0.0-rc45 installed at /isaac-sim
- ubuntu 24.04.4

## Viam setup

1. Go to app.viam.com and follow the account creation flow, or sign in if you already have an account
1. Go to your [fleet page](https://app.viam.com/fleet/) and use the 'Add machine' button on the right to create a machine we'll use to host the simulation
1. If you're on your personal dev computer, you probably don't want to install the viam daemon. Instead, pick a folder on your computer to download the viam-server binary:
	- `~/viam-isaac` is a good default
	- inside the new folder, download viam-server with `wget https://storage.googleapis.com/packages.viam.com/apps/viam-server/viam-server-stable-$(uname -m) && mv viam-server-stable-$(uname -m) viam-server && chmod +x viam-server && ./viam-server -version`
1. Grab your credentials: back in the web UI, go to the status dropdown on the top menu bar. It's likely in the blue 'awaiting setup' state. Open it, hit the 'Machine cloud credentials' button, then paste the credentials into a `viam.json` file in your `~/viam-isaac` folder.
1. Boot viam: in your terminal, run `ISAAC_SIM_PATH=/isaac-sim ./viam-server -config viam.json`. As it comes up, in the web UI, you should see the status dropdown turn to a green 'Online' state.

## Start Isaac in Viam

In the 'configure' tab of the web UI, hit the '+' button or tap 'A', then tap 'B' for blocks, then find the `isaac-sim-pick-and-place` fragment from the `viam-dev` org (it pulls in the private `viam:isaac-sim-devin` registry module — the machine must be in `viam-dev` to see it) and install it. Click 'Save' in the top right.

The fragment card has a 'Variables' section with nine entries: `table-height-m`, `pick-block-color`, `distractor-color-green`, `distractor-color-blue`, `distractor-color-yellow`, `distractor-color-purple`, `distractor-color-orange`, `detect-color`, and `hue-tolerance-pct`. Each one has a default value. Leave them all unset for your first run. The defaults boot the exact shipped cell: a table, a red block to pick, five distractor blocks (green, blue, yellow, purple, orange), and a place pad.

Switch to the 'logs' tab to watch the installed components start up. On the test machine this takes around 15 seconds. You'll see an 'event=complete' event from the rdk.activity logger when this is done.

Now switch to the 'control' tab to interact with the cameras and arm. The cell has three cameras: `wrist-cam` (rides the arm's flange), `scene-cam` (a fixed overview), and `side-cam` (a fixed camera across the table, used to measure the tallest block). Open the `scene-cam` livestream. You should see a table with six colored blocks on it (red, green, blue, yellow, purple, orange) and a UR5e arm with a gripper mounted at one corner of the table. If the table, blocks, or arm are missing, something failed during boot. Check the logs tab before continuing.

If something goes wrong, the place to debug is the logs tab; to cut down on noise, find the components list on the left-side menu bar and click the `viam_isaac-sim-devin` module (or whatever you named the local module) to filter down the output.

## Run the pick-and-place client

Once the machine is online and the cell looks right in the livestream, run the client script from your dev machine to pick the red block and place it on the pad.

Clone this repository on your dev machine, then create the venv it uses:

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

First, find your connection details. In the web UI, go to the 'connect' tab and select 'Python'. Or go to the 'code sample' tab, which shows a runnable connection snippet with your machine's address and API key filled in. Copy the address, the API key, and the API key ID from there.

Then run:

```
.venv/bin/python examples/pick_red_block.py --address <machine-address> --api-key <key> --api-key-id <key-id> --support-z-mm 750 --randomize-seed <n> --randomize-size-mm 40,90
```

`--support-z-mm 750` tells the script the block rests 750 mm up, on top of the table, instead of on the floor. The table itself is already a motion obstacle: the sim world serves every prop's geometry to the planner live, so no extra flag is needed.

`--randomize-seed <n>` (any integer) scatters the six blocks into a new layout before the pick. `--randomize-size-mm 40,90` also redraws each block's size in that 40-90 mm range; in the `scene-cam` livestream, the blocks visibly change size between runs. Leave both off to use the fragment's fixed starting layout and sizes.

On success, the script prints several marker lines: `MEASURED_BLOCK_JSON=` with the target block's own measured footprint and height, `MEASURED_TALLEST_JSON=` with the tallest scattered object's measured height (`"source": "side"` when the fixed side camera made the measurement), and finally `PLACED_BLOCK_JSON=` with `"placed_on_pad": true`. That last one means the arm found the red block, picked it up, cleared every other block on the way, and set it down on the place pad.

If the randomized target measures over 75 mm, the script refuses the grasp cleanly and leaves the arm parked instead of attempting a doomed pick - that's correct behavior for an oversize block, not a failure.
