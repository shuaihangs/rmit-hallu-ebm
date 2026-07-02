# GPU Server Runbook

When the user says "activate GPU server", use this server and project layout.

## Server

- SSH target: `suns11@131.170.118.63`
- Project path: `/home/suns11/working_dir/ebm_hallu/PHD Projects/EnergyNeighbourProb`
- Virtual environment: `Energy`
- Do not write the SSH password into files or logs.

## Activate Project Environment

```bash
ssh suns11@131.170.118.63
cd "$HOME/working_dir/ebm_hallu/PHD Projects/EnergyNeighbourProb"
source Energy/bin/activate
```

## Start Jupyter Lab

Run Jupyter from `/home/suns11/working_dir/ebm_hallu` so the user can browse all
project folders under `ebm_hallu`.

```bash
tmux kill-session -t ebm_jupyter 2>/dev/null || true
tmux new-session -d -s ebm_jupyter 'bash -lc "cd \"$HOME/working_dir/ebm_hallu/PHD Projects/EnergyNeighbourProb\" && source Energy/bin/activate && cd \"$HOME/working_dir/ebm_hallu\" && python3 -m jupyter lab --no-browser --port=8888"'
tmux capture-pane -pt ebm_jupyter -S -80
```

Then start the local tunnel from the user's machine:

```bash
ssh -fN -L 8888:localhost:8888 suns11@131.170.118.63
```

Use the token printed by Jupyter in the tmux output:

```text
http://127.0.0.1:8888/lab?token=<token>
```

## Training Job

The main experiment runs from the project path:

```bash
cd "$HOME/working_dir/ebm_hallu/PHD Projects/EnergyNeighbourProb"
source Energy/bin/activate
python -u run_model.py
```

Preferred tmux session for training:

```bash
tmux kill-session -t hallu_exp 2>/dev/null || true
tmux new-session -d -s hallu_exp 'bash -lc "cd \"$HOME/working_dir/ebm_hallu/PHD Projects/EnergyNeighbourProb\" && source Energy/bin/activate && python -u run_model.py 2>&1 | tee outputs/hallu_tuning.log; python plotting.py 2>&1 | tee -a outputs/hallu_tuning.log"'
```

Check progress:

```bash
tail -n 120 outputs/hallu_tuning.log
tmux capture-pane -pt hallu_exp -S -120
```
