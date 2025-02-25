#!/bin/bash

# Set the Anaconda environment
ANACONDA_ENV="tandic-training"

# Name of the tmux session
TMUX_SESSION="training_session"

# Start a new tmux session and run the commands
tmux new-session -d -s $TMUX_SESSION

# Activate the Anaconda environment and run the script inside the tmux session
tmux send-keys -t $TMUX_SESSION "source activate $ANACONDA_ENV" C-m
tmux send-keys -t $TMUX_SESSION "echo 'Running script with Anaconda environment: $ANACONDA_ENV'" C-m

# Run the Python script with the current date
date=$(date '+%Y-%m-%d %H:%M:%S')
tmux send-keys -t $TMUX_SESSION "python training/train.py --d '${date}'" C-m

# After the script is done, kill the tmux session
tmux send-keys -t $TMUX_SESSION "tmux kill-session -t $TMUX_SESSION" C-m
