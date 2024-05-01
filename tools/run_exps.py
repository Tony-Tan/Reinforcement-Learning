import subprocess
import time

games = [
    "ALE/Pitfall-v5",
    "ALE/Seaquest-v5",
    "ALE/Enduro-v5",
    "ALE/Breakout-v5",
    "ALE/Frostbite-v5",
    "ALE/Gravitar-v5",
    "ALE/Phoenix-v5",
    "ALE/Qbert-v5",
    "ALE/Centipede-v5"
]

# Base command to run the training script
base_command = ("/root/miniconda3/bin/conda run -n Reinforcement-Learning --no-capture-output python "
                "/root/rl/algorithms/dqn.py --env_name")

python_path = ("export PYTHONPATH=$PYTHONPATH:/root/rl")
# Activate your virtual environment command

# Loop through each game, creating a screen session for each one
for game in games:
    # Creating a unique screen name for each game session
    screen_name = f"training_{game.split('/')[1]}"

    # Combine commands to activate env and start training in one command string
    combined_command = f"{python_path}&&{base_command} '{game}'"

    # Full command to run in screen
    screen_command = f"screen -dmS {screen_name} bash -c '{combined_command}; exec bash'"

    # Execute the command to start the screen session with the training running
    subprocess.run(screen_command, shell=True)

print("Training sessions started in separate screen sessions.")