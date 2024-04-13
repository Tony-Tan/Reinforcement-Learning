import subprocess

# Full path to your Python executable
python_path = "/usr/local/miniconda3/bin/python3"  # Update this to your actual Python path

# List of environment names
# env_names = ["ALE/Pong-v5", "ALE/SpaceInvaders-v5", "ALE/Breakout-v5", "ALE/Asteroids-v5", "ALE/Centipede-v5"]
# List of environment names
# env_names = ["ALE/Skiing-v5", "ALE/Solaris-v5", "ALE/Tennis-v5", "ALE/Tutankham-v5", "ALE/Venture-v5"]
# List of environment names
env_names = ["ALE/Krull-v5", "ALE/Enduro-v5", "ALE/Defender-v5", "ALE/Centipede-v5", "ALE/Kangaroo-v5"]
# Loop through the environment names and run the script
for env_name in env_names:
    for i in range(5):
        print(f"Running environment {env_name} iteration {i+1}")
        subprocess.run([python_path, "../algorithms/dqn.py", "--env_name", env_name])