import subprocess

# Full path to your Python executable
python_path = "/usr/local/miniconda3/bin/python3"  # Update this to your actual Python path

# List of environment names
env_names = ["ALE/VideoPinball-v5", "ALE/Boxing-v5", "ALE/Breakout-v5", "ALE/StarGunner-v5", "ALE/Robotank-v5"]
# env_names = [ "ALE/StarGunner-v5", "ALE/Robotank-v5"]
# List of environment names
# env_names = ["ALE/Atlantis-v5", "ALE/CrazyClimber-v5", "ALE/Gopher-v5", "ALE/DemonAttack-v5", "ALE/Krull-v5"]
# env_names = [ "ALE/DemonAttack-v5", "ALE/Krull-v5"]
# List of environment names
# env_names = ["ALE/Pong-v5", "ALE/RoadRunner-v5", "ALE/Kangaroo-v5", "ALE/Tennis-v5", "ALE/Assault-v5"]
# env_names = ["ALE/Tennis-v5", "ALE/Assault-v5"]
# Loop through the environment names and run the script
for env_name in env_names:
    for i in range(5):
        print(f"Running environment {env_name} iteration {i+1}")
        subprocess.run([python_path, "../algorithms/dqn.py", "--env_name", env_name])