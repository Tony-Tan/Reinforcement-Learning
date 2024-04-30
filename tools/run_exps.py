import subprocess
import time

atari_game_batches = [
    ["ALE/Adventure-v5", "ALE/AirRaid-v5", "ALE/Alien-v5"],
    ["ALE/Amidar-v5", "ALE/Assault-v5", "ALE/Asterix-v5"],
    ["ALE/Asteroids-v5", "ALE/Atlantis-v5", "ALE/BankHeist-v5"],
    ["ALE/BattleZone-v5", "ALE/BeamRider-v5", "ALE/Berzerk-v5"],
    ["ALE/Bowling-v5", "ALE/Boxing-v5", "ALE/Breakout-v5"],
    ["ALE/Carnival-v5", "ALE/Centipede-v5", "ALE/ChopperCommand-v5"],
    ["ALE/CrazyClimber-v5", "ALE/Defender-v5", "ALE/DemonAttack-v5"],
    ["ALE/DoubleDunk-v5", "ALE/ElevatorAction-v5", "ALE/Enduro-v5"],
    ["ALE/FishingDerby-v5", "ALE/Freeway-v5", "ALE/Frostbite-v5"],
    ["ALE/Gopher-v5", "ALE/Gravitar-v5", "ALE/Hero-v5"],
    ["ALE/IceHockey-v5", "ALE/JamesBond-v5", "ALE/JourneyEscape-v5"],
    ["ALE/Kangaroo-v5", "ALE/Krull-v5", "ALE/KungFuMaster-v5"],
    ["ALE/MontezumaRevenge-v5", "ALE/MsPacman-v5", "ALE/NameThisGame-v5"],
    ["ALE/Phoenix-v5", "ALE/Pitfall-v5", "ALE/Pong-v5"],
    ["ALE/Pooyan-v5", "ALE/PrivateEye-v5", "ALE/Qbert-v5"],
    ["ALE/RiverRaid-v5", "ALE/RoadRunner-v5", "ALE/RobotTank-v5"],
    ["ALE/Seaquest-v5", "ALE/Skiing-v5", "ALE/Solaris-v5"],
    ["ALE/SpaceInvaders-v5", "ALE/StarGunner-v5", "ALE/Tennis-v5"],
    ["ALE/TimePilot-v5", "ALE/Tutankham-v5", "ALE/UpNDown-v5"],
    ["ALE/Venture-v5", "ALE/VideoPinball-v5", "ALE/WizardOfWor-v5"],
    ["ALE/Zaxxon-v5", "ALE/Asteroids-v5", "ALE/Centipede-v5"]
]


# 训练脚本路径
training_script_path = '../algorithms/dqn.py'

# 遍历每个批次
for batch in atari_game_batches:
    # 为每个游戏启动一个screen会话来运行训练脚本
    for game in batch:
        print('start training: '+str(game))
        screen_command = f'screen -dmS train_{game.split("/")[-1]} python {training_script_path} --env_name {game}'
        subprocess.run(screen_command, shell=True, check=True)

    # 等待当前批次的所有训练完成
    for game in batch:
        subprocess.run(f'screen -S train_{game.split("/")[-1]} -X quit', shell=True)
