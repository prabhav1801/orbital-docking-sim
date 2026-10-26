import sys
import os

# Add current directory to python path so 'src' resolves correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.ppo_docking import train_ppo_agent

if __name__ == "__main__":
    train_ppo_agent()
