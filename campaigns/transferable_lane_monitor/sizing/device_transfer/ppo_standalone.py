"""PPO directly ranks the shared width proposals using observed target history and coordinates."""
from ppo_critic_buffer import main
from ppo_sizing import ROOT, train_actor

ARM = "standalone"
ARCHIVE = ROOT / "ppo_standalone_measurements.zip"


def train(rows):
    return {ARM: train_actor(rows, [None] * len(rows), gamma=1.)}


if __name__ == "__main__":
    main(ARM, ARCHIVE, train, "ppo_standalone")
