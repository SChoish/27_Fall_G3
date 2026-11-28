"""Compare cross-fitted and latest value predictions on identical on-policy transitions."""
import json
from zipfile import ZipFile

import numpy as np

from ppo_critic_buffer import ARCHIVE, ARM
from ppo_sizing import train, train_actor


def main():
    with ZipFile(ARCHIVE) as archive:
        sources = json.loads(archive.read("source.json"))
        recorded = json.loads(archive.read("training.json"))
    for pdk, rows in sources.items():
        models = [train(rows[:i] + rows[i+1:], only=("anon400",))["anon400"]
                  for i in range(len(rows))]
        errors = []

        def observe(returns, latest, chosen):
            errors.append(np.column_stack(((returns - latest)**2, (returns - chosen)**2)))

        result = train_actor(rows, models, gamma=1., buffer_size=4, observe_baseline=observe)
        np.testing.assert_allclose(result["weights"], recorded[pdk][ARM]["weights"], rtol=0, atol=1e-10)
        assert result["critic_age_counts"] == recorded[pdk][ARM]["critic_age_counts"]
        mse = np.concatenate(errors).mean(axis=0)
        print(f"{pdk} transitions={sum(map(len, errors))} latest={mse[0]:.8f} selected={mse[1]:.8f} "
              f"relative_change={mse[1]/mse[0]-1:+.2%}")


if __name__ == "__main__":
    main()
