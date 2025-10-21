import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('/content/drive')

simlex_path = '/content/drive/MyDrive/SimLex-999.txt'
df = pd.read_csv(simlex_path, sep='\t')

simlex_scores = {}
assoc_scores = {}

safe_targets = set()

for (w1, w2, pos), score in simlex_scores.items():
    if score is not None and score > 0:
        safe_targets.add((w1, pos))
        safe_targets.add((w2, pos))

for _, row in df.iterrows():
    w1, w2, pos = row['word1'], row['word2'], row['POS']
    key1 = (w1.lower(), w2.lower(), pos)
    key2 = (w2.lower(), w1.lower(), pos)
    simlex_scores[key1] = row['SimLex999']
    simlex_scores[key2] = row['SimLex999']
    assoc_scores[key1] = row['Assoc(USF)']
    assoc_scores[key2] = row['Assoc(USF)']

vocab = {}
for _, row in df.iterrows():
    for word in [row['word1'], row['word2']]:
        key = (word.lower(), row['POS'])
        vocab.setdefault(row['POS'], set()).add(word.lower())

def similarity(trial_id):
    method = "vector"

    while True:
        pos = random.choice(list(vocab.keys()))
        candidates = list(vocab[pos])
        eligible_targets = [w for w in candidates if (w, pos) in safe_targets]
        if len(eligible_targets) >= 2:
            break

    for _ in range(5):
        try:
            target = random.choice(eligible_targets)
            remaining_candidates = [w for w in candidates if w != target]
            if not remaining_candidates:
                continue

            current = random.choice(remaining_candidates)
            remaining_candidates.remove(current)
            total_guesses = 0

            while current != target:
                if not remaining_candidates:
                    raise RuntimeError(f"Ran out of candidates without finding target {target}")

                guess = random.choice(remaining_candidates)
                remaining_candidates.remove(guess)

                score_current = simlex_scores.get((current, target, pos))
                score_guess = simlex_scores.get((guess, target, pos))

                while score_guess is None:
                    if not remaining_candidates:
                        raise RuntimeError(f"Ran out of candidates without valid guesses for target {target}")
                    guess = random.choice(remaining_candidates)
                    remaining_candidates.remove(guess)
                    score_guess = simlex_scores.get((guess, target, pos))

                total_guesses += 1

                if score_current is None or score_guess > score_current:
                    current = guess
                    score_current = score_guess

            return {
                "trial": trial_id,
                "total": total_guesses,
                "method": method,
                "target": f"{target}|{pos}",
                "highest": f"{current}|{pos}",
                "guess": f"{guess}|{pos}"
            }

        except RuntimeError:
            continue

    raise RuntimeError(f"Failed to complete trial after multiple attempts")

def association(trial_id):
    method = "association"

    while True:
        pos = random.choice(list(vocab.keys()))
        candidates = list(vocab[pos])
        eligible_targets = [w for w in candidates if (w, pos) in safe_targets]
        if len(eligible_targets) >= 2:
            break

    for _ in range(5):
        try:
            target = random.choice(eligible_targets)
            remaining_candidates = [w for w in candidates if w != target]
            if not remaining_candidates:
                continue

            current = random.choice(remaining_candidates)
            remaining_candidates.remove(current)
            total_guesses = 0

            while current != target:
                if not remaining_candidates:
                    raise RuntimeError(f"Ran out of candidates without finding target {target}")

                guess = random.choice(remaining_candidates)
                remaining_candidates.remove(guess)

                score_current = assoc_scores.get((current, target, pos))
                score_guess = assoc_scores.get((guess, target, pos))

                while score_guess is None:
                    if not remaining_candidates:
                        raise RuntimeError(f"Ran out of candidates without valid guesses for target {target}")
                    guess = random.choice(remaining_candidates)
                    remaining_candidates.remove(guess)
                    score_guess = assoc_scores.get((guess, target, pos))

                total_guesses += 1

                if score_current is None or score_guess > score_current:
                    current = guess
                    score_current = score_guess

            return {
                "trial": trial_id,
                "total": total_guesses,
                "method": method,
                "target": f"{target}|{pos}",
                "highest": f"{current}|{pos}",
                "guess": f"{guess}|{pos}"
            }

        except RuntimeError:
            continue

    raise RuntimeError(f"Failed to complete trial after multiple attempts")

save_file = '/content/trials.csv'

if os.path.exists(save_file):
    trials_df = pd.read_csv(save_file)
    start_trial = trials_df['trial'].max() + 1
else:
    trials_df = pd.DataFrame(columns=["trial", "total", "method", "target", "highest", "guess"])
    start_trial = 1

new_trials = 2 ##will update if works

for trial_id in tqdm(range(start_trial, start_trial + new_trials), desc="Running trials"):
    if random.random() < 0.5:
        result = similarity(trial_id)
    else:
        result = association(trial_id)

    trials_df = pd.concat([trials_df, pd.DataFrame([result])], ignore_index=True)
    trials_df.to_csv(save_file, index=False)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.histplot(data=trials_df, x="total", hue="method", kde=True, bins=30)
plt.title("Histogram of Guesses Needed to Find Target Word")
plt.xlabel("Total Guesses")
plt.ylabel("Count")
plt.grid(True)
plt.show()

print(trials_df.groupby('method')['total'].describe())

