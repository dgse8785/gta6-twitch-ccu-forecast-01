# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dgse8785/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dgse8785/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour from the first hour's print, as five quantiles per hour (5%, 25%, 50%, 75%, 95%). The model takes weighted quantiles of log(viewers_h / viewers_1) across past big launch days shipped in the case file — a Gaussian kernel on launch size, boosted 3x for console-only launches with a rolling midnight-local unlock — with the spread floored at 0.18 log units. Python standard library only, deterministic, ~0.04 s per case.

Practice score (pinball loss on the log scale, lower is better): **0.277 mean** across the four practice launches (Wolverine 0.566, Red Dead Redemption 2 0.202, God of War Ragnarök 0.147, Marvel's Spider-Man 2 0.192) vs the house persistence baseline's 0.387.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dc5kkh81&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
