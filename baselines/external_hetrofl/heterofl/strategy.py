import random


MODEL_RATES = {
    "a": 1.0,
    "b": 0.5,
    "c": 0.25,
    "d": 0.125,
    "e": 0.0625,
}


def parse_model_mode(model_mode):
    levels = []

    for part in model_mode.split("-"):
        level = part[0]
        count = int(part[1:])

        for _ in range(count):
            levels.append(level)

    return levels


def sample_model_rate(model_mode="a1-b1-c1"):
    levels = parse_model_mode(model_mode)
    level = random.choice(levels)
    return MODEL_RATES[level], level
