import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime

ROUND_START_PATTERN = re.compile(r"\[ROUND\s+(\d+)\]")

METRIC_PATTERNS = {
    "train_loss": re.compile(r"train_loss[=:]\s*([0-9.]+)", re.IGNORECASE),
    "train_accuracy": re.compile(r"train_accuracy[=:]\s*([0-9.]+)", re.IGNORECASE),
    "test_loss": re.compile(r"test_loss[=:]\s*([0-9.]+)", re.IGNORECASE),
    "test_accuracy": re.compile(r"test_accuracy[=:]\s*([0-9.]+)", re.IGNORECASE),
}

ROUND_METRIC_PATTERN = re.compile(r"Round\s+(\d+)", re.IGNORECASE)


def parse_round_start(line):
    m = ROUND_START_PATTERN.search(line)
    return int(m.group(1)) if m else None


def parse_round_metric_line(line):
    m = ROUND_METRIC_PATTERN.search(line)
    if not m:
        return None, {}

    round_num = int(m.group(1))
    metrics = {}

    for name, pattern in METRIC_PATTERNS.items():
        found = pattern.search(line)
        if found:
            metrics[name] = float(found.group(1))

    return round_num, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.cmd or args.cmd[0] != "--":
        print("Usage: python run_with_round_timing.py ... -- <command>")
        sys.exit(1)

    cmd = args.cmd[1:]

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    current_round = None
    current_start = None
    metrics_by_round = {}

    fieldnames = [
        "method",
        "dataset",
        "model",
        "round",
        "start_time",
        "end_time",
        "round_seconds",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]

    def write_round(writer, csvfile, round_num, start_time, end_time):
        metrics = metrics_by_round.get(round_num, {})

        writer.writerow({
            "method": args.method,
            "dataset": args.dataset,
            "model": args.model,
            "round": round_num,
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "round_seconds": round((end_time - start_time).total_seconds(), 3),
            "train_loss": metrics.get("train_loss", ""),
            "train_accuracy": metrics.get("train_accuracy", ""),
            "test_loss": metrics.get("test_loss", ""),
            "test_accuracy": metrics.get("test_accuracy", ""),
        })
        csvfile.flush()

    with open(args.csv, "w", newline="") as csvfile, open(args.log, "w") as logfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            timestamp = datetime.now()
            timestamp_str = timestamp.isoformat(timespec="seconds")

            stamped_line = f"[{timestamp_str}] {line}"
            print(stamped_line, end="")
            logfile.write(stamped_line)
            logfile.flush()

            metric_round, found_metrics = parse_round_metric_line(line)
            if metric_round is not None and found_metrics:
                metrics_by_round.setdefault(metric_round, {}).update(found_metrics)

            detected_round = parse_round_start(line)

            if detected_round is not None:
                if current_round is not None and current_start is not None:
                    write_round(writer, csvfile, current_round, current_start, timestamp)

                current_round = detected_round
                current_start = timestamp

        return_code = process.wait()

        if current_round is not None and current_start is not None:
            end_time = datetime.now()
            write_round(writer, csvfile, current_round, current_start, end_time)

    sys.exit(return_code)


if __name__ == "__main__":
    main()