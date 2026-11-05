#!/bin/bash
set -euo pipefail
module load pytorch-conda/2.8
cd "$(dirname "$0")/../.."

task_mode=${1:-smoke}
task_python=${TRACK4_PYTHON:-$(command -v python)}
export TRACK4_PYTHON="$task_python"

[[ -x "$task_python" ]] || {
    echo "Python missing: $task_python. Set TRACK4_PYTHON." >&2
    exit 1
}
[[ -f track4/jobs/prepared.json ]] || {
    echo "Run prepare first." >&2
    exit 1
}
mkdir -p track4/logs

submission_log="track4/jobs/${task_mode}_submission.txt"
[[ ! -e "$submission_log" ]] || {
    echo "Submission already recorded: $submission_log" >&2
    echo "Check those jobs before submitting again." >&2
    exit 1
}

if [[ "$task_mode" == smoke ]]; then
    (set -o noclobber; : > "$submission_log")
    job_id=$(sbatch --parsable --job-name=t4_smoke --array=0-23%4 \
        --time=00:30:00 \
        --output='track4/logs/smoke_%A_%a.out' \
        --error='track4/logs/smoke_%A_%a.err' \
        track4/jobs/experiment.sbatch smoke)
    printf 'smoke %s\n' "$job_id" | tee -a "$submission_log"

elif [[ "$task_mode" == production ]]; then
    "$task_python" track4/jobs/manage.py check-smoke
    (set -o noclobber; : > "$submission_log")

    for task_group in iid10 iid20 niid10 niid20; do
        job_id=$(sbatch --parsable --job-name="t4_$task_group" \
            --array=0-89%1 \
            --output="track4/logs/${task_group}_%A_%a.out" \
            --error="track4/logs/${task_group}_%A_%a.err" \
            track4/jobs/experiment.sbatch production "$task_group")
        printf '%s %s\n' "$task_group" "$job_id" | tee -a "$submission_log"
    done
else
    echo "Usage: bash track4/jobs/submit.sh smoke|production" >&2
    exit 1
fi
