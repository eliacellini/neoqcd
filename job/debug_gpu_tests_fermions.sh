#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-gpu-tests-fermions
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo_work/INF26_sft/ecellini/neoqcd_test

# Runs in the test worktree; never in the production checkout.
# reports/ must exist in the test worktree before sbatch (SLURM opens -o/-e before the script runs).

# GPU-node checks of the fermion code on Leonardo's torch 2.0 / Python 3.11 (code validation, not
# measurements). Points (POINTS, run in order; a failing point does not stop the next ones):
#   tests   the fused-kernel device/stream tests (test_fermions_fused.CudaDeviceAndStreamTest; the
#           second-device test needs two GPUs: submit with --gpus-per-node=2), then the whole fermion
#           unit-test suite (python -m unittest discover -s test -t . -p 'test_fermions*'), one
#           discover per test file, the files run in parallel on the job's CPUs (OMP_NUM_THREADS=1).
#           Logs in output/gpu_tests_fermions/<job id>/<file>.log, summary in the job log.
#   HMC checks with main/main_hmc.py, always from scratch in output/hmc_checks_fermions/<job id>/<point>/
#   (never resumed, never touching output/hmc_fermions/):
#   w4f     the w4f point of job/debug_hmc_fermions.sh (4^4, beta 1.0, kappa 0.1, B = 4, 50 + 300
#           trajectories, exact log det, --fast --kernel cuda, seed 1234): reference 0.06084(19)
#           (openQCD 0.060765(28)); same seed as the 14 November run of that point;
#   hb8f    fast Hasenbusch (--kappa-heavy 0.14, nested layout), 8^4, beta 5.6, kappa 0.153, B = 4,
#           cold start, 10 + 20 trajectories;
#   stag8f  fast staggered (N_f = 4), 8^4, beta 5.6, mass 0.1, B = 4, cold start, 10 + 20 trajectories;
#   w8fast  plain --fast Wilson at the hb8f point (cost reference for hb8f);
#   hb4f / hb4r    fast / reference Hasenbusch, 4^4, beta 5.6, kappa 0.153, kappa' 0.14, B = 4, 20
#           trajectories from cold, same seed (dH and accept pattern compared trajectory by trajectory);
#   stag4f / stag4r  fast / reference staggered, 4^4, beta 5.6, mass 0.1, as hb4f / hb4r.
# Usage: sbatch --gpus-per-node=2 --cpus-per-task=16 job/debug_gpu_tests_fermions.sh   (POINTS=tests)
#        POINTS="w4f hb8f stag8f w8fast hb4f hb4r stag4f stag4r" sbatch job/debug_gpu_tests_fermions.sh
#        [extra flags appended to every main_hmc.py point]

set -uo pipefail

module load profile/deeplrn
module load cineca-ai/
# The environment lives only in the production checkout: activate it by absolute path, but import
# the package of the test worktree (PYTHONPATH first).
source /leonardo_work/INF26_sft/ecellini/neoqcd/.sunenv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p reports
echo "=== worktree $PWD commit $(git rev-parse HEAD)"
echo "=== python $(python -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__, "cuda devices", torch.cuda.device_count())')"
echo "=== nvcc: $(command -v nvcc || echo none) $(nvcc --version 2>/dev/null | tail -1)"
export OMP_NUM_THREADS=1

POINTS="${POINTS:-tests}"
SEED="${SEED:-1234}"
JOB="${SLURM_JOB_ID:-manual}"
HMC_BASE="output/hmc_checks_fermions/${JOB}"

run_tests() {
  local logdir="output/gpu_tests_fermions/${JOB}"
  local npar="${SLURM_CPUS_PER_TASK:-8}"
  mkdir -p "$logdir"
  echo "=== fused device/stream tests $(date -Is)"
  python -m unittest test.test_fermions_fused.CudaDeviceAndStreamTest -v 2>&1 | tee "$logdir/fused_device_stream.log"
  echo "=== fermion unit-test suite $(date -Is): $(ls test/test_fermions*.py | wc -l) files, ${npar} in parallel -> ${logdir}"
  export LOGDIR="$logdir"
  ls test/test_fermions*.py | xargs -P "$npar" -n 1 bash -c '
    m=$(basename "$1" .py); t0=$(date +%s)
    python -m unittest discover -s test -t . -p "$m.py" -v > "$LOGDIR/$m.log" 2>&1
    echo "$? $(( $(date +%s) - t0 ))" > "$LOGDIR/$m.rc"' _
  echo "=== suite done $(date -Is)"
  python - "$logdir" <<'EOF'
import re, sys
from pathlib import Path
logdir = Path(sys.argv[1])
tot = {"ran": 0, "fail": 0, "err": 0, "skip": 0}
bad = []
print(f"{'file':42s} {'rc':>3s} {'s':>5s} {'ran':>4s} result")
for rc_file in sorted(logdir.glob("test_fermions*.rc")):
    rc, secs = rc_file.read_text().split()
    log = rc_file.with_suffix(".log").read_text(errors="replace")
    ran = re.findall(r"^Ran (\d+) tests? in", log, re.M)
    res = re.findall(r"^(OK.*|FAILED.*)$", log, re.M)
    n = int(ran[-1]) if ran else 0
    tot["ran"] += n
    m = re.search(r"failures=(\d+)", res[-1]) if res else None
    tot["fail"] += int(m.group(1)) if m else 0
    m = re.search(r"errors=(\d+)", res[-1]) if res else None
    tot["err"] += int(m.group(1)) if m else 0
    m = re.search(r"skipped=(\d+)", res[-1]) if res else None
    tot["skip"] += int(m.group(1)) if m else 0
    print(f"{rc_file.stem:42s} {rc:>3s} {secs:>5s} {n:4d} {res[-1] if res else 'NO RESULT LINE'}")
    if rc != "0" or not res or not res[-1].startswith("OK"):
        bad.append(rc_file.stem)
print("TOTAL", tot)
print("=== skipped tests (reason):")
for log in sorted(logdir.glob("*.log")):
    for line in log.read_text(errors="replace").splitlines():
        if re.search(r"\.\.\. skipped", line):
            print(f"  {log.stem}: {line.strip()}")
for name in bad:
    print(f"=== FAILED file {name}: last 150 lines of its log")
    print("\n".join((logdir / f"{name}.log").read_text(errors="replace").splitlines()[-150:]))
EOF
}

run_hmc() {
  # run_hmc <point> <main_hmc.py arguments...>
  local name="$1"; shift
  local out="${HMC_BASE}/${name}"
  mkdir -p "$out"
  echo "=== ${name} $(date -Is) -> ${out}"
  srun python main/main_hmc.py --device cuda --seed "$SEED" --out "$out" "$@" || echo "=== ${name} FAILED (exit $?)"
}

W8=(--D 4 --T 8 --L 8 --N 3 --beta 5.6 --batch 4 --tau 1.0 --n-steps 10 --n-gauge-substeps 4
    --integrator omelyan --start cold --therm 10 --n-traj 20 --meas-every 5 --checkpoint-every 0)
W4=(--D 4 --T 4 --L 4 --N 3 --beta 5.6 --batch 4 --tau 1.0 --n-steps 8 --n-gauge-substeps 4
    --integrator omelyan --start cold --therm 0 --n-traj 20 --meas-every 5 --checkpoint-every 0)

ran_hmc=0
for point in $POINTS; do
  case "$point" in
    tests) run_tests ;;
    w4f) ran_hmc=1; run_hmc w4f \
          --D 4 --T 4 --L 4 --N 3 --fermions wilson --beta 1.0 --kappa 0.1 --batch 4 \
          --tau 1.0 --n-steps 6 --n-gauge-substeps 2 --integrator omelyan --start cold \
          --therm 50 --n-traj 300 --meas-every 1 --checkpoint-every 50 --exact-logdet --fast --kernel cuda "$@" ;;
    hb8f) ran_hmc=1; run_hmc hb8f "${W8[@]}" --fermions wilson --kappa 0.153 --kappa-heavy 0.14 --fast "$@" ;;
    stag8f) ran_hmc=1; run_hmc stag8f "${W8[@]}" --fermions staggered --mass 0.1 --fast "$@" ;;
    w8fast) ran_hmc=1; run_hmc w8fast "${W8[@]}" --fermions wilson --kappa 0.153 --fast "$@" ;;
    hb4f) ran_hmc=1; run_hmc hb4f "${W4[@]}" --fermions wilson --kappa 0.153 --kappa-heavy 0.14 --fast "$@" ;;
    hb4r) ran_hmc=1; run_hmc hb4r "${W4[@]}" --fermions wilson --kappa 0.153 --kappa-heavy 0.14 "$@" ;;
    stag4f) ran_hmc=1; run_hmc stag4f "${W4[@]}" --fermions staggered --mass 0.1 --fast "$@" ;;
    stag4r) ran_hmc=1; run_hmc stag4r "${W4[@]}" --fermions staggered --mass 0.1 "$@" ;;
    *) echo "unknown point $point" >&2 ;;
  esac
done

if [[ "$ran_hmc" == 1 ]]; then
  echo "=== HMC summary $(date -Is)"
  python - "$HMC_BASE" <<'EOF'
import csv, json, math, sys
from pathlib import Path
base = Path(sys.argv[1])

def rows(d):
    p = d / "observables.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []

def fmt(o):
    return "-" if o is None else (f"{o['value']:.6g}({o['error']:.2g})" if isinstance(o, dict) and "value" in o else str(o))

for d in sorted(p for p in base.iterdir() if p.is_dir()):
    meta_p = d / "run_meta.json"
    if not meta_p.exists():
        print(f"{d.name}: no run_meta.json"); continue
    s = json.load(open(meta_p)).get("summary", {})
    r = rows(d)
    meas = [x for x in r if x.get("therm") == "0"]
    res = [float(x["max_residual"]) for x in r if x.get("max_residual") not in (None, "")]
    dh = [abs(float(x["dH"])) for x in meas]
    print(f"{d.name}: plaq {fmt(s.get('plaquette'))} acc {fmt(s.get('acceptance'))} <e^-dH> {fmt(s.get('exp_minus_dH'))} "
          f"dH {s.get('dH')} max|dH| {max(dh) if dh else '-'} s/traj {s.get('seconds_per_traj')} "
          f"solves/traj {s.get('solves_per_traj')} cg/solve {s.get('cg_iter_per_solve')} "
          f"max_residual max {max(res) if res else '-'} (rows {len(r)}, nan {sum(math.isnan(v) for v in res)}) "
          f"logdet {fmt(s.get('logdet'))}")
for a, b in (("hb4f", "hb4r"), ("stag4f", "stag4r")):
    ra, rb = rows(base / a), rows(base / b)
    if ra and rb and len(ra) == len(rb):
        ddh = max(abs(float(x["dH"]) - float(y["dH"])) for x, y in zip(ra, rb))
        same = all(x["accepted"] == y["accepted"] for x, y in zip(ra, rb))
        dp = max(abs(float(x["plaquette"]) - float(y["plaquette"])) for x, y in zip(ra, rb))
        print(f"{a} vs {b}: rows {len(ra)} max|dH diff| {ddh:.3e} same accept pattern {same} max|plaq diff| {dp:.3e}")
    elif ra or rb:
        print(f"{a} vs {b}: rows {len(ra)} vs {len(rb)} (not compared)")
EOF
fi
echo "=== end $(date -Is)"
