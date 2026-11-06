"""RealPDE Track 1 (Sim2Real) -- Decision Phase entry point.

train(data_dir, output_dir, device)
    Re-runs the release pipeline (scripts/release/run_all.sh) from scratch on
    the public training data:
      1. prep       train_real/*.h5 -> 32x64 float32 memmap (output_dir/work/data32)
      2. train x8   4 ship nets (all real data; the shipped ensemble) and
                    4 CV nets (Re 13950 + 22875 held out; used only to fit the
                    interval-width table on held-out residuals)
      3. calibrate  width table + variance head + archive assembly
    The assembled inference bundle (the same layout as the Codabench archive)
    is unpacked into output_dir/predictor; per-step logs go to output_dir/logs.

load_predictor(output_dir, device)
    Imports output_dir/predictor/submission.py (the archive's inference code)
    and returns its SubmissionModel, whose predict(input_array) takes
    (N, 20, 32, 64, 3) float32 in physical units and returns
    {"prediction", "lower", "upper"} of the same shape.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHIP = ("shipA_s250", "shipB_s250", "shipA_s350", "shipB_s350")
CV = ("cvA_s250", "cvB_s250", "cvA_s270", "cvB_s270")


def _run(cmd, env, log_path):
    """Run one pipeline step, streaming its output to stdout and a log file."""
    print(f"[solution] $ {' '.join(cmd)}  (log: {log_path})", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    print(f"[solution] exit {rc} after {time.time() - t0:.0f} s", flush=True)
    if rc != 0:
        raise RuntimeError(f"step failed ({rc}): {' '.join(cmd)} -- see {log_path}")


def train(data_dir, output_dir, device):
    assert str(device).startswith("cuda"), "training needs a CUDA device"
    work = os.path.join(output_dir, "work")
    logs = os.path.join(output_dir, "logs")
    os.makedirs(work, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ,
               RPDE_RAW=os.path.abspath(data_dir),
               RPDE_DATA32=os.path.join(work, "data32"),
               RPDE_WORK=work,
               PYTHON=sys.executable,
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "8")
    t0 = time.time()

    # The method trains on train_real only; train_sim is not used.
    _run([sys.executable, "scripts/prep_data.py", "real"], env,
         os.path.join(logs, "prep.log"))
    for name in SHIP + CV:
        _run(["bash", "scripts/release/run_all.sh", "train", name], env,
             os.path.join(logs, f"{name}.log"))
    _run(["bash", "scripts/release/run_all.sh", "calibrate"], env,
         os.path.join(logs, "calibrate.log"))

    pred_dir = os.path.join(output_dir, "predictor")
    shutil.rmtree(pred_dir, ignore_errors=True)
    with zipfile.ZipFile(os.path.join(work, "submission.zip")) as z:
        z.extractall(pred_dir)
    shutil.rmtree(env["RPDE_DATA32"], ignore_errors=True)   # 1.1 GB, rebuilt by prep
    print(f"[solution] train done in {(time.time() - t0) / 3600:.2f} h -> {pred_dir}",
          flush=True)


def load_predictor(output_dir, device):
    pred_dir = os.path.join(output_dir, "predictor")
    spec = importlib.util.spec_from_file_location(
        "rpde_submission", os.path.join(pred_dir, "submission.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SubmissionModel(pred_dir, device)
