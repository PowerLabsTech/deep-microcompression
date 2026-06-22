#!/usr/bin/env python3
"""
Launch DMC NAS data-generation jobs on Vertex AI.

Features:
  - Distributes 20 jobs evenly across 4 GCP regions (5 jobs per region).
  - GCS-backed checkpoint: tracks job status so re-runs skip completed /
    in-flight jobs automatically.
  - --status flag shows which jobs are done, running, or pending.

Workflow:
  1. Generate config pools locally for each experiment:
       cd experiments/paper_experiments/<exp>
       python generate_config_pool.py --n_configs 1000
  2. Launch:
       python launch_nas_vertexai.py --all
  3. Monitor:
       python launch_nas_vertexai.py --all --status
  4. Re-run after a failure — already-completed jobs are skipped automatically.

Prerequisites:
  - gcloud auth login && gcloud auth application-default login
  - pip install google-cloud-aiplatform

Usage:
  export GCP_PROJECT=your-project-id
  export GCS_BUCKET=dmc-nas-data

  python launch_nas_vertexai.py --experiments lenet5 vgg13
  python launch_nas_vertexai.py --all
  python launch_nas_vertexai.py --all --dry_run
  python launch_nas_vertexai.py --all --status
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

import torch

# ---------------------------------------------------------------------------
# Configuration  — edit these or export as env vars
# ---------------------------------------------------------------------------

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT", "valued-lambda-490314-q3")
GCS_BUCKET     = os.environ.get("GCS_BUCKET", "nilm-490314-q3")
GCS_PREFIX     = "dmc-nas-data"  # subdirectory within the shared bucket

# Jobs are spread round-robin across these locations (5 jobs per location for 20 total)
LOCATIONS: List[str] = [
    "us-central1",
    "europe-west4",
    "asia-east1",
    "us-west1",
]

MACHINE_TYPE  = "n1-standard-8"
ACCELERATOR   = "NVIDIA_TESLA_V100"
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

PACKAGE_NAME    = "deep_microcompression"
PACKAGE_VERSION = "0.1.0"
LOCAL_DIST_PATH = f"dist/{PACKAGE_NAME}-{PACKAGE_VERSION}.tar.gz"
GCS_PACKAGE_URI = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/packages/{PACKAGE_NAME}-{PACKAGE_VERSION}.tar.gz"


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

@dataclass
class NASExperiment:
    key:            str
    display:        str
    module:         str
    gcs_subdir:     str
    extra_args:     List[str] = field(default_factory=list)
    needs_baseline: bool = False

    @property
    def _gcs_base(self) -> str:
        return f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{self.gcs_subdir}"

    @property
    def pool_gcs_uri(self) -> str:
        return f"{self._gcs_base}/config_pool.pth"

    @property
    def baseline_gcs_uri(self) -> str:
        return f"{self._gcs_base}/baseline.pth"

    @property
    def output_gcs_dir(self) -> str:
        return f"{self._gcs_base}/results/"

    @property
    def checkpoint_gcs_uri(self) -> str:
        return f"{self._gcs_base}/launch_checkpoint.json"

    @property
    def local_pool_path(self) -> str:
        root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root, "experiments", "paper_experiments",
                            self.gcs_subdir, "config_pool.pth")

    @property
    def local_baseline_path(self) -> str:
        root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root, "experiments", "paper_experiments",
                            self.gcs_subdir, "models", "baseline.pth")

    def result_gcs_uri(self, start: int, end: int) -> str:
        return f"{self.output_gcs_dir}nas_{start}_{end}.pth"


EXPERIMENTS: Dict[str, NASExperiment] = {
    "lenet5": NASExperiment(
        key="lenet5", display="dmc-lenet5-mnist",
        module="experiments.paper_experiments.lenet5_mnist.generate_nas_data",
        gcs_subdir="lenet5_mnist", needs_baseline=True,
    ),
    "mobilenetv1": NASExperiment(
        key="mobilenetv1", display="dmc-mobilenetv1-cifar100",
        module="experiments.paper_experiments.mobilenetv1_cifar100.generate_nas_data",
        gcs_subdir="mobilenetv1_cifar100",
        extra_args=["--width_mult", "0.5"], needs_baseline=True,
    ),
    "mobilenetv2": NASExperiment(
        key="mobilenetv2", display="dmc-mobilenetv2-cifar100",
        module="experiments.paper_experiments.mobilenetv2_cifar100.generate_nas_data",
        gcs_subdir="mobilenetv2_cifar100",
    ),
    "resnet56": NASExperiment(
        key="resnet56", display="dmc-resnet56-cifar100",
        module="experiments.paper_experiments.resnet56_cifar100.generate_nas_data",
        gcs_subdir="resnet56_cifar100",
    ),
    "vgg13": NASExperiment(
        key="vgg13", display="dmc-vgg13-cifar100",
        module="experiments.paper_experiments.vgg13_cifar100.generate_nas_data",
        gcs_subdir="vgg13_cifar100",
    ),
}

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------


# ── Shell helpers ────────────────────────────────────────────────────────────

def _run(cmd: list, error_prefix: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{error_prefix}\n"
            f"  command : {' '.join(cmd)}\n"
            f"  stdout  : {result.stdout.strip()}\n"
            f"  stderr  : {result.stderr.strip()}"
        )


def gcs_exists(uri: str) -> bool:
    """Return True if the GCS object exists."""
    result = subprocess.run(["gsutil", "-q", "stat", uri], capture_output=True)
    return result.returncode == 0


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(exp: NASExperiment) -> dict:
    """
    Download and parse the checkpoint JSON from GCS.
    Returns an empty checkpoint dict if none exists yet.
    """
    if not gcs_exists(exp.checkpoint_gcs_uri):
        return {"experiment": exp.key, "jobs": {}}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        _run(["gsutil", "cp", exp.checkpoint_gcs_uri, tmp],
             "Failed to download checkpoint")
        with open(tmp) as f:
            return json.load(f)
    finally:
        os.unlink(tmp)


def save_checkpoint(exp: NASExperiment, checkpoint: dict) -> None:
    """Upload the checkpoint dict to GCS."""
    checkpoint["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(checkpoint, f, indent=2)
        tmp = f.name
    try:
        _run(["gsutil", "cp", tmp, exp.checkpoint_gcs_uri],
             "Failed to upload checkpoint")
    finally:
        os.unlink(tmp)


def mark_job(checkpoint: dict, job_id: str, status: str,
             location: str = "", vertex_job: str = "",
             start: int = None, end: int = None) -> None:
    entry = checkpoint.setdefault("jobs", {}).setdefault(str(job_id), {})
    entry["status"] = status
    if location:
        entry["location"] = location
    if vertex_job:
        entry["vertex_job"] = vertex_job
    if start is not None:
        entry["start"] = start
    if end is not None:
        entry["end"] = end
    if status == "launched":
        entry["launched_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    elif status == "completed":
        entry["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()


def job_status(exp: NASExperiment, checkpoint: dict, job_id: str) -> str:
    """
    Determine the current status of a job.
    Returns: "completed" | "launched" | "pending"
    """
    entry = checkpoint.get("jobs", {}).get(job_id, {})
    start = entry.get("start")
    end   = entry.get("end")
    # Ground truth: result file in GCS means the job finished successfully
    if start is not None and end is not None and gcs_exists(exp.result_gcs_uri(start, end)):
        return "completed"
    return entry.get("status", "pending")


# ── Package helpers ──────────────────────────────────────────────────────────

def build_package() -> None:
    logger.info("Building source distribution …")
    _run([sys.executable, "setup.py", "sdist", "--formats=gztar"],
         "Failed to build source distribution")
    if not os.path.exists(LOCAL_DIST_PATH):
        raise FileNotFoundError(f"Expected dist at {LOCAL_DIST_PATH}")
    logger.info(f"Built: {LOCAL_DIST_PATH}")


def upload_package() -> None:
    logger.info(f"Uploading package → {GCS_PACKAGE_URI} …")
    _run(["gsutil", "cp", LOCAL_DIST_PATH, GCS_PACKAGE_URI],
         "Failed to upload package — check gcloud auth and bucket access")
    logger.info("Package upload complete.")


def upload_experiment_assets(exp: NASExperiment, dry_run: bool) -> None:
    if not os.path.exists(exp.local_pool_path):
        raise FileNotFoundError(
            f"Pool missing for {exp.key}: {exp.local_pool_path}\n"
            f"  Run: python experiments/paper_experiments/{exp.gcs_subdir}/generate_config_pool.py"
        )
    logger.info(f"[{exp.key}] Uploading pool → {exp.pool_gcs_uri}")
    if not dry_run:
        _run(["gsutil", "cp", exp.local_pool_path, exp.pool_gcs_uri],
             f"Failed to upload pool for {exp.key}")

    if exp.needs_baseline:
        if not os.path.exists(exp.local_baseline_path):
            raise FileNotFoundError(
                f"Baseline missing for {exp.key}: {exp.local_baseline_path}\n"
                "  Run reproduce.ipynb first."
            )
        logger.info(f"[{exp.key}] Uploading baseline → {exp.baseline_gcs_uri}")
        if not dry_run:
            _run(["gsutil", "cp", exp.local_baseline_path, exp.baseline_gcs_uri],
                 f"Failed to upload baseline for {exp.key}")


# ── Vertex AI helpers ────────────────────────────────────────────────────────

def _staging_bucket(location: str) -> str:
    """Return (and create if needed) a regional Vertex AI staging bucket."""
    bucket_name = f"nilm-vertex-{location}"
    uri = f"gs://{bucket_name}"
    result = subprocess.run(
        ["gcloud", "storage", "buckets", "describe", uri, "--format=value(name)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.info(f"Creating staging bucket {uri} in {location} …")
        _run(
            ["gcloud", "storage", "buckets", "create", uri,
             f"--location={location}",
             f"--project={GCP_PROJECT_ID}",
             "--uniform-bucket-level-access"],
            f"Failed to create staging bucket {uri}",
        )
        logger.info(f"Staging bucket created: {uri}")
    return uri


def launch_vertexai_job(
    display_name: str,
    module_name:  str,
    training_args: List[str],
    location: str,
    sync: bool = False,
) -> str:
    """Launch a Vertex AI CustomPythonPackageTrainingJob. Returns the job resource name."""
    from google.cloud import aiplatform  # type: ignore[import-not-found]

    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=location,
        staging_bucket=_staging_bucket(location),
    )

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=GCS_PACKAGE_URI,
        python_module_name=module_name,
        container_uri=CONTAINER_URI,
    )

    job.run(
        replica_count=1,
        machine_type=MACHINE_TYPE,
        accelerator_count=1,
        accelerator_type=ACCELERATOR,
        args=training_args,
        sync=sync,
    )
    logger.info(f"Launched [{location}]: {display_name}")
    try:
        return job.resource_name or display_name
    except RuntimeError:
        return display_name


# ── Status command ───────────────────────────────────────────────────────────

def print_status(exp: NASExperiment) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"Status: {exp.display}")
    checkpoint = load_checkpoint(exp)
    jobs = checkpoint.get("jobs", {})

    if not jobs:
        logger.info("  No jobs dispatched yet.")
        logger.info(f"  results: {exp.output_gcs_dir}")
        return

    counts = {"completed": 0, "launched": 0, "pending": 0}
    for job_id in sorted(jobs, key=lambda j: int(j.split("-")[0])):
        status   = job_status(exp, checkpoint, job_id)
        entry    = jobs[job_id]
        start    = entry.get("start", "?")
        end      = entry.get("end", "?")
        location = entry.get("location", "?")
        counts[status] += 1
        icon = {"completed": "✓", "launched": "⏳", "pending": "·"}[status]
        logger.info(f"  {icon} [{start}:{end}] [{location}]  {status}")

    pool_head = 0
    if os.path.exists(exp.local_pool_path):
        pool_head = torch.load(exp.local_pool_path, weights_only=False).get("head", 0)

    logger.info(
        f"\n  completed={counts['completed']}  launched={counts['launched']}  "
        f"total dispatched={sum(counts.values())}  pool head={pool_head}"
    )
    logger.info(f"  results: {exp.output_gcs_dir}")


# ── Main launch logic ────────────────────────────────────────────────────────

def launch_experiment(
    exp: NASExperiment, dry_run: bool, sync: bool,
    n_jobs: int, configs_per_job: int, location: str,
) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"Launching: {exp.display}")

    checkpoint = load_checkpoint(exp)
    pool_data  = torch.load(exp.local_pool_path, weights_only=False)
    head       = pool_data["head"]
    n_available = len(pool_data["configs"]) - head

    if n_jobs * configs_per_job > n_available:
        raise ValueError(
            f"Not enough configs: need {n_jobs * configs_per_job}, "
            f"only {n_available} remain (pool head={head})"
        )

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    launched  = 0
    skipped   = 0

    for i in range(n_jobs):
        start  = head + i * configs_per_job
        end    = start + configs_per_job
        job_id = f"{start}-{end}"

        status = job_status(exp, checkpoint, job_id)
        if status in ("completed", "launched"):
            logger.info(f"  [{start}:{end}] already {status} — skipping")
            skipped += 1
            continue

        training_args = [
            "--start",          str(start),
            "--end",            str(end),
            "--pool_gcs_uri",   exp.pool_gcs_uri,
            "--output_gcs_dir", exp.output_gcs_dir,
        ]
        if exp.needs_baseline:
            training_args += ["--baseline_gcs_uri", exp.baseline_gcs_uri]
        training_args += exp.extra_args

        display_name = f"{exp.display}_{start}_{end}_{timestamp}"
        logger.info(f"  [{start}:{end}] launching → {location}")

        if dry_run:
            logger.info(f"    [DRY RUN] module : {exp.module}")
            logger.info(f"    [DRY RUN] args   : {' '.join(training_args)}")
        else:
            vertex_job = launch_vertexai_job(
                display_name, exp.module, training_args, location, sync=sync,
            )
            mark_job(checkpoint, job_id, "launched",
                     location=location, vertex_job=vertex_job,
                     start=start, end=end)
            save_checkpoint(exp, checkpoint)
        launched += 1

    # Advance head past all claimed slices so the next run picks up from here
    pool_data["head"] = head + n_jobs * configs_per_job
    torch.save(pool_data, exp.local_pool_path)

    logger.info(f"\n  launched={launched}  skipped={skipped}  / {n_jobs} requested")
    if not dry_run and launched:
        logger.info(f"  results will land in: {exp.output_gcs_dir}")
        logger.info(f"  checkpoint: {exp.checkpoint_gcs_uri}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DMC NAS jobs on Vertex AI")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), required=True,
                        help="Experiment to run: " + ", ".join(EXPERIMENTS))
    parser.add_argument("--status",      action="store_true",
                        help="Print job status only — do not launch anything")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Print what would be launched without submitting")
    parser.add_argument("--sync",        action="store_true",
                        help="Wait for each job to complete (use only for single-job testing)")
    parser.add_argument("--skip_build",  action="store_true",
                        help="Skip sdist build (re-use existing dist/)")
    parser.add_argument("--skip_upload", action="store_true",
                        help="Skip package + pool upload (assets already in GCS)")
    parser.add_argument("--location",        required=True, choices=LOCATIONS,
                        help="GCP region to launch jobs in")
    parser.add_argument("--n_jobs",          type=int, default=20,
                        help="Number of jobs to launch")
    parser.add_argument("--configs_per_job", type=int, default=50,
                        help="Configs each job processes")
    args = parser.parse_args()

    if not GCP_PROJECT_ID:
        raise EnvironmentError(
            "GCP_PROJECT is not set.\n"
            "  export GCP_PROJECT=your-project-id"
        )

    exp = EXPERIMENTS[args.experiment]

    logger.info(f"Experiment  : {exp.key}")
    logger.info(f"Location    : {args.location}")
    logger.info(f"Jobs        : {args.n_jobs}  ({args.configs_per_job} configs each"
                f" = {args.n_jobs * args.configs_per_job} total NAS samples)")

    # Status-only mode
    if args.status:
        print_status(exp)
        return

    # Build & upload package once
    if not args.skip_build and not args.dry_run:
        build_package()
    if not args.skip_upload and not args.dry_run:
        upload_package()

    if not args.skip_upload:
        upload_experiment_assets(exp, dry_run=args.dry_run)
    launch_experiment(
        exp, dry_run=args.dry_run, sync=args.sync,
        n_jobs=args.n_jobs, configs_per_job=args.configs_per_job,
        location=args.location,
    )

    if not args.dry_run:
        logger.info("\nAll jobs submitted.")
        logger.info(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={GCP_PROJECT_ID}")
        logger.info(f"Status:  python launch_nas_vertexai.py --experiment {exp.key} --location {args.location} --status")


if __name__ == "__main__":
    main()
