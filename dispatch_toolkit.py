import argparse
import subprocess
from pathlib import Path
from subprocess import PIPE


# XXX: This needs to appear before the constants since it is used to extract user toolkit info
def _run_shell_cmd(cmd: str, hide_stderr=False):
    """Run a shell command and return the output"""
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=PIPE,
        stderr=PIPE if hide_stderr else None,
        universal_newlines=True,
    )
    return result.stdout.replace("\n", "")


TOOLKIT_USER_ORG = "snow"
TOOLKIT_USER_ACCOUNT = "snow.rg_climate_benchmark"
TOOLKIT_USER_USERNAME = "n_lehmann"
TOOLKIT_USER = f"{TOOLKIT_USER_ORG}.{TOOLKIT_USER_ACCOUNT}.{TOOLKIT_USER_USERNAME}"

# Get current git branch (used to tag code by user+branch)
GIT_BRANCH = _run_shell_cmd("git rev-parse --abbrev-ref HEAD")

# Infrastructure setup variables
# Note: Give access to new users by adding them to the eai.rg_climate_benchmark.admin role.
# TOOLKIT_IMAGE = "registry.console.elementai.com/snow.rg_climate_benchmark/base:main"
TOOLKIT_IMAGE = (
    " registry.console.elementai.com/snow.interactive_toolkit/climate_benchmark:main"
)
TOOLKIT_DATA = "snow.rg_climate_benchmark.data"
TOOLKIT_CODE = "snow.rg_climate_benchmark.code"
TOOLKIT_RESULTS = "snow.rg_climate_benchmark.results"
# code version so you can continue working on code and not interrupt existing jobs
TOOLKIT_CODE_VERSION = f"{TOOLKIT_USER}_{GIT_BRANCH}"


# bootstrap command in order to have the toolkit environment ready to run the script
# we need to copy the code from the data mount to the /tmp folder, so that we won't break these running jobs
# by changing the code in the data mount (continue working on other stuff)
# TOOLKIT_BOOTSTRAP_CMD = (
#     'cp -r /mnt/code/* /tmp && '
#     'cd /tmp && '
#     # 'export PATH=$PATH:/tmp/.local/bin && '
#     'echo "Bootstrap completed. Starting execution...\\n\\n\\n"'
# )

TOOLKIT_BOOTSTRAP_CMD = (
    "cp -r /mnt/code/* /tmp && "
    'echo "Bootstrap completed. Starting execution...\\n\\n\\n"'
)
TOOLKIT_ENVS = (
    "CC_BENCHMARK_SOURCE_DATASETS=/mnt/data/cc_benchmark/source",
    "CC_BENCHMARK_CONVERTED_DATASETS=/mnt/data/cc_benchmark/converted",
)

# Computational requirements
TOOLKIT_CPU = 8
TOOLKIT_GPU = 4
TOOLKIT_MEM = 32


def _load_envs():
    """Load environment variables that must be set in toolkit"""
    return [env.strip() for env in open(".envs", "r")]


def toolkit_job(script_path: Path, env_vars=()):
    """Launch a job on toolkit along with a specific script (assumed runnable with sh).

    Args:
        script_path: path to .sh script to exectute on toolkit
    """
    print("Launching job...")
    job_name = script_path.parent.name.lower()
    for char in (".", "="):  # TODO replace all non alpha numeric chars by '_'.
        job_name = job_name.replace(char, "_")

    # General job config
    cmd = f"eai job new -i {TOOLKIT_IMAGE} --restartable".split(" ")

    # Computational requirements
    cmd += ["--cpu", str(TOOLKIT_CPU)]
    cmd += ["--gpu", str(TOOLKIT_GPU)]
    cmd += ["--mem", str(TOOLKIT_MEM)]
    cmd += ["--account", str(TOOLKIT_USER_ACCOUNT)]

    # cmd += ["--gpu-model-filter", "!A100"]
    cmd += ["--gpu-model-filter", "v100-sxm2-32gb"]

    # Mount data objects
    cmd += ["--data", f"{TOOLKIT_DATA}:/mnt/data"]
    cmd += ["--data", f"{TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION}:/mnt/code"]
    cmd += ["--data", f"{TOOLKIT_RESULTS}:/mnt/results"]
    cmd += ["--data", "none.n_lehmann.home:/home/toolkit"]
    cmd += ["--env", "HOME=/home/toolkit"]

    # Set all environment variables
    for e in TOOLKIT_ENVS + env_vars:
        cmd += ["--env", e]

    # this is the command that will be executed on the interactive node, so
    # the bash script you specified will be executed here
    # so from the tmp directory we move to the copied mounted code and execute the script
    cmd += [
        "--",
        f"sh -c '{TOOLKIT_BOOTSTRAP_CMD} && cd /mnt/code/ && pwd && ls && sh {str(script_path)}'",
        # f"sh -c '{TOOLKIT_BOOTSTRAP_CMD}'",
    ]  # TODO do we have to put absolute path? Could we use relative path of script?

    # cmd += [
    #     "--",
    #     # f"sh -c '{TOOLKIT_BOOTSTRAP_CMD} && cd /mnt/code/ && pwd && ls'",
    #     f"sh -c '{TOOLKIT_BOOTSTRAP_CMD} && while true; do sleep 3600; done'",
    #     # f"sh -c '{TOOLKIT_BOOTSTRAP_CMD}'",
    # ]  #

    # cmd += ["--", "bash" , "-c", "'while true; do sleep 3600; done'"]

    # Launch the job
    command = " ".join(cmd)
    print(command)
    _run_shell_cmd(command)
    print("Launched.")


def push_code(dir):
    """Push the local code to the cluster"""
    print("Pushing code...")
    _run_shell_cmd(
        f"eai data branch add {TOOLKIT_CODE}@empty {TOOLKIT_CODE_VERSION}",
        hide_stderr=True,
    )
    _run_shell_cmd(
        f"eai data content rm {TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION} .",
        hide_stderr=False,
    )

    rsync_cmd = f"rsync -a '{dir}' /tmp/project_folder --delete --exclude-from='{dir}/.eaiignore'"
    _run_shell_cmd(rsync_cmd)
    # rsync mmcv as well
    # rsync_cmd = f"rsync -a /mnt/rg_climate_benchmark/code/nils/mmcv /tmp/project_folder --delete"
    # _run_shell_cmd(rsync_cmd)
    _run_shell_cmd(
        f"eai data push {TOOLKIT_CODE}@{TOOLKIT_CODE_VERSION} /tmp/project_folder:/"
    )
    _run_shell_cmd("rm -rf /tmp/project_folder")
    print("Finished pushing code.")


def start(script_path):
    push_code("./")
    toolkit_job(script_path=Path(script_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push local code to the cluster.")
    parser.add_argument("--path", type=str, help="The path to the directory to push.")
    args = parser.parse_args()
    start(args.path)
