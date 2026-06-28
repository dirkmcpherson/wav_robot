import contextlib
import gc
import os
import sys

sys.path.append(
    os.path.join(os.getcwd(), "wav/diffusion")
)  # For diffusion4robotics imports

from wav.runtime.config_loader import build_config_from_cli, finalize_runtime_config
from wav.train_loop import run_training_pipeline

# Headless rendering. EGL is preferred but its context teardown fails on some drivers
# (this box); allow override via MUJOCO_GL (we use osmesa locally, which renders cleanly).
_gl = os.environ.get("MUJOCO_GL", "egl")
os.environ["MUJOCO_GL"] = _gl
os.environ["PYOPENGL_PLATFORM"] = _gl


if __name__ == "__main__":
    final_config = build_config_from_cli()
    final_config, _suite, _task = finalize_runtime_config(final_config)
    run_training_pipeline(final_config)
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        gc.collect()  # Force garbage collection to run
