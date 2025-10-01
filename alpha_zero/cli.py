import argparse
import shutil
import subprocess
import sys

DEFAULT_HOST_MODELS = r"C:\alphazero\models"
DEFAULT_HOST_TREES = r"C:\alphazero\trees"
IMAGE_NAME = "alpha_zero"


def run(cmd, interactive=False):
    print(">>>", " ".join(cmd))
    try:
        if interactive:
            return subprocess.run(cmd, check=True)
        else:
            return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Command failed:", e, file=sys.stderr)
        if not interactive:
            print("stdout:", e.stdout.decode(errors="ignore"))
            print("stderr:", e.stderr.decode(errors="ignore"))
        sys.exit(e.returncode)


def build_image():
    if shutil.which("docker") is None:
        print("docker not found in PATH.", file=sys.stderr)
        sys.exit(2)
    cmd = ["docker", "build", "-t", IMAGE_NAME, "."]
    run(cmd,interactive=True)


def train(game, host_models=DEFAULT_HOST_MODELS):
    if game == "ttt":
        script = "python3 examples/train_tic_tac_toe.py"
    elif game == "ckr":
        script = "python3 examples/train_checkers.py"
    else:
        raise ValueError("unknown game: " + str(game))

    mount = f"{host_models}:/app/models"
    cmd = ["docker", "run", "--gpus", "all", "-v", mount, IMAGE_NAME] + script.split()
    run(cmd,interactive=True)


def play(game, host_models=DEFAULT_HOST_MODELS):
    if game == "ttt":
        script = "python3 examples/play_tic_tac_toe.py"
    elif game == "ckr":
        script = "python3 examples/play_checkers.py"
    else:
        raise ValueError("unknown game: " + str(game))

    mount = f"{host_models}:/app/models"
    cmd = ["docker", "run", "--gpus", "all", "-it", "-v", mount, IMAGE_NAME] + script.split()
    run(cmd, interactive=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="alpha_zero", description="AlphaZero docker automation CLI")
    # make train/play optional so --build can be used alone
    p.add_argument("--train", choices=["ttt", "ckr"], help="train : ttt or ckr")
    p.add_argument("--play", choices=["ttt", "ckr"], help="play  : ttt or ckr")
    p.add_argument("--build", action="store_true", help="build the docker image (if provided) - can be used alone")
    p.add_argument("--host-models", default=DEFAULT_HOST_MODELS,
                   help=f'Host models dir (default: {DEFAULT_HOST_MODELS})')
    p.add_argument("--host-trees", default=DEFAULT_HOST_TREES,
                   help=f'Host trees dir (default: {DEFAULT_HOST_TREES}) - currently not used by default commands')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # If user asked only for build, run it and exit.
    if args.build and not (args.train or args.play):
        print("Building Docker image and exiting (--build provided without --train/--play)...")
        build_image()
        print("Build complete.")
        return

    # If build and train/play provided, build first then continue
    if args.build:
        print("Building Docker image before running requested action...")
        build_image()

    # Run training or playing as requested
    if args.train and args.play:
        print("Please specify only one of --train or --play.", file=sys.stderr)
        sys.exit(2)

    if args.train:
        print(f"Starting training for {args.train} (models -> {args.host_models})")
        train(args.train, host_models=args.host_models)
    elif args.play:
        print(f"Starting play for {args.play} (models -> {args.host_models})")
        play(args.play, host_models=args.host_models)
    else:
        # Nothing meaningful requested
        print("No action requested. Use --build, --train <ttt|ckr>, or --play <ttt|ckr>.")
        print("For example:")
        print("  poetry run alpha_zero --build               # build the image only")
        print("  poetry run alpha_zero --train ttt           # run tic-tac-toe training")
        print("  poetry run alpha_zero --build --train ckr   # build then train checkers")
        sys.exit(0)
