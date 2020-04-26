import os, sys, stat, subprocess
from argparse import ArgumentParser

os.environ['NCCL_LL_THRESHOLD'] = '0'


def parse_args():
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--world_size", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--init_method", default="tcp://127.0.0.1:29000", type=str,
                        help="Init method of distributed system.")
    parser.add_argument("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass "
                             "'local rank'. For legacy reasons, the default value is False. "
                             "If set to True, the script will not pass "
                             "--local_rank as argument, and will instead set LOCAL_RANK.")

    # positional
    parser.add_argument("--training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    return parser.parse_known_args()


def main():
    args, script_args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.world_size

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    assert args.init_method.startswith("tcp://"), "init_method should start with \"tcp://\"."
    master_addr, master_port = args.init_method[6:].split(":")
    current_env["MASTER_ADDR"] = master_addr
    current_env["MASTER_PORT"] = str(master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # For some store true args.
        new_script_args = []
        for script_arg in script_args:
            if script_arg.endswith("=True"):
                new_script_args.append(script_arg[:-5])
            elif script_arg.endswith("=False"):
                pass
            else:
                new_script_args.append(script_arg)
        script_args = new_script_args

        # spawn the processes
        if args.use_env:
            cmd = [sys.executable, "-u",
                   args.training_script] + script_args
        else:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank)] + script_args
        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)


if __name__ == "__main__":
    main()
