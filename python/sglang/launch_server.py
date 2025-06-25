"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import PortArgs, prepare_server_args
from sglang.srt.utils import kill_process_tree, configure_logger

from p2p.server import launch_p2p_server

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    port_args = PortArgs.init_new(server_args)
    configure_logger(server_args)

    launch_p2p_server(server_args, port_args=port_args)
    
    try:
        if server_args.pp_start_layer == 0:
            launch_server(server_args, port_args=port_args)
        else:
            # TODO: only launch the scheduler process
            launch_server(server_args, port_args=port_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
