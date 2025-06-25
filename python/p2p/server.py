import multiprocessing as mp
import threading
import time
import enum
import dataclasses
import logging
import httpx
from typing import List, Optional
import builtins

# set hivemind grpc message size to 1GB
_original_import = builtins.__import__
def patched_import(name, *args, **kwargs):
    module = _original_import(name, *args, **kwargs)
    if name == 'hivemind.p2p.p2p_daemon_bindings.control':
        if hasattr(module, 'DEFAULT_MAX_MSG_SIZE'):
            module.DEFAULT_MAX_MSG_SIZE = 2**30
    return module

builtins.__import__ = patched_import

from hivemind import DHT, P2PContext
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import PeerID

import zmq
from p2p.proto import forward_pb2
from p2p.utils import AsyncWorker

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_zmq_socket
import dijkstar

logger = logging.getLogger(__name__)

# Define server state enum
class ServerState(enum.Enum):
    JOINING = 'joining'
    INITIALIZING = 'initializing'
    READY = 'ready'
    OFFLINE = 'offline'
    ERROR = 'error'

# Define server info data class
@dataclasses.dataclass
class ServerInfo:
    state: ServerState
    throughput: Optional[float] = None
    max_batch_size: Optional[int] = None
    max_sequence_len: Optional[int] = None
    error_message: Optional[str] = None


def send_notify(notify_url, block_start_index, block_end_index, request, status):
    payload = [{
        "session_id": req.rid,
        # the output length of started is from previouse rank, if this is head rank we need add 1
        "step_id": req.output_length + (block_start_index == 0 and status == "started"),
        "block_idx": block_start_index,
        "total_blocks": block_end_index - block_start_index,
        "status": status,
    } for req in request.reqs]
    
    logger.info(f"Send {status} notification, batch size: {len(payload)}")
    
    if notify_url is not None:
        async def send_async(notify_url, payload):
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(notify_url, json=payload)
                except Exception as e:
                    pass

        if not hasattr(send_notify, "async_worker"):
            send_notify.async_worker = AsyncWorker()
        send_notify.async_worker.run_coroutine(send_async(notify_url, payload), return_future=True)

class TransformerConnectionHandler(ConnectionHandler):
    """
    Handles RPC requests from clients, forwarding them to the appropriate TransformerBackend.
    Inherits from hivemind's ConnectionHandler.
    """

    def __init__(
        self,
        dht: DHT,
        port_args: PortArgs,
        block_start_index: int,
        block_end_index: int,
        notify_url: str = None,
    ):
        # Initialize the base class
        super().__init__(dht, {})
        self.port_args = port_args
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.notify_url = notify_url
        self.recv_from_peer = None

    """
    receive the next_token_ids and pp_proxy_tensors from the previous server
    """

    async def rpc_pp_forward(self, request: forward_pb2.ForwardRequest, context: P2PContext) -> forward_pb2.ForwardResponse:
        if self.recv_from_peer is None:
            self.recv_from_peer = get_zmq_socket(
                zmq.Context(
                    2), zmq.PUSH, self.port_args.recv_peer_ipc_name, True
            )

        """Handle forward pass request with explicit proxy tensors support"""
        try:
            send_notify(self.notify_url, self.block_start_index, self.block_end_index, request, "started")

            self.recv_from_peer.send(request.SerializeToString())
        except Exception as e:
            logger.exception(f"Error in rpc_pp_forward: {e}")
        return forward_pb2.ForwardResponse()


# Main server class
class GradientServer:
    def __init__(
        self,
        port_args: PortArgs,
        initial_peers: List[str] = [],
        block_start_index: int = 0,
        block_end_index: int = 1,
        hidden_layers: int = 128,
        dht_prefix: str = "gradient",
        notify_url: str = None,
        **kwargs
    ):
        self.port_args = port_args
        self.initial_peers = initial_peers
        self.block_start_index = block_start_index
        self.block_end_index = block_end_index
        self.hidden_layers = hidden_layers
        self.dht_prefix = dht_prefix
        self.notify_url = notify_url
        self.kwargs = kwargs

        self.node_id = f"{dht_prefix}_{block_start_index}"
        self.dht = None
        self.routing_table = None
        self.routing_table_update_interval = 10
        self.server_info = ServerInfo(state=ServerState.JOINING)
        self.stop_event = threading.Event()

    def run(self):
        """Initialize DHT network connection"""
        if len(self.initial_peers) > 0:
            logger.info(
                f"Connecting to DHT network, initial peers: {self.initial_peers}")
        else:
            logger.info("No initial peers provided, start a new DHT network")

        self.dht = DHT(
            initial_peers=self.initial_peers,
            start=True,
            client_mode=False,
            **self.kwargs
        )
        visible_maddrs = [str(addr) for addr in self.dht.get_visible_maddrs()]
        logger.info(f"Server visible addresses: {visible_maddrs}")

        self.p2p = RemoteExpertWorker.run_coroutine(self.dht.replicate_p2p())

        self.connection_handler = TransformerConnectionHandler(
            dht=self.dht,
            port_args=self.port_args,
            block_start_index=self.block_start_index,
            block_end_index=self.block_end_index,
            notify_url=self.notify_url
        )
        self.connection_handler.run_in_background()  # sub-process
        self.start_node_announcer()  # thread
        self.start_routing_table_updater()  # thread
        self.start_node_sender()  # main loop

    def find_servers(self):
        """Find available servers in the DHT network"""
        # Find all announced blocks
        server_blocks = []
        for block_index in range(self.hidden_layers):
            block_announced_key = f"{self.dht_prefix}_{block_index}"
            block_servers = self.dht.get(block_announced_key)
            if block_servers is None:
                continue
            for peer_id, value in block_servers.value.items():
                server_blocks.append({
                    "peer_id": peer_id,
                    "block_start_index": block_index,
                    "block_end_index": value.value['block_end_index']
                })

        if len(server_blocks) == 1:
            # TODO: restart dht
            logger.warning(f"Only one server found, this is not normal, need to restart dht")

        return server_blocks

    def start_routing_table_updater(self):
        def _updater_thread():
            while True:
                try:
                    graph = dijkstar.Graph()
                    servers = self.find_servers()
                    for server in servers:
                        start_index = server['block_start_index']
                        end_index = server['block_end_index']
                        peer_id = server['peer_id']

                        # TODO: set weight to the distance between start_index and end_index, eg. network latency
                        graph.add_edge(start_index, end_index, (1, peer_id))
                    try:
                        path = dijkstar.find_path(
                            graph, self.block_end_index, self.hidden_layers, cost_func=lambda u, v, e, prev_path: e[0])
                        routing_table = [self.dht.peer_id.to_base58()] + [edge[1] for edge in path.edges]
                        if self.routing_table != routing_table:
                            self.routing_table = routing_table
                            logger.info(f"Set routing table: {routing_table}")
                            # exit when routing table is set as our node is static right now.
                            # TODO: remove this when we have dynamic node
                            break
                    except dijkstar.NoPathError:
                        self.routing_table = None
                        logger.warning(f"No path found from 0 to {self.hidden_layers}, find servers {servers}")
                except Exception as e:
                    logger.exception(f"Error in routing table updater: {e}")

                time.sleep(self.routing_table_update_interval)

        if self.block_start_index == 0:
            self.updater = threading.Thread(
                target=_updater_thread, daemon=True)
            self.updater.start()

    def start_node_sender(self):
        send_to_peer = get_zmq_socket(
            zmq.Context(2), zmq.PULL, self.port_args.send_peer_ipc_name, True
        )

        while True:
            try:
                if self.block_start_index == 0 and self.routing_table is None:
                    logger.info(
                        "Routing table is not ready in head rank, waiting for it to be set")
                    time.sleep(self.routing_table_update_interval)
                    continue

                forward_request_serialized = send_to_peer.recv()
                forward_request = forward_pb2.ForwardRequest()
                forward_request.ParseFromString(forward_request_serialized)
                if len(forward_request.reqs) == 0:
                    raise RuntimeError("No requests in the forward request")

                # suppose all requests have same routing table
                routing_table = None
                for req in forward_request.reqs:
                    if len(req.routing_table) == 0:
                        assert self.block_start_index == 0, "Request routing table is not set for non-head rank"
                        req.routing_table.extend(self.routing_table)
                    # Assume all requests have same routing table
                    routing_table = list(req.routing_table)

                try:
                    self_index = routing_table.index(self.dht.peer_id.to_base58())
                except ValueError:
                    raise RuntimeError("Can not find self in the routing table")

                next_peer_id = routing_table[(self_index + 1) % len(routing_table)]
                stub = TransformerConnectionHandler.get_stub(self.p2p, PeerID.from_base58(next_peer_id))
                start = time.time()
                response = RemoteExpertWorker.run_coroutine(stub.rpc_pp_forward(forward_request), return_future=True)
                send_notify(self.notify_url, self.block_start_index, self.block_end_index, forward_request, "completed")

                response = response.result()
                logger.info(
                    f"Forwarding data to {next_peer_id}, total size: {len(forward_request_serialized) / (1024 * 1024):.3f} MB, cost time: {(time.time() - start) * 1000:.3f} ms, speed: {len(forward_request_serialized) / (time.time() - start) / (1024 * 1024):.3f} MB/s")

            except Exception as e:
                logger.exception(f"Error in handle_request: {e}")
                time.sleep(1)

    def start_node_announcer(self):
        """Start a thread that regularly announces this module's presence on DHT"""
        def _announcer_thread():
            try:
                while not self.stop_event.is_set():
                    # Announce the range ID
                    try:
                        self.dht.store(
                            key=self.node_id,
                            subkey=self.dht.peer_id.to_string(),
                            value={
                                "block_end_index": self.block_end_index,
                            },
                            expiration_time=time.time() + 60,  # Valid for 60 seconds
                        )
                        logger.info(f"Announced {self.node_id} on DHT")
                    except Exception as e:
                        logger.warning(
                            f"Failed to announce {self.node_id}: {e}")

                    # Wait and repeat
                    time.sleep(30)
            except Exception as e:
                logger.exception(f"Module announcer thread error: {e}")

        # Start announcer thread
        self.announcer = threading.Thread(
            target=_announcer_thread, daemon=True)
        self.announcer.start()


def launch_p2p_server(server_args: ServerArgs, port_args: PortArgs):
    host_maddrs = server_args.host_maddrs
    dht_port = server_args.dht_port
    if dht_port is not None:
        assert host_maddrs is None, "You can't use --port and --host_maddrs at the same time"
    else:
        dht_port = 0
    if host_maddrs is None:
        host_maddrs = [
            f"/ip4/0.0.0.0/tcp/{dht_port}", f"/ip6/::/tcp/{dht_port}"]

    announce_maddrs = server_args.announce_maddrs
    public_ip = server_args.public_ip
    if public_ip is not None:
        assert announce_maddrs is None, "You can't use --public_ip and --announce_maddrs at the same time"
        assert dht_port != 0, "Please specify a fixed non-zero --port when you use --public_ip (e.g., --port 31337)"
        announce_maddrs = [f"/ip4/{public_ip}/tcp/{dht_port}"]

    # Extra kwargs for server
    kwargs = {
        "host_maddrs": host_maddrs,
        "announce_maddrs": announce_maddrs
    }

    model_config = ModelConfig.from_server_args(server_args)

    # Run the server in a separate thread to keep the main thread free for event loop
    server = GradientServer(
        port_args=port_args,
        initial_peers=server_args.initial_peers,
        block_start_index=server_args.pp_start_layer,
        block_end_index=server_args.pp_end_layer,
        hidden_layers=model_config.num_hidden_layers,
        dht_prefix=server_args.dht_prefix,
        notify_url=server_args.notify_url,
        **kwargs
    )

    # Start the server
    mp.Process(target=server.run).start()
