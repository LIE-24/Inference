import io
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union

from p2p.proto import forward_pb2
from sglang.srt.managers.schedule_batch import Req, SamplingParams
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def tensor_to_proto(tensor: torch.Tensor) -> forward_pb2.Tensor:
    """Convert PyTorch tensor to protobuf Tensor message."""
    
    proto = forward_pb2.Tensor()
    
    # Convert tensor to CPU
    if tensor.device.type != 'cpu':
        cpu_tensor = tensor.cpu()
    else:
        cpu_tensor = tensor
    
    # Set tensor properties
    proto.dtype = str(tensor.dtype)
    proto.size.extend(list(tensor.shape))
    
    # Store buffer
    buffer = io.BytesIO()
    torch.save(cpu_tensor, buffer)
    proto.buffer = buffer.getvalue()
    
    return proto


def proto_to_tensor(proto: forward_pb2.Tensor, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    """Convert protobuf Tensor message to PyTorch tensor."""
    
    # Load tensor from buffer
    buffer = io.BytesIO(proto.buffer)
    tensor = torch.load(buffer)
    
    # Move to specified device if provided
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def sampling_params_to_proto(params: SamplingParams) -> forward_pb2.SamplingParams:
    """Convert SamplingParams to protobuf message."""
    proto = forward_pb2.SamplingParams()

    proto.max_new_tokens = params.max_new_tokens
    proto.min_new_tokens = params.min_new_tokens
    proto.temperature = params.temperature
    proto.top_p = params.top_p
    proto.top_k = params.top_k
    proto.min_p = params.min_p
    if params.stop_token_ids is not None:
        proto.stop_token_ids.extend(params.stop_token_ids)
    proto.ignore_eos = params.ignore_eos
    proto.stop_strs.extend(params.stop_strs)
    proto.repetition_penalty = params.repetition_penalty
    proto.presence_penalty = params.presence_penalty
    proto.frequency_penalty = params.frequency_penalty
    if params.json_schema is not None:
        proto.json_schema = params.json_schema

    return proto


def proto_to_sampling_params(proto: forward_pb2.SamplingParams) -> SamplingParams:
    """Convert protobuf message to SamplingParams."""
    
    sampling_params = SamplingParams(
        max_new_tokens=proto.max_new_tokens,
        min_new_tokens=proto.min_new_tokens,
        temperature=proto.temperature,
        top_p=proto.top_p,
        min_p=proto.min_p,
        top_k=proto.top_k,
        stop=list(proto.stop_strs),
        stop_token_ids=list(proto.stop_token_ids),
        ignore_eos=proto.ignore_eos,
        repetition_penalty=proto.repetition_penalty,
        presence_penalty=proto.presence_penalty,
        frequency_penalty=proto.frequency_penalty,
        json_schema=proto.json_schema,
    )
    return sampling_params


def req_to_proto(req: Req, full_req: bool = False) -> forward_pb2.Req:
    """Convert Req to protobuf message."""
    
    proto = forward_pb2.Req()
    proto.rid = req.rid
    proto.output_length = len(req.output_ids)
    if req.routing_table is not None:
        proto.routing_table.extend(req.routing_table)
    
    # only send full req when prefill
    if full_req:
        proto.input_ids.extend(req.origin_input_ids)
        proto.sampling_params.CopyFrom(sampling_params_to_proto(req.sampling_params))

    return proto


def proto_to_req(proto: forward_pb2.Req) -> Req:
    """Convert protobuf message to Req."""
    
    req = Req(
        rid=proto.rid,
        origin_input_text="",
        origin_input_ids=list(proto.input_ids),
        sampling_params=proto_to_sampling_params(proto.sampling_params) if proto.HasField("sampling_params") else SamplingParams(),
        routing_table=list(proto.routing_table),
    )

    return req


def ppproxy_tensors_to_proto(proxy_tensors: PPProxyTensors) -> forward_pb2.PPProxyTensorsMessage:
    """Convert PPProxyTensors to protobuf message."""
    
    proto = forward_pb2.PPProxyTensorsMessage()
    
    for name, tensor in proxy_tensors.tensors.items():
        named_tensor = proto.tensors.add()
        named_tensor.name = name
        named_tensor.tensor.CopyFrom(tensor_to_proto(tensor))
    
    return proto


def proto_to_ppproxy_tensors(proto: forward_pb2.PPProxyTensorsMessage, device=None) -> Optional[PPProxyTensors]:
    """Convert protobuf message to PPProxyTensors."""
    if proto is None:
        return None
    
    tensor_dict = {}
    for named_tensor in proto.tensors:
        tensor_dict[named_tensor.name] = proto_to_tensor(named_tensor.tensor, device)
    
    return PPProxyTensors(tensor_dict)


def forward_request_to_proto(
        forward_mode: forward_pb2.ForwardMode,
        reqs: List[Req],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        next_token_ids: Optional[List[int]] = None,
        full_req: bool = False
    ) -> forward_pb2.ForwardRequest:
    """Convert forward request data to protobuf message."""
    proto = forward_pb2.ForwardRequest()
    proto.forward_mode = forward_mode

    if reqs is not None:
        for req in reqs:
            req_proto = req_to_proto(req, full_req)
            proto.reqs.append(req_proto)
    
    if pp_proxy_tensors is not None:
        proto.pp_proxy_tensors.CopyFrom(ppproxy_tensors_to_proto(pp_proxy_tensors))
    
    if next_token_ids is not None:
        proto.next_token_ids.extend(next_token_ids)
    
    return proto


def proto_to_forward_request(proto: forward_pb2.ForwardRequest, device=None) -> Tuple[forward_pb2.ForwardMode, List[Req], Optional[PPProxyTensors], Optional[List[int]]]:
    """Convert protobuf message to forward request data."""    
    forward_mode = proto.forward_mode

    # Convert reqs
    reqs = [proto_to_req(req_proto) for req_proto in proto.reqs]
    
    # Convert proxy tensors if available
    pp_proxy_tensors = None
    if proto.HasField("pp_proxy_tensors"):
        pp_proxy_tensors = proto_to_ppproxy_tensors(proto.pp_proxy_tensors, device)

    next_token_ids = torch.tensor(list(proto.next_token_ids), dtype=torch.int64, device=device) if proto.next_token_ids else None
    
    return forward_mode, reqs, pp_proxy_tensors, next_token_ids


def forward_response_to_proto(uid: str, pp_proxy_tensors: Optional[PPProxyTensors] = None, next_token_ids: Optional[List[int]] = None) -> forward_pb2.ForwardResponse:
    """Convert forward response data to protobuf message."""
    proto = forward_pb2.ForwardResponse()
        
    if pp_proxy_tensors is not None:
        proto.pp_proxy_tensors.CopyFrom(ppproxy_tensors_to_proto(pp_proxy_tensors))
    
    if next_token_ids is not None:
        proto.next_token_ids.extend(next_token_ids)
    
    return proto


def proto_to_forward_response(proto: forward_pb2.ForwardResponse, device=None) -> Tuple[str, Optional[PPProxyTensors], Optional[List[int]]]:
    """Convert protobuf message to forward response data."""
    uid = proto.uid
    
    proxy_tensors = None
    next_token_ids = None
    if proto.HasField("pp_proxy_tensors"):
        proxy_tensors = proto_to_ppproxy_tensors(proto.pp_proxy_tensors, device)
    if proto.next_token_ids:
        next_token_ids = torch.tensor(list(proto.next_token_ids), dtype=torch.int32, device=device)
    
    return uid, proxy_tensors, next_token_ids

