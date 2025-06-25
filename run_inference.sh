#!/bin/bash

# 推理服务运行脚本
# 使用方法: ./run_inference.sh [last|head] [peer_info]

ECR_REPOSITORY="public.ecr.aws/s9c5a2r1/inference_parallax"
IMAGE_TAG="latest"

if [ "$1" = "last" ]; then
    echo "启动最后一个节点 (last node)..."
    docker run --gpus all -p 3000:3000 -p 5000:5000 -it --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        $ECR_REPOSITORY:$IMAGE_TAG \
        python3 python/sglang/launch_server.py \
        --disable-overlap-schedule \
        --disable-radix-cache \
        --model-path Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \
        --pp-start-layer 40 \
        --pp-end-layer 80 \
        --dtype bfloat16 \
        --dht-port 5000 \
        --host 0.0.0.0 \
        --port 3000 \
        --attention-backend flashinfer

elif [ "$1" = "head" ]; then
    if [ -z "$2" ]; then
        echo "错误: head节点需要提供peer信息"
        echo "使用方法: ./run_inference.sh head '/ip4/{peers_public_ip}/tcp/{peers_public_port_to_5000}/p2p/{peer_id}'"
        exit 1
    fi
    
    echo "启动头节点 (head node)..."
    docker run --gpus all -p 3000:3000 -p 5000:5000 -it --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -e DISPLAY=$DISPLAY \
        $ECR_REPOSITORY:$IMAGE_TAG \
        python3 python/sglang/launch_server.py \
        --disable-overlap-schedule \
        --disable-radix-cache \
        --model-path Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \
        --pp-start-layer 0 \
        --pp-end-layer 40 \
        --dtype bfloat16 \
        --dht-port 5000 \
        --host 0.0.0.0 \
        --port 3000 \
        --attention-backend flashinfer \
        --initial-peers "$2"

else
    echo "使用方法:"
    echo "  启动last节点: ./run_inference.sh last"
    echo "  启动head节点: ./run_inference.sh head '/ip4/{peers_public_ip}/tcp/{peers_public_port_to_5000}/p2p/{peer_id}'"
    echo ""
    echo "示例:"
    echo "  ./run_inference.sh last"
    echo "  ./run_inference.sh head '/ip4/192.168.1.100/tcp/5000/p2p/12D3KooWABC123'"
fi 