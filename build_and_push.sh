#!/bin/bash

# 设置变量
# 请设置您的AWS凭证（不要提交到版本控制中）
# AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
# AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
AWS_DEFAULT_REGION="us-east-1"
ECR_REPOSITORY="public.ecr.aws/s9c5a2r1/inference_parallax"
IMAGE_TAG="latest"

echo "设置AWS凭证..."
# 使用环境变量或在此设置您的AWS凭证
# export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
# export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

echo "安装AWS CLI（如果未安装）..."
if ! command -v aws &> /dev/null; then
    echo "请先安装AWS CLI："
    echo "MacOS: brew install awscli"
    echo "或者手动安装: https://aws.amazon.com/cli/"
    exit 1
fi

echo "登录到AWS ECR..."
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws

echo "构建Docker镜像..."
docker build --platform=linux/amd64 -f Dockerfile.inference -t $ECR_REPOSITORY:$IMAGE_TAG .

echo "推送镜像到ECR..."
docker push $ECR_REPOSITORY:$IMAGE_TAG

echo "完成！镜像已推送到: $ECR_REPOSITORY:$IMAGE_TAG"

echo ""
echo "==================== 使用说明 ===================="
echo "1. 在第一个节点（last node）运行："
echo "docker run --gpus all -p 3000:3000 -p 5000:5000 $ECR_REPOSITORY:$IMAGE_TAG \\"
echo "  python3 python/sglang/launch_server.py \\"
echo "  --disable-overlap-schedule \\"
echo "  --disable-radix-cache \\"
echo "  --model-path Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \\"
echo "  --pp-start-layer 40 \\"
echo "  --pp-end-layer 80 \\"
echo "  --dtype bfloat16 \\"
echo "  --dht-port 5000 \\"
echo "  --host 0.0.0.0 \\"
echo "  --port 3000 \\"
echo "  --attention-backend flashinfer"
echo ""
echo "2. 在第二个节点（head node）运行："
echo "docker run --gpus all -p 3000:3000 -p 5000:5000 $ECR_REPOSITORY:$IMAGE_TAG \\"
echo "  python3 python/sglang/launch_server.py \\"
echo "  --disable-overlap-schedule \\"
echo "  --disable-radix-cache \\"  
echo "  --model-path Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \\"
echo "  --pp-start-layer 0 \\"
echo "  --pp-end-layer 40 \\"
echo "  --dtype bfloat16 \\"
echo "  --dht-port 5000 \\"
echo "  --host 0.0.0.0 \\"
echo "  --port 3000 \\"
echo "  --attention-backend flashinfer \\"
echo "  --initial-peers /ip4/{peers_public_ip}/tcp/{peers_public_port_to_5000}/p2p/{peer_id}"
echo ""
echo "注意：请将{peers_public_ip}、{peers_public_port_to_5000}和{peer_id}替换为实际的值"
echo "==================================================" 