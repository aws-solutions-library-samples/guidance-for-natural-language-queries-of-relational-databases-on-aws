#!/bin/sh

# Builds three Docker images for Natural Language Query (NLQ) demo using Amazon RDS for PostgreSQL:
# 1/ Amazon SageMaker JumpStart Foundation Models
# 2/ Amazon Bedrock
# 3/ OpenAI's LLM models via their API
# Author: Gary A. Stafford
# Date: 2023-07-12
# run: chmod a+rx build.sh
# sh ./build.sh

# Value located in the output from the nlq-genai-infra CloudFormation template
# e.g. 111222333444.dkr.ecr.us-east-1.amazonaws.com/nlq-genai
ECS_REPOSITORY="<you_ecr_repository>"

aws ecr get-login-password --region us-east-1 |
	docker login --username AWS --password-stdin $ECS_REPOSITORY

# Option 1: SageMaker JumpStart FM Endpoint
TAG="1.0.0-sm"
docker build -f Dockerfile_SageMaker -t $ECS_REPOSITORY:$TAG .
docker push $ECS_REPOSITORY:$TAG

# Option 2: Amazon Bedrock
TAG="1.0.1-bedrock"
docker build -f Dockerfile_Bedrock -t $ECS_REPOSITORY:$TAG .
docker push $ECS_REPOSITORY:$TAG

# Option 3: OpenAI API
TAG="1.0.0-oai"
docker build -f Dockerfile_OpenAI -t $ECS_REPOSITORY:$TAG .
docker push $ECS_REPOSITORY:$TAG

docker image ls
