.PHONY: help build-gpu build-cpu run-gpu run-cpu stop clean

# 加载环境变量
include .env
export

# 默认配置
PROJECT_NAME=deepchem-api
CONTAINER_NAME=$(PROJECT_NAME)
IMAGE_NAME=$(PROJECT_NAME)

help:
	@echo "使用说明："
	@echo "make build-gpu     - 构建GPU版本Docker镜像"
	@echo "make build-cpu     - 构建CPU版本Docker镜像"
	@echo "make run-gpu       - 运行GPU版本容器"
	@echo "make run-cpu       - 运行CPU版本容器"
	@echo "make stop          - 停止运行中的容器"
	@echo "make clean         - 清理所有容器和镜像"

build-gpu:
	docker build -t $(IMAGE_NAME):gpu-latest -f docker/Dockerfile.gpu --build-arg PORT=$(PORT) .

build-cpu:
	docker build -t $(IMAGE_NAME):cpu-latest -f docker/Dockerfile.cpu --build-arg PORT=$(PORT) .

run-gpu:
	docker run --rm -it --name $(CONTAINER_NAME)-gpu \
		--gpus all \
		-p $(PORT):$(PORT) \
		-e PORT=$(PORT) \
		-e USE_GPU=true \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(IMAGE_NAME):gpu-latest

run-cpu:
	docker run --rm -it --name $(CONTAINER_NAME)-cpu \
		-p $(PORT):$(PORT) \
		$(IMAGE_NAME):cpu-latest

stop:
	-docker stop $(CONTAINER_NAME)-gpu $(CONTAINER_NAME)-cpu

clean: stop
	-docker rmi $(IMAGE_NAME):gpu-latest $(IMAGE_NAME):cpu-latest

# 默认目标
default: help 