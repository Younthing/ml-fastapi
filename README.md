# Docker 配置说明

本项目提供了多种Docker配置文件，支持不同的运行环境：

- `Dockerfile` - 标准版本，支持GPU
- `Dockerfile.gpu` - 专门为GPU优化的版本
- `Dockerfile.cpu` - CPU版本，不需要GPU

## 环境变量

所有Docker配置都支持以下环境变量：

- `CONTAINER_PORT` - 容器内服务端口 (默认: 8000)
- `HOST` - 服务监听地址 (默认: 0.0.0.0)
- `PORT` - 宿主机映射端口 (默认: 8000)
- `WORKERS` - 工作进程数 (默认: 4)
- `LOG_LEVEL` - 日志级别 (默认: info)
- `USE_GPU` - 是否使用GPU (默认: true/false 取决于镜像)
- `CUDA_VISIBLE_DEVICES` - 可用的GPU设备 (默认: 0)

## 使用方法

### 通过Makefile使用

项目根目录的Makefile提供了简单的命令来构建和运行容器：

```bash
# 构建GPU版本镜像
make build-gpu

# 构建CPU版本镜像
make build-cpu

# 运行GPU版本容器
make run-gpu

# 运行CPU版本容器
make run-cpu

# 使用Docker Compose启动所有服务
make up

# 使用Docker Compose启动GPU服务
make up-gpu

# 使用Docker Compose启动CPU服务
make up-cpu

# 停止所有容器
make stop

# 清理所有容器和镜像
make clean
```

### 手动构建和运行

也可以直接使用Docker命令构建和运行：

```bash
# 构建GPU版本
docker build -t deepchem-api:gpu-latest -f docker/Dockerfile.gpu --build-arg CONTAINER_PORT=8000 .

# 构建CPU版本
docker build -t deepchem-api:cpu-latest -f docker/Dockerfile.cpu --build-arg CONTAINER_PORT=8000 .

# 运行GPU版本
docker run -d --name deepchem-api-gpu \
    --gpus all \
    -p 8000:8000 \
    -e CONTAINER_PORT=8000 \
    -e USE_GPU=true \
    -e CUDA_VISIBLE_DEVICES=0 \
    deepchem-api:gpu-latest

# 运行CPU版本
docker run -d --name deepchem-api-cpu \
    -p 8000:8000 \
    -e CONTAINER_PORT=8000 \
    -e USE_GPU=false \
    deepchem-api:cpu-latest
```

### 使用Docker Compose

```bash
# 启动所有服务
cd docker && docker-compose up -d

# 启动GPU服务
cd docker && docker-compose up -d deepchem-api-gpu

# 启动CPU服务
cd docker && docker-compose up -d deepchem-api-cpu
``` 


curl -X POST http://172.18.0.2:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CCO",
    "name": "Ethanol"
  }'
