# 使用CUDA基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=${CONTAINER_PORT:-8000} \
    WORKERS=4 \
    LOG_LEVEL=info

# 安装基本依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext2 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# 设置工作目录
WORKDIR /app

# 复制环境配置文件和依赖文件
COPY environment.yml requirements.txt ./

# 创建并激活conda环境
RUN conda env create -f environment.yml && \
    conda clean -afy

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE ${PORT}

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

# 运行应用
SHELL ["/bin/bash", "--login", "-c"]
CMD ["conda", "run", "-n", "deepchem-api", "uvicorn", "app.main:app", "--host", "${HOST}", "--port", "${PORT}", "--workers", "${WORKERS}", "--log-level", "${LOG_LEVEL}"] 