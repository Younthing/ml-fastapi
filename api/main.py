# main.py
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional
import deepchem as dc
import time
import os
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
import functools
import hashlib
import json
from contextlib import asynccontextmanager
import aiofiles
import aiofiles.os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 服务配置
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# 配置日志
logger = logging.getLogger(__name__)

# 模型配置和版本控制
MODEL_DIR = os.environ.get("MODEL_DIR", "AttentiveModel")
MODEL_VERSION_FILE = os.path.join(MODEL_DIR, "version.txt")
RELOAD_INTERVAL = int(os.environ.get("MODEL_RELOAD_INTERVAL", 3600))  # 默认1小时
CACHE_EXPIRY = int(os.environ.get("CACHE_EXPIRY", 3600))  # 默认缓存1小时
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 100000))
THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", 4))

# GPU配置
USE_GPU = os.environ.get("USE_GPU", "false").lower() == "true"
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# 异步线程池执行器
thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)


# 模型和缓存管理类
class ModelManager:
    def __init__(self):
        self.model = None
        self.featurizer = None
        self.version = "1.0.0"
        self.last_load_time = None
        self.lock = asyncio.Lock()
        self.cache = {}
        self.cache_expiry = {}
        self._shutdown_event = asyncio.Event()
        self._reload_task = None

    async def initialize(self):
        """初始化模型管理器"""
        await self.load_model()
        # 启动模型重载后台任务
        self._reload_task = asyncio.create_task(self._periodic_reload())
        logger.info("模型管理器初始化完成")

    async def shutdown(self):
        """关闭模型管理器"""
        if self._reload_task:
            self._shutdown_event.set()
            await self._reload_task
        logger.info("模型管理器已关闭")

    async def _get_model_version(self):
        """获取模型版本"""
        try:
            if await aiofiles.os.path.exists(MODEL_VERSION_FILE):
                async with aiofiles.open(MODEL_VERSION_FILE, "r") as f:
                    version = await f.read()
                    return version.strip()
            return datetime.now().strftime("%Y%m%d%H%M%S")  # 默认使用时间戳作为版本
        except Exception as e:
            logger.error(f"读取模型版本失败: {str(e)}")
            return "unknown"

    async def _should_reload(self):
        """检查是否应该重新加载模型"""
        if not self.last_load_time:
            return True

        # 检查版本文件是否更新
        current_version = await self._get_model_version()
        if current_version != self.version:
            logger.info(
                f"检测到新模型版本: {current_version}, 当前版本: {self.version}"
            )
            return True

        # 检查是否超过重载间隔
        time_since_load = datetime.now() - self.last_load_time
        if time_since_load > timedelta(seconds=RELOAD_INTERVAL):
            logger.info(f"模型加载时间超过重载间隔: {time_since_load}")
            return True

        return False

    async def _periodic_reload(self):
        """周期性检查并重新加载模型"""
        logger.info("启动模型周期性重载任务")
        try:
            while not self._shutdown_event.is_set():
                # 等待一段时间或直到关闭事件被设置
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=60,  # 每分钟检查一次
                    )
                except asyncio.TimeoutError:
                    pass

                if self._shutdown_event.is_set():
                    break

                # 检查是否需要重新加载
                if await self._should_reload():
                    logger.info("开始周期性重新加载模型")
                    await self.load_model()
                    logger.info("周期性重新加载模型完成")
        except Exception as e:
            logger.error(f"周期性重载任务错误: {str(e)}")
        finally:
            logger.info("周期性重载任务已结束")

    async def load_model(self):
        """异步加载模型"""
        async with self.lock:
            try:
                logger.info(f"开始加载AttentiveFPModel: {MODEL_DIR}")
                # 在线程池中执行CPU密集型操作
                self.model = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, functools.partial(self._load_model_sync)
                )

                logger.info("初始化MolGraphConvFeaturizer")
                self.featurizer = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, functools.partial(self._load_featurizer_sync)
                )

                self.version = await self._get_model_version()
                self.last_load_time = datetime.now()
                logger.info(f"模型加载成功，版本: {self.version}")

                # 清除过期缓存
                self.clear_expired_cache()

                return True
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                logger.exception("详细错误信息:")
                return False

    def _load_model_sync(self):
        """同步加载模型（在线程池中执行）"""
        model = dc.models.AttentiveFPModel(
            model_dir=MODEL_DIR,
            n_tasks=1,
            num_layers=4,
            graph_feat_size=128,
        )
        model.restore()
        return model

    def _load_featurizer_sync(self):
        """同步加载特征化器（在线程池中执行）"""
        return dc.feat.MolGraphConvFeaturizer(use_edges=True)

    def get_cache_key(self, smiles):
        """生成分子的缓存键"""
        return hashlib.md5(smiles.encode()).hexdigest()

    def get_from_cache(self, smiles):
        """从缓存获取预测结果"""
        key = self.get_cache_key(smiles)
        if key in self.cache:
            # 检查是否过期
            if datetime.now() < self.cache_expiry.get(key, datetime.min):
                return self.cache[key]
            else:
                # 过期缓存，删除
                self.cache.pop(key, None)
                self.cache_expiry.pop(key, None)
        return None

    def add_to_cache(self, smiles, prediction):
        """添加预测结果到缓存"""
        key = self.get_cache_key(smiles)
        self.cache[key] = prediction
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=CACHE_EXPIRY)

    def clear_expired_cache(self):
        """清除过期的缓存项"""
        now = datetime.now()
        expired_keys = [k for k, exp in self.cache_expiry.items() if now > exp]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_expiry.pop(key, None)
        logger.info(
            f"已清除 {len(expired_keys)} 个过期缓存项，剩余 {len(self.cache)} 个缓存项"
        )

    async def predict_single(self, smiles):
        """预测单个分子"""
        # 先检查缓存
        cached = self.get_from_cache(smiles)
        if cached is not None:
            return cached

        # 没有缓存，执行预测
        try:
            # 在线程池中执行CPU密集型操作
            features = await asyncio.get_event_loop().run_in_executor(
                thread_pool, functools.partial(self.featurizer.featurize, [smiles])
            )

            raw_predictions = await asyncio.get_event_loop().run_in_executor(
                thread_pool, functools.partial(self.model.predict_on_batch, features)
            )

            # 处理预测结果
            if raw_predictions.ndim > 1 and raw_predictions.shape[1] > 1:
                # 多任务模型
                prediction = {"values": raw_predictions[0].tolist()}
            else:
                # 单任务模型
                prediction = {"value": float(raw_predictions[0][0])}

            # 添加到缓存
            self.add_to_cache(smiles, prediction)

            return prediction
        except Exception as e:
            logger.error(f"预测分子错误: {smiles}, 错误: {str(e)}")
            raise

    async def predict_batch(self, smiles_list):
        """批量预测多个分子"""
        results = []
        to_predict = []
        to_predict_indices = []

        # 先检查每个分子的缓存
        for idx, smiles in enumerate(smiles_list):
            cached = self.get_from_cache(smiles)
            if cached is not None:
                results.append((idx, cached))
            else:
                to_predict.append(smiles)
                to_predict_indices.append(idx)

        # 如果有需要预测的分子
        if to_predict:
            # 将待预测分子分成多个批次
            batch_size = min(MAX_BATCH_SIZE, len(to_predict))
            batches = [
                to_predict[i : i + batch_size]
                for i in range(0, len(to_predict), batch_size)
            ]
            batch_indices = [
                to_predict_indices[i : i + batch_size]
                for i in range(0, len(to_predict_indices), batch_size)
            ]

            # 处理每个批次
            for batch_idx, (batch_smiles, indices) in enumerate(
                zip(batches, batch_indices)
            ):
                try:
                    logger.info(
                        f"处理批次 {batch_idx + 1}/{len(batches)}, 大小: {len(batch_smiles)}"
                    )

                    # 特征化当前批次
                    features = await asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        functools.partial(self.featurizer.featurize, batch_smiles),
                    )

                    # 预测当前批次
                    raw_predictions = await asyncio.get_event_loop().run_in_executor(
                        thread_pool,
                        functools.partial(self.model.predict_on_batch, features),
                    )

                    # 处理当前批次的结果
                    for i, (smiles, orig_idx) in enumerate(zip(batch_smiles, indices)):
                        try:
                            if (
                                raw_predictions.ndim > 1
                                and raw_predictions.shape[1] > 1
                            ):
                                prediction = {"values": raw_predictions[i].tolist()}
                            else:
                                prediction = {"value": float(raw_predictions[i][0])}

                            # 添加到缓存
                            self.add_to_cache(smiles, prediction)

                            # 添加到结果
                            results.append((orig_idx, prediction))
                        except Exception as e:
                            logger.error(f"处理批次结果错误: {smiles}, 错误: {str(e)}")
                except Exception as e:
                    logger.error(f"批次 {batch_idx + 1} 处理错误: {str(e)}")
                    # 对于失败的批次，尝试单独处理每个分子
                    for smiles, orig_idx in zip(batch_smiles, indices):
                        try:
                            prediction = await self.predict_single(smiles)
                            results.append((orig_idx, prediction))
                        except Exception as inner_e:
                            logger.error(
                                f"单独预测分子错误: {smiles}, 错误: {str(inner_e)}"
                            )

        # 按原始索引排序结果
        sorted_results = [pred for _, pred in sorted(results, key=lambda x: x[0])]

        # 检查是否所有分子都有预测结果
        if len(sorted_results) < len(smiles_list):
            logger.warning(
                f"某些分子预测失败: 预期 {len(smiles_list)}, 实际 {len(sorted_results)}"
            )

        return sorted_results


# 创建模型管理器实例
model_manager = ModelManager()


# 应用启动和关闭事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载
    await model_manager.initialize()
    yield
    # 关闭时清理
    await model_manager.shutdown()


# 创建FastAPI应用
app = FastAPI(
    title="DeepChem AttentiveFPModel API",
    description="基于AttentiveFPModel提供分子属性预测服务的REST API",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加CORS中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API数据模型


# 修改验证器
class Molecule(BaseModel):
    smiles: str = Field(..., description="分子的SMILES表示", example="CCO")
    name: Optional[str] = Field(None, description="分子名称（可选）", example="Ethanol")
    additional_info: Optional[Dict[str, Any]] = Field(
        None, description="其他分子相关信息"
    )

    @field_validator("smiles")  # 改用 field_validator
    def validate_smiles(cls, v):
        if not v or len(v) < 1:
            raise ValueError("SMILES字符串不能为空")
        return v


class BatchMoleculeRequest(BaseModel):
    molecules: List[Molecule] = Field(..., description="分子列表")
    options: Optional[Dict[str, Any]] = Field(None, description="批量预测的配置选项")

    @field_validator("molecules")  # 改用 field_validator
    def validate_molecules(cls, v):
        if not v:
            raise ValueError("分子列表不能为空")
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"分子数量不能超过 {MAX_BATCH_SIZE}")
        return v


class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="请求的唯一标识符")
    molecule: Molecule
    predictions: Dict[str, Any] = Field(..., description="预测结果")
    processing_time_ms: float = Field(..., description="处理时间（毫秒）")
    from_cache: bool = Field(False, description="结果是否来自缓存")


class BatchPredictionResponse(BaseModel):
    request_id: str = Field(..., description="批量请求的唯一标识符")
    results: List[PredictionResponse] = Field(..., description="每个分子的预测结果")
    total_molecules: int = Field(..., description="处理的分子总数")
    failed_molecules: int = Field(..., description="预测失败的分子数量")
    total_processing_time_ms: float = Field(..., description="总处理时间（毫秒）")
    cache_hits: int = Field(0, description="缓存命中数量")


class ApiError(BaseModel):
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误描述")
    details: Optional[Dict[str, Any]] = Field(None, description="详细错误信息")


class ModelInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: str = Field(..., description="模型状态")
    version: str = Field(..., description="模型版本")
    last_loaded: Optional[str] = Field(None, description="最后加载时间")
    model_type: Optional[str] = Field(None, description="模型类型")
    model_dir: Optional[str] = Field(None, description="模型目录")
    cache_stats: Dict[str, int] = Field(..., description="缓存统计")
    model_parameters: Optional[Dict[str, Any]] = Field(None, description="模型配置")


# 依赖函数
async def get_model_manager():
    """依赖注入函数，用于获取模型管理器实例"""
    if model_manager.model is None or model_manager.featurizer is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "MODEL_NOT_READY",
                "message": "模型尚未加载完成，请稍后再试",
            },
        )
    return model_manager


# 自定义异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = {
        "error_code": "INTERNAL_ERROR",
        "message": "服务器内部错误",
        "details": {"error": str(exc)},
    }
    logger.error(f"全局异常: {str(exc)}")
    return JSONResponse(status_code=500, content=error_detail)


# API端点
@app.get("/api/status", response_model=ModelInfo)
async def get_status(model: ModelManager = Depends(get_model_manager)):
    """获取API服务状态"""
    return {
        "status": "online",
        "version": model.version,
        "last_loaded": model.last_load_time.isoformat()
        if model.last_load_time
        else None,
        "model_type": "AttentiveFPModel",
        "model_dir": MODEL_DIR,
        "cache_stats": {
            "items": len(model.cache),
            "size_bytes": sum(len(json.dumps(v)) for v in model.cache.values()),
        },
        "model_config": {
            "n_tasks": getattr(model.model, "n_tasks", 1),
            "mode": getattr(model.model, "mode", "unknown"),
            "task_type": getattr(model.model, "task_type", "unknown"),
            "output_types": [
                str(t) for t in getattr(model.model, "output_types", ["unknown"])
            ],
            "batch_size": getattr(model.model, "batch_size", 0),
        },
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_single(
    molecule: Molecule, model: ModelManager = Depends(get_model_manager)
):
    """对单个分子进行预测"""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # 检查缓存
        cached_prediction = model.get_from_cache(molecule.smiles)
        from_cache = cached_prediction is not None

        if from_cache:
            prediction = cached_prediction
        else:
            # 执行预测
            prediction = await model.predict_single(molecule.smiles)

        # 计算处理时间
        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            request_id=request_id,
            molecule=molecule,
            predictions=prediction,
            processing_time_ms=processing_time,
            from_cache=from_cache,
        )
    except Exception as e:
        logger.error(f"预测错误: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "PREDICTION_ERROR",
                "message": "预测过程中发生错误",
                "details": {"error": str(e), "smiles": molecule.smiles},
            },
        )


@app.post("/api/batch-predict", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchMoleculeRequest, model: ModelManager = Depends(get_model_manager)
):
    """批量预测多个分子"""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    all_smiles = [molecule.smiles for molecule in request.molecules]
    results = []
    failed_count = 0
    cache_hits = 0

    try:
        # 批量预测所有分子
        all_predictions = await model.predict_batch(all_smiles)

        # 处理结果
        for i, (molecule, prediction) in enumerate(
            zip(request.molecules, all_predictions)
        ):
            # 检查是否来自缓存
            from_cache = model.get_from_cache(molecule.smiles) is not None
            if from_cache:
                cache_hits += 1

            # 为每个分子记录处理时间
            # 注意：这里简化处理，不计算单个分子的处理时间
            mol_processing_time = 0.0

            results.append(
                PredictionResponse(
                    request_id=f"{request_id}_{i}",
                    molecule=molecule,
                    predictions=prediction,
                    processing_time_ms=mol_processing_time,
                    from_cache=from_cache,
                )
            )
    except Exception as e:
        logger.error(f"批量预测整体错误: {str(e)}")
        # 错误处理已由模型管理器处理

    # 计算总处理时间
    total_processing_time = (time.time() - start_time) * 1000

    # 计算失败的分子数量
    failed_count = len(request.molecules) - len(results)

    return BatchPredictionResponse(
        request_id=request_id,
        results=results,
        total_molecules=len(request.molecules),
        failed_molecules=failed_count,
        total_processing_time_ms=total_processing_time,
        cache_hits=cache_hits,
    )


@app.post("/api/reload-model")
async def reload_model():
    """手动触发模型重新加载"""
    try:
        success = await model_manager.load_model()
        if success:
            return {
                "status": "success",
                "message": "模型重新加载成功",
                "version": model_manager.version,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "MODEL_RELOAD_FAILED",
                    "message": "模型重新加载失败",
                },
            )
    except Exception as e:
        logger.error(f"手动重载模型错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "MODEL_RELOAD_ERROR",
                "message": "模型重新加载过程中发生错误",
                "details": {"error": str(e)},
            },
        )


@app.post("/api/clear-cache")
async def clear_cache(model: ModelManager = Depends(get_model_manager)):
    """清除预测结果缓存"""
    try:
        cache_count = len(model.cache)
        model.cache.clear()
        model.cache_expiry.clear()
        return {
            "status": "success",
            "message": f"已清除 {cache_count} 个缓存项",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"清除缓存错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "CACHE_CLEAR_ERROR",
                "message": "清除缓存过程中发生错误",
                "details": {"error": str(e)},
            },
        )


@app.get("/api/models/info")
async def get_model_info(model: ModelManager = Depends(get_model_manager)):
    """获取AttentiveFPModel模型信息"""
    try:
        model_info = {
            "type": "AttentiveFPModel",
            "directory": MODEL_DIR,
            "version": model.version,
            "last_loaded": model.last_load_time.isoformat()
            if model.last_load_time
            else None,
            "featurizer": "MolGraphConvFeaturizer",
            "featurizer_config": {"use_edges": True},
            "model_config": {
                "n_tasks": getattr(model.model, "n_tasks", 1),
                "mode": getattr(model.model, "mode", "unknown"),
                "task_type": getattr(model.model, "task_type", "unknown"),
                "output_types": [
                    str(t) for t in getattr(model.model, "output_types", ["unknown"])
                ],
                "batch_size": getattr(model.model, "batch_size", 0),
            },
            "cache_info": {
                "item_count": len(model.cache),
                "memory_usage_approx": sum(
                    len(json.dumps(v)) for v in model.cache.values()
                ),
                "expiry_time": CACHE_EXPIRY,
            },
        }

        return model_info
    except Exception as e:
        logger.error(f"获取模型信息错误: {str(e)}")
        return {"status": "error", "message": "获取模型信息失败", "error": str(e)}


@app.get("/api/featurizer/info")
async def get_featurizer_info(model: ModelManager = Depends(get_model_manager)):
    """获取特征化器信息"""
    try:
        featurizer_info = {
            "type": "MolGraphConvFeaturizer",
            "use_edges": getattr(model.featurizer, "use_edges", True),
            "atom_properties": getattr(model.featurizer, "atom_properties", []),
            "use_chirality": getattr(model.featurizer, "use_chirality", False),
        }

        return featurizer_info
    except Exception as e:
        logger.error(f"获取特征化器信息错误: {str(e)}")
        return {"status": "error", "message": "获取特征化器信息失败", "error": str(e)}


@app.get("/api/examples")
async def get_examples(model: ModelManager = Depends(get_model_manager)):
    """获取API使用示例"""
    examples = {
        "single_prediction": {
            "url": "/api/predict",
            "method": "POST",
            "request_body": {"smiles": "CCO", "name": "Ethanol"},
            "curl_example": 'curl -X POST "http://localhost:8000/api/predict" -H "Content-Type: application/json" -d \'{"smiles": "CCO", "name": "Ethanol"}\'',
            "response_example": {
                "request_id": "123e4567-e89b-12d3-a456-426614174000",
                "molecule": {"smiles": "CCO", "name": "Ethanol"},
                "predictions": {"value": 0.85},
                "processing_time_ms": 125.45,
                "from_cache": False,
            },
        },
        "batch_prediction": {
            "url": "/api/batch-predict",
            "method": "POST",
            "request_body": {
                "molecules": [
                    {"smiles": "CCO", "name": "Ethanol"},
                    {"smiles": "CC(=O)O", "name": "Acetic acid"},
                ]
            },
            "curl_example": 'curl -X POST "http://localhost:8000/api/batch-predict" -H "Content-Type: application/json" -d \'{"molecules": [{"smiles": "CCO", "name": "Ethanol"}, {"smiles": "CC(=O)O", "name": "Acetic acid"}]}\'',
        },
        "model_management": {
            "reload_model": {
                "url": "/api/reload-model",
                "method": "POST",
                "description": "手动触发模型重新加载",
                "curl_example": 'curl -X POST "http://localhost:8000/api/reload-model"',
            },
            "clear_cache": {
                "url": "/api/clear-cache",
                "method": "POST",
                "description": "清除所有预测结果缓存",
                "curl_example": 'curl -X POST "http://localhost:8000/api/clear-cache"',
            },
        },
    }

    # 添加基于当前模型的示例
    try:
        # 获取一些预设的示例分子
        example_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        predictions = []

        for smiles in example_smiles:
            try:
                prediction = await model.predict_single(smiles)
                predictions.append({"smiles": smiles, "prediction": prediction})
            except Exception as e:
                logger.error(f"示例预测错误: {smiles}, 错误: {str(e)}")

        if predictions:
            examples["model_specific"] = {
                "description": "基于当前模型的预测示例",
                "predictions": predictions,
            }
        else:
            examples["model_specific"] = {"description": "当前模型没有可用的示例预测"}
    except Exception as e:
        logger.error(f"获取模型示例错误: {str(e)}")
        examples["model_specific"] = {
            "description": "获取模型示例时发生错误",
            "error": str(e),
        }
    return examples


# 启动FastAPI应用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
