from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import subprocess
from typing import Dict, Optional

gpu_status_router = APIRouter()


class GPUMetrics(BaseModel):
    timestamp: str
    memory_allocated_mb: float
    memory_reserved_mb: float
    memory_used_mb: float
    memory_total_mb: float
    gpu_utilization: float
    device_name: str
    device_count: int
    cuda_version: str
    error: Optional[str] = None


def get_nvidia_smi_metrics() -> Dict:
    """Get GPU metrics using nvidia-smi command."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        timestamp, memory_used, memory_total, gpu_util = output.strip().split(",")
        return {
            "timestamp": timestamp.strip(),
            "memory_used": float(memory_used.strip()),
            "memory_total": float(memory_total.strip()),
            "gpu_utilization": float(gpu_util.strip()),
        }
    except (subprocess.SubprocessError, ValueError) as e:
        return {"error": f"Error reading nvidia-smi metrics: {str(e)}"}


def get_torch_cuda_metrics() -> Dict:
    """Get GPU metrics using PyTorch CUDA APIs."""
    try:
        return {
            "memory_allocated": torch.cuda.memory_allocated() / 1024**2,
            "memory_reserved": torch.cuda.memory_reserved() / 1024**2,
            "device_name": torch.cuda.get_device_name(),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }
    except Exception as e:
        return {"error": f"Error reading PyTorch CUDA metrics: {str(e)}"}


@gpu_status_router.get("/gpu/status", response_model=GPUMetrics)
async def get_gpu_stats():
    """
    Get comprehensive GPU statistics combining nvidia-smi and PyTorch CUDA metrics.

    Returns:
        GPUMetrics: Object containing various GPU metrics

    Raises:
        HTTPException: If GPU metrics cannot be retrieved
    """
    if not torch.cuda.is_available():
        raise HTTPException(
            status_code=400, detail="CUDA is not available on this system"
        )

    nvidia_metrics = get_nvidia_smi_metrics()
    torch_metrics = get_torch_cuda_metrics()

    if "error" in nvidia_metrics or "error" in torch_metrics:
        error_msg = []
        if "error" in nvidia_metrics:
            error_msg.append(nvidia_metrics["error"])
        if "error" in torch_metrics:
            error_msg.append(torch_metrics["error"])
        raise HTTPException(status_code=500, detail=" | ".join(error_msg))

    # Combine metrics
    return GPUMetrics(
        timestamp=nvidia_metrics["timestamp"],
        memory_allocated_mb=round(torch_metrics["memory_allocated"], 2),
        memory_reserved_mb=round(torch_metrics["memory_reserved"], 2),
        memory_used_mb=nvidia_metrics["memory_used"],
        memory_total_mb=nvidia_metrics["memory_total"],
        gpu_utilization=nvidia_metrics["gpu_utilization"],
        device_name=torch_metrics["device_name"],
        device_count=torch_metrics["device_count"],
        cuda_version=torch_metrics["cuda_version"],
    )
