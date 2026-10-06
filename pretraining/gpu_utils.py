"""
GPU monitoring utilities with minimal overhead for training loops.
"""

import os
from typing import Dict, List, Optional
import torch

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """Lightweight GPU monitoring with caching to minimize overhead."""

    def __init__(self):
        self.initialized = False
        self.gpu_handles = []
        self.device_count = 0

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)
                ]
                self.initialized = True
            except pynvml.NVMLError:
                pass

    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics with minimal overhead. Returns averages across all GPUs."""
        if not self.initialized:
            return {}

        try:
            utilizations = []
            memory_utils = []
            temperatures = []
            power_draws = []

            for handle in self.gpu_handles:
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilizations.append(util.gpu)
                memory_utils.append(util.memory)

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temperatures.append(temp)

            stats = {
                "gpu_utilization_avg": sum(utilizations) / len(utilizations) if utilizations else 0,
                "gpu_memory_util_avg": sum(memory_utils) / len(memory_utils) if memory_utils else 0,
                "gpu_temp_avg": sum(temperatures) / len(temperatures) if temperatures else 0,
            }

            return stats

        except pynvml.NVMLError:
            return {}

    def get_detailed_stats(self) -> Dict[str, List[float]]:
        """Get per-GPU detailed statistics. Use sparingly to avoid overhead."""
        if not self.initialized:
            return {}

        try:
            stats = {
                "gpu_utilizations": [],
                "memory_utilizations": [],
                "temperatures": [],
                # "power_draws": []
            }

            for i, handle in enumerate(self.gpu_handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["gpu_utilizations"].append(util.gpu)
                stats["memory_utilizations"].append(util.memory)

                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                stats["temperatures"].append(temp)

            return stats

        except pynvml.NVMLError:
            return {}


# Global monitor instance to avoid re-initialization overhead
_gpu_monitor = None


def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor


def get_gpu_stats() -> Dict[str, float]:
    """Convenience function to get GPU stats quickly."""
    return get_gpu_monitor().get_gpu_stats()


def get_combined_gpu_stats() -> Dict[str, float]:
    """Get both NVML and PyTorch GPU stats combined."""
    stats = {}
    stats.update(get_gpu_stats())
    return stats
