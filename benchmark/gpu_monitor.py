import multiprocessing
from typing import Optional
import pynvml
import torch
from logger import _LOGGER_MAIN

logger = _LOGGER_MAIN


def _device_to_uuid_cuda(cuda_idx: Optional[int] = None) -> str:
    import uuid as _uuid
    from cuda.bindings import driver as cuda  # Use the low-level driver API

    def CUASSERT(cuda_ret):
        err = cuda_ret[0]
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA ERROR: {err}")
        if len(cuda_ret) > 1:
            return cuda_ret[1]
        return None

    CUASSERT(cuda.cuInit(0))
    cuda_idx = cuda_idx if cuda_idx is not None else torch.cuda.current_device()
    dev = CUASSERT(cuda.cuDeviceGet(cuda_idx))
    uuid_struct = CUASSERT(cuda.cuDeviceGetUuid(dev))
    uuid_str = str(_uuid.UUID(bytes=bytes(uuid_struct.bytes)))
    return uuid_str


def _device_to_uuid_torch(cuda_idx: Optional[int] = None):
    device_properties = torch.cuda.get_device_properties(cuda_idx)
    uuid_str = str(device_properties.uuid)
    return uuid_str


def get_nvml_id_by_cuda_uuid(cuda_id):
    errors = []
    func_list = [
        (_device_to_uuid_torch, "torch.cuda"),
        (_device_to_uuid_cuda, "cuda-python"),
    ]
    for get_uuid_func, name in func_list:
        try:
            uuid_str = get_uuid_func(cuda_id)
            break
        except Exception as e:
            errors.append(f"Using {name}: {e}")
    else:
        raise RuntimeError(f"Failed to get UUID for device {cuda_id}: {errors}")

    pynvml.nvmlInit()
    encoded = uuid_str.replace("-", "").encode("utf-8")
    try:
        # Get the NVML device handle using the UUID
        handle = pynvml.nvmlDeviceGetHandleByUUID(encoded)
        id = pynvml.nvmlDeviceGetIndex(handle)
        return id

    except pynvml.NVMLError as e:
        raise RuntimeError(f"PyNVML Error: {e}")

    finally:
        pynvml.nvmlShutdown()


def monitor_gpu_memory(queue, running_flag, nvml_id):
    pynvml.nvmlInit()
    max_memory_usage = 0

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_id)
        while running_flag.value:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage = info.used / (1024 * 1024)  # Convert to MiB
            max_memory_usage = max(max_memory_usage, memory_usage)
    finally:
        pynvml.nvmlShutdown()
        queue.put(max_memory_usage)


class GPUMemoryMonitor:
    def __init__(self, device: Optional[int] = None):
        self.process = None
        self.queue = multiprocessing.Queue()
        self.running_flag = multiprocessing.Value("b", False)
        self.device = torch.device(
            device if device is not None else torch.cuda.current_device()
        )

        pynvml.nvmlInit()
        try:
            self.nvml_id = get_nvml_id_by_cuda_uuid(self.device.index)
            logger.debug(f"Initialized GPU monitor for device {self.device} (NVML ID: {self.nvml_id})")
        except Exception as e:
            logger.warning(f"Error getting NVML device index: {e}. Using NVML ID 0.")
            self.nvml_id = 0
        finally:
            pynvml.nvmlShutdown()

    def start(self):
        if self.running_flag.value:
            logger.warning("Cannot start GPU monitor: it is already running.")
            return
        self.running_flag.value = True
        self.process = multiprocessing.Process(
            target=monitor_gpu_memory,
            args=(self.queue, self.running_flag, self.nvml_id),
        )
        self.process.start()
        logger.info(f"GPU monitor started for device {self.device} (NVML ID: {self.nvml_id})")

    def stop(self):
        if not self.running_flag.value:
            logger.warning("Cannot stop GPU monitor: it is not running.")
            return
        self.running_flag.value = False
        self.process.join()
        logger.info("GPU monitor stopped")

    def get_max_memory_usage(self):
        if not self.queue.empty():
            max_usage = self.queue.get()
            logger.debug(f"Retrieved max GPU memory usage: {max_usage:.2f} MiB")
            return max_usage
        logger.warning("No memory usage data available")
        return 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


if __name__ == "__main__":
    monitor = GPUMemoryMonitor()
    monitor.start()

    # Simulate some workload

    monitor.stop()

    logger.info(f"Maximum GPU memory usage (MB): {monitor.get_max_memory_usage()}")
