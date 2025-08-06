import multiprocessing
import pynvml


def monitor_gpu_memory(queue, running_flag):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
    max_memory_usage = 0

    try:
        while running_flag.value:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_usage = info.used / (1024 * 1024)  # Convert to MiB
            max_memory_usage = max(max_memory_usage, memory_usage)
    finally:
        pynvml.nvmlShutdown()
        queue.put(max_memory_usage)


class GPUMemoryMonitor:
    def __init__(self):
        self.process = None
        self.queue = multiprocessing.Queue()
        self.running_flag = multiprocessing.Value("b", False)

    def start(self):
        if self.running_flag.value:
            print("GPU monitor is already running.")
            return
        self.running_flag.value = True
        self.process = multiprocessing.Process(
            target=monitor_gpu_memory, args=(self.queue, self.running_flag)
        )
        self.process.start()

    def stop(self):
        if not self.running_flag.value:
            print("GPU monitor is not running.")
            return
        self.running_flag.value = False
        self.process.join()

    def get_max_memory_usage(self):
        if not self.queue.empty():
            return self.queue.get()
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

    print("Maximum GPU memory usage (MB):", monitor.get_max_memory_usage())
