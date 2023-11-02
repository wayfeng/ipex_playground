import torch
import numpy as np

def measure_latency(model, input, niter=100):
    # INIT LOGGERS
    starter, ender = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    timings = np.zeros((niter,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for i in range(niter):
            starter.record()
            _ = model(input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.xpu.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time
    
    mean = np.sum(timings) / niter
    std = np.std(timings)
    return mean, std

def measure_throught(model, input, batch_size=1, niter=100):
    total_time = 0
    with torch.no_grad():
        for i in range(niter):
            starter, ender = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            starter.record()
            _ = model(input)
            ender.record()
            torch.xpu.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (niter * batch_size) / total_time
    return throughput
