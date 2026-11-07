#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

// ============================================================================
// Device-side spin delay with a run-time duration (criticality probe).
//
// NEW KERNEL. Nothing existing is replaced; the only caller is the GLM-5
// decode criticality probe (batchgen/models/glm/glm5/criticality_probe.py),
// which inserts one launch per instrumented site into a captured decode graph
// and is absent from the graph unless a batch requests it.
//
// WHY NOT torch.cuda._sleep: it takes its cycle count as a KERNEL ARGUMENT, and
// a captured graph replays the arguments it recorded. Changing the delay would
// then mean recapturing the graph for every level of a sweep. Here the graph
// bakes only the POINTER to a device int64 table and the site's index into it;
// the duration is LOADED when the kernel runs, so a host->device copy into the
// table between two replays changes the delay of the same graph.
//
// Semantics:
//   * one block of 32 threads; thread 0 alone reads delay_ns[site] and waits,
//     the other lanes exit at once (one warp is the smallest launch unit, and
//     the block occupies a single SM slot while it waits);
//   * delay_ns[site] <= 0 returns immediately -- the delay-0 cost of a site is
//     the bare launch, which the probe reports as its baseline shift;
//   * otherwise the thread polls %globaltimer (nanoseconds, device-global)
//     with __nanosleep between polls until at least delay_ns[site] ns have
//     elapsed since its first read. The resolution is the globaltimer update
//     granularity plus one sleep quantum; it is calibrated on the node by
//     tests/gpu/test_spin_delay_gpu.py, not assumed.
//
// The kernel reads one int64 and writes nothing, so it cannot change any value
// the graph computes; it only moves time on the stream it is launched on.
// ============================================================================

namespace {

// Back-off between polls. Short against the 10-100 us delays the probe
// injects, long enough that the waiting warp does not hammer the issue slot of
// the SM it shares with real work.
constexpr unsigned kPollSleepNs = 128;

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void spin_delay_kernel(const int64_t* __restrict__ delay_ns, int64_t site) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t wait_ns = delay_ns[site];
    if (wait_ns <= 0) {
        return;
    }
    const uint64_t start = globaltimer_ns();
    const uint64_t target = static_cast<uint64_t>(wait_ns);
    while (globaltimer_ns() - start < target) {
        __nanosleep(kPollSleepNs);
    }
}

}  // namespace

void spin_delay(torch::Tensor delay_ns, int64_t site) {
    TORCH_CHECK(delay_ns.is_cuda(), "spin_delay: delay_ns must be a CUDA tensor");
    TORCH_CHECK(delay_ns.scalar_type() == torch::kInt64,
                "spin_delay: delay_ns must be int64 nanoseconds, got ",
                delay_ns.scalar_type());
    TORCH_CHECK(delay_ns.dim() == 1 && delay_ns.is_contiguous(),
                "spin_delay: delay_ns must be a contiguous 1-D table");
    TORCH_CHECK(site >= 0 && site < delay_ns.numel(),
                "spin_delay: site ", site, " is outside the ", delay_ns.numel(),
                "-entry delay table");

    const c10::cuda::CUDAGuard device_guard(delay_ns.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    spin_delay_kernel<<<1, 32, 0, stream>>>(delay_ns.data_ptr<int64_t>(), site);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spin_delay", &spin_delay,
          "Spin on the current stream for delay_ns[site] nanoseconds, read at "
          "kernel run time (CUDA)",
          py::arg("delay_ns"),
          py::arg("site"));
}
