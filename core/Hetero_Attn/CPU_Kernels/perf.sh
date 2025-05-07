#!/bin/bash
# numactl --cpunodebind=0 --membind=0 ./test_kernel 1 10 &
./test_kernel 1 10 
PID=$!

# Wait for initialization to complete
# You might want to implement a more reliable mechanism here
# sleep 1  # Adjust based on your initialization time

# Start profiling with per-thread stats and focusing on IPC and cache metrics
# perf stat -p $PID -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cache-references,cache-misses
# perf stat -p $PID -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-store-misses,node-loads,node-load-misses
