#!/bin/bash
# KV Cache Position Verification Script
# Analyzes debug output to verify KV cache positions are sequential (not repeated)
#
# Usage:
#   ./verify_kv_positions.sh [log_file]
#
# Default log file: /tmp/kv_debug_output.log

LOG_FILE=${1:-/tmp/kv_debug_output.log}

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    echo ""
    echo "Run the test first with debug flags:"
    echo "  export BATCHGEN_DEBUG_DECODE=1"
    echo "  export BATCHGEN_DEBUG_KV_WRITE=1"
    echo "  python test/gpt_oss_mmlu_pro_test/gpt_oss_mmlu_pro_batch_test.py --max_prompts 2 --max_tokens 20 2>&1 | tee /tmp/kv_debug_output.log"
    exit 1
fi

echo "=========================================="
echo "KV Cache Position Verification"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo ""

# Count total decode iterations
iter_count=$(grep -c "\[DECODE DEBUG\] Iteration" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Total decode iterations found: $iter_count"
echo ""

# Check cache_seqlens progression
echo "=== cache_seqlens Progression ==="
cache_seqlens=$(grep "cache_seqlens\[:5\]" "$LOG_FILE" | head -10)
if [ -n "$cache_seqlens" ]; then
    echo "$cache_seqlens"
else
    echo "(No cache_seqlens debug output found)"
fi
echo ""

# Check KV write positions (token_indices)
echo "=== KV Write Positions (token_indices) ==="
kv_positions=$(grep "\[GPU KV WRITE L0\] token_indices" "$LOG_FILE" | head -10)
if [ -n "$kv_positions" ]; then
    echo "$kv_positions"
else
    echo "(No KV write debug output found - is BATCHGEN_DEBUG_KV_WRITE=1 set?)"
fi
echo ""

# Check for duplicates (BUG indicator)
echo "=== Duplicate Position Check ==="
if grep -q "\[GPU KV WRITE L0\] token_indices" "$LOG_FILE"; then
    dup_lines=$(grep "\[GPU KV WRITE L0\] token_indices" "$LOG_FILE" | sort | uniq -d)
    dup_count=$(echo "$dup_lines" | grep -c . || echo "0")

    if [ "$dup_count" -gt 0 ] && [ -n "$dup_lines" ]; then
        echo "❌ BUG CONFIRMED: Found duplicate KV write positions!"
        echo ""
        echo "Duplicate lines:"
        echo "$dup_lines"
        echo ""
        echo "This confirms cache_seqlens is NOT being updated between decode iterations."
        echo "Fix: Update cache_seqlens after each token generation in batchgen_worker.py"
        exit 1
    else
        echo "✓ No duplicate KV write positions detected"
        echo ""
        echo "KV cache positioning appears correct."
    fi
else
    echo "⚠ Could not verify - no KV write debug output found"
    echo "Make sure BATCHGEN_DEBUG_KV_WRITE=1 is set"
fi
echo ""

# Check for "WeWeWeWe" pattern in output
echo "=== Repetitive Output Check ==="
wewe_count=$(grep -c "WeWeWe\|WeWe" "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$wewe_count" -gt 0 ]; then
    echo "⚠ Found $wewe_count instances of repetitive 'WeWe' pattern in output"
    echo "Sample:"
    grep "WeWeWe\|WeWe" "$LOG_FILE" | head -2
else
    echo "✓ No obvious repetitive patterns detected"
fi
echo ""

echo "=========================================="
echo "Verification Complete"
echo "=========================================="
