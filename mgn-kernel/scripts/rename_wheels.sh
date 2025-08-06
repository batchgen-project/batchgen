#!/usr/bin/env bash
set -ex

WHEEL_DIR="dist"

wheel_files=($WHEEL_DIR/*.whl)
for wheel in "${wheel_files[@]}"; do
    if [[ "$wheel" == *manylinux2014* ]]; then
        echo "Skipping already formatted wheel: $wheel"
        continue
    fi

    intermediate_wheel="${wheel/linux/manylinux2014}"

    new_wheel="${intermediate_wheel}"

    if [[ "$wheel" != "$new_wheel" ]]; then
        echo "Renaming $wheel to $new_wheel"
        mv -- "$wheel" "$new_wheel"
    fi
done

echo "Wheel renaming completed."