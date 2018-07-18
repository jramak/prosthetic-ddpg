#!/bin/bash

set -e

interrupt=false
while true; do
    if [[ $interrupt = true ]]; then
        break
    fi
    for filename in saved-models/*.index; do
        if [[ $interrupt = true ]]; then
            break
        fi
        model=$(echo "$filename" | sed -e 's,\.index,,' | sed -e 's,saved-models/,,')
        echo "Checking ${model}..."
        [[ -f $filename ]] || {
            echo "  The TensorFlow Saver() might have beaten us to deleting this one."
            continue
        }
        out=$(python -m baselines.ddpg.main \
            --model 3D \
            --difficulty 0 \
            --reward-shaping \
            --relative-x-pos \
            --nb-eval-steps 1001 \
            --restore-model-name $model \
            --evaluation \
            --eval-only)
        echo "$out" | grep "Assign requires shapes of both tensors to match" && {
            echo "  Moving it to mismatched-models/";
            mv saved-models/${model}.* mismatched-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [6-9][0-9][0-9]' && {
            echo "  Moving it to top-models/";
            mv saved-models/${model}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [1-9][0-9][0-9][0-9]' && {
            echo "  Moving it to top-models/";
            mv saved-models/${model}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return "
        echo "  Removing it."
        rm saved-models/${model}.*
    done
    sleep 10
done

shutdown() {
    interrupt=true
}

trap "shutdown" SIGINT SIGTERM
