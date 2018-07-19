#!/bin/bash

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
        [ "$filename" = "saved-models/*.index" ] && continue
        echo "Checking ${model}..."
        [ -f "$filename" ] || {
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
            --eval-only 2>&1)
        echo "$out" | grep "Key adaptive_param_noise_actor/LayerNorm/beta not found in checkpoint"
        if [ $? = 0 ]; then
            echo "  This one didn't use adaptive parameter noise. Let's try Ornstein Uhlenbeck."
            out=$(python -m baselines.ddpg.main \
                --model 3D \
                --difficulty 0 \
                --reward-shaping \
                --relative-x-pos \
                --nb-eval-steps 1001 \
                --restore-model-name $model \
                --noise-type ou_0.2 \
                --evaluation \
                --eval-only 2>&1)
        fi
        echo "$out" | grep "Assign requires shapes of both tensors to match" && {
            echo "  Moving it to mismatched-models/"
            mv saved-models/${model}.* mismatched-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [6-9][0-9][0-9]' && {
            echo "  Moving it to top-models/"
            mv saved-models/${model}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [1-9][0-9][0-9][0-9]' && {
            echo "  Moving it to top-models/"
            mv saved-models/${model}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [1-9].[0-9]e+[0-9][0-9]' && {  # e.g., 1.1e+03
            echo "  Moving it to top-models/"
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
