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
        model_name=$(echo "$filename" | sed -e 's,\.index,,' | sed -e 's,saved-models/,,')
        [ "$filename" = "saved-models/*.index" ] && continue
        echo "Checking ${model_name}..."
        [ -f "$filename" ] || {
            echo "  The TensorFlow Saver() might have beaten us to deleting this one."
            continue
        }
        echo "$filename" | grep -q "layer256"
        if [ $? = 0 ]; then
            actor_layer_sizes='[256,256]'
            critic_layer_sizes='[256,256]'
        else
            actor_layer_sizes='[64,64]'
            critic_layer_sizes='[64,64]'
        fi
        echo "$filename" | grep -q -i "\-2d\-"
        if [ $? = 0 ]; then
            model='2D'
        else
            model='3D'
        fi
        echo "  Using a $model simulator model with $actor_layer_sizes network layers."
        out==$(python -m baselines.ddpg.main \
            --model $model \
            --difficulty 0 \
            --reward-shaping \
            --relative-x-pos \
            --nb-eval-steps 1001 \
            --restore-model-name $model_name \
            --actor-layer-sizes $actor_layer_sizes \
            --critic-layer-sizes $critic_layer_sizes \
            --evaluation \
            --eval-only 2>&1)
        echo "$out" | grep -q "Key adaptive_param_noise_actor/LayerNorm/beta not found in checkpoint"
        if [ $? = 0 ]; then
            echo "  This one didn't use adaptive parameter noise. Let's try Ornstein Uhlenbeck."
            out=$(python -m baselines.ddpg.main \
                --model 3D \
                --difficulty 0 \
                --reward-shaping \
                --relative-x-pos \
                --nb-eval-steps 1001 \
                --restore-model-name $model_name \
                --noise-type ou_0.2 \
                --evaluation \
                --eval-only 2>&1)
        fi
        echo "$out" | grep -q "rhs shape\= \[256,19\]"
        if [ $? = 0 ]; then
            echo "  This one is using a bigger neural network. Let's try [256,256]."
            out==$(python -m baselines.ddpg.main \
                --model 3D \
                --difficulty 0 \
                --reward-shaping \
                --relative-x-pos \
                --nb-eval-steps 1001 \
                --restore-model-name $model_name \
                --actor-layer-sizes [256,256] \
                --critic-layer-sizes [256,256] \
                --evaluation \
                --eval-only 2>&1)
        fi
        echo -n "  "
        echo "$out" | grep "Assign requires shapes of both tensors to match" && {
            echo "  Moving it to mismatched-models/"
            mv saved-models/${model_name}.* mismatched-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [6-9][0-9][0-9]' && {
            echo "  Moving it to top-models/"
            mv saved-models/${model_name}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [1-9][0-9][0-9][0-9]' && {
            echo "  Moving it to top-models/"
            mv saved-models/${model_name}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return " | grep ' [1-9].[0-9]e+[0-9][0-9]' && {  # e.g., 1.1e+03
            echo "  Moving it to top-models/"
            mv saved-models/${model_name}.* top-models/
            continue
        }
        echo "$out" | grep "eval/return "
        echo "  Removing it."
        rm saved-models/${model_name}.*
    done
    sleep 10
done

shutdown() {
    interrupt=true
}

trap "shutdown" SIGINT SIGTERM
