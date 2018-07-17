#!/bin/bash

for filename in saved-models/*.index; do
	model=$(echo "$filename" | sed -e 's,\.index,,' | sed -e 's,saved-models/,,')
	echo $model
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
		echo "moving saved-models/${model} to mismatched-models/";
		mv saved-models/${model}.* mismatched-models/
		continue
	}
	echo "$out" | grep "eval/return_history" | grep '[6-9][0-9][0-9]' && {
		mv saved-models/${model}.* top-models/
		continue
	}
	echo "$out" | grep "eval/return_history" | grep '[1-9][0-9][0-9][0-9]' && {
		mv saved-models/${model}.* top-models/
		continue
	}
	echo "$out" | grep "eval/return_history"
	echo "removing saved-models/${model}..."
	rm saved-models/${model}.*
done
