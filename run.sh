bash_file_name=$(basename $0)
export CUDA_VISIBLE_DEVICES=1
for dataset in "imagenet" 
do
      for seed in 0 1 2
      do
            for tta_method in "Source" "BN" "Tent" "SAR" "CoTTA" "RoTTA" "TRIBE"
            do
            python L-CS.py \
                  -acfg configs/adapter/${dataset}/${tta_method}.yaml \
                  -dcfg configs/dataset/${dataset}.yaml \
                  -ocfg configs/order/${dataset}/0.yaml \
                  SEED $seed \
                  TEST.BATCH_SIZE 64 \
                  bash_file_name $bash_file_name \
                  CORRUPTION.SEVERITY '[5]'
            done
      done
done
