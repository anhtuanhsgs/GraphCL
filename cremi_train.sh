cat cremi_train.sh 
python -W ignore main.py --env 256_s1.5_m0.5_final_render --gpu-id 1 --workers 0 --lbl-agents 0 --valid-gpu 7 \
--num-steps 6 --max-episode-length 6 --reward seg --model AttUNet2 --out-radius 12 18 \
--use-masks --size 256 256 --log-period 1 --features 32 64 128 256 512 --entropy-alpha 0.03 \
--downsample -1 --data 256_cremi --in-radius 0.8 --lr 2e-5 \
--fgbg-ratio 0.7 --st-fgbg-ratio 0.7 --mer_w 0.5 --spl_w 1.5  --save-period 500 --minsize 20 \
--max-temp-steps 99 \
--multi 1 \
--log-dir logs/Mar2020/AdditionalMaterials/ \
--dilate-fac 2 \
--load trained_models/cremi/256_s1.5_m0.5_AttUNet2_masks_seg_cremi/256_s1.5_m0.5_AttUNet2_masks_seg_cremi_5350.dat 

#--load trained_models/256_cremi/256_s0.8_m1.2_cnt3_AttUNet2_masks_seg_256_cremi/256_s0.8_m1.2_cnt3_AttUNet2_masks_seg_256_cremi_5200.dat  

#trained_models/256_cremi/256_s1.0_m1.0_cnt2_AttUNet2_masks_seg_256_cremi/256_s1.0_m1.0_cnt2_AttUNet2_masks_seg_256_cremi_2000.dat 

#trained_models/256_cremi/256_s1.2_m0.8_cnt1_AttUNet2_masks_seg_256_cremi/256_s1.2_m0.8_cnt1_AttUNet2_masks_seg_256_cremi_1100.dat \

 \

