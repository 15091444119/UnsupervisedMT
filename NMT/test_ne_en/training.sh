Src=en
Tgt=ne
DataDir=""
MonoDataset="$Src:$DataDir/train.$src.pth,,;$Tgt:$DataDir/train.$Tgt.pth,,"
ParaDataset="$Src-$Tgt:,$DataDir/test.$Src-$Tgt.XX.pth,$DataDir/valid.$Src-$Tgt.XX.pth"
PretrainedEmb="$DataDir/emb.$Src.mapped.txt,$DataDir/emb.$Tgt.mapped.txt"

python main.py \
    --exp_name test_en_ne \
    --transformer True \
    --n_enc_layers 4 \
    --n_dec_layers 4 \
    --share_enc 3 \
    --share_dec 3 \
    --share_lang_emb False \
    --share_output_emb False \
    --langs '$Src,$Tgt' \
    --n_mono -1 \
    --mono_dataset $MONO_DATASET \
    --para_dataset $PARA_DATASET \
    --mono_directions '$Src,$Tgt' \
    --word_shuffle 3 \
    --word_dropout 0.1 \
    --word_blank 0.2 \
    --pivo_directions '$Src-$Tgt-$Src,$Tgt-$Src-$Tgt' \
    --pretrained_emb $PRETRAINED \
    --pretrained_out True \
    --lambda_xe_mono '0:1,100000:0.1,300000:0' \
    --lambda_xe_otfd 1 \
    --otf_num_processes 30 \
    --otf_sync_params_every 1000 \
    --enc_optimizer adam,lr=0.0001 \
    --group_by_size True \
    --batch_size 32 \
    --epoch_size 500000 \
    --stopping_criterion bleu_$Src_$Tgt_valid,10 \
    --freeze_enc_emb False \
    --freeze_dec_emb False \