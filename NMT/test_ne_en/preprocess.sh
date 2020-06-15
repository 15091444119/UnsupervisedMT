
DataDir="/home/zhouzh/data/XLM_data/mono/ne_en_500w_sample"
ParaDir="/home/user_data55/zhouzh/low-resource_mt/flores/data/wiki_ne_en_bpe5000"
DestDir="/home/data_ti4_c/zhouzh/low-resource-mt/UNMT_preprocessed_data"
NumMerge=30000
FastBpe="/home/data_ti4_c/zhouzh/low-resource-mt/tools/fastBPE/fast"
Word2Vec="/home/data_ti4_c/zhouzh/low-resource-mt/tools/word2vec/word2vec"
VecMap="python3 map_embeddings.py --identical"
Src=en
Tgt=ne

# learn bpe and apply bpe, binarize monolingual validing data
for Lang in "$Src" "$Tgt"; do
    $FastBpe learnbpe $NumMerge $DataDir/valid.$Lang > $DestDir/$Lang_codes
    $FastBpe applybpe $DestDir/valid.$Lang $DataDir/valid.$Lang $DestDir/$Lang_codes
    $FastBpe getvocab $DestDir/valid.$Lang > $DestDir/$Lang_vocab
    $Binarize $DestDir/$Lang_vocab $DestDir/valid.$Lang
done

# process parallel data
for Splt in "valid" "test"; do
    for Lang in "$Src" "$Tgt"; do
        $FastBpe applybpe $DestDir/$Splt.$Src-$Tgt.$Lang $ParaDir/$Splt.$Lang $DestDir/$Lang_codes
        $Binarize $DestDir/$Lang_vocab $DestDir/$Splt.$Src-$Tgt.$Lang
    done
done

# word2vec
$Word2Vec -valid $DestDir/valid.$Src -output $DestDir/emb.$Src.txt -cbow 0 -size 512 -window 10 -negative 10 -hs 0 -sample 1e-4 -threads 50 -binary 0 -min-count 5 -iter 10
$Word2Vec -valid $DestDir/valid.$Tgt -output $DestDir/emb.$Tgt.txt -cbow 0 -size 512 -window 10 -negative 10 -hs 0 -sample 1e-4 -threads 50 -binary 0 -min-count 5 -iter 10

# vecmap
$VecMap $DestDir/emb.$Src.txt $DestDir/emb.$Tgt.txt $DestDir/emb.$Src.mapped.txt $DestDir/emb.$Tgt.mapped.txt