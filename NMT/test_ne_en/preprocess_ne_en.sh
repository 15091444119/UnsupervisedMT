set -e
set -u

export CUDA_VISIBLE_DEVICES="5"
DataDir="/home/zhouzh/data/tokenized_data//mono/ne_en_500w_sample"
ParaDir="/home/zhouzh/data/tokenized_data/para/ne-en/"
DestDir="/home/data_ti4_c/zhouzh/low-resource-mt/UNMT_preprocessed_data/en-ne"
NumMerge=30000
FastBpe="/home/data_ti4_c/zhouzh/low-resource-mt/tools/fastBPE/fast"
Word2Vec="/home/data_ti4_c/zhouzh/low-resource-mt/tools/word2vec/word2vec"
VecMap="python3 /home/data_ti4_c/zhouzh/low-resource-mt/tools/vecmap/map_embeddings.py"
Binarize="python3 /home/data_ti4_c/zhouzh/low-resource-mt/UnsupervisedMT/NMT/preprocess.py"
Src=en
Tgt=ne

# check if src <= tgt
if [ $Src \> $Tgt ]; then
    echo "source language $Src > target language $Tgt, wrong data format"
    exit
fi

# check if destdir already exsits
if [ -d $DestDir ];then
    echo "dest dir already exsits, please remove it first"
    exit
else
    echo "DestDir=$DestDir"
    mkdir -p $DestDir
fi


# learn bpe and apply bpe, binarize monolingual validing data
echo -e "\n Preprocess monolingual data \n"
for Lang in "$Src" "$Tgt"; do
    $FastBpe learnbpe $NumMerge $DataDir/train.$Lang > $DestDir/$Lang.codes
    $FastBpe applybpe $DestDir/train.$Lang $DataDir/train.$Lang $DestDir/$Lang.codes
    $FastBpe getvocab $DestDir/train.$Lang > $DestDir/$Lang.vocab
    $Binarize $DestDir/$Lang.vocab $DestDir/train.$Lang
done

# process parallel data
echo -e "\n Preprocess parallel data \n"
for Splt in "valid" "test"; do
    for Lang in "$Src" "$Tgt"; do
        $FastBpe applybpe $DestDir/$Splt.$Src-$Tgt.$Lang $ParaDir/$Splt.$Src-$Tgt.$Lang $DestDir/$Lang.codes
        $Binarize $DestDir/$Lang.vocab $DestDir/$Splt.$Src-$Tgt.$Lang
    done
done

echo -e "\n Learn Word2vec \n"
# word2vec
$Word2Vec -train $DestDir/train.$Src -output $DestDir/emb.$Src.txt -cbow 0 -size 512 -window 10 -negative 10 -hs 0 -threads 50 -binary 0 -iter 10
$Word2Vec -train $DestDir/train.$Tgt -output $DestDir/emb.$Tgt.txt -cbow 0 -size 512 -window 10 -negative 10 -hs 0 -threads 50 -binary 0 -iter 10

# vecmap
echo -e "\n Map embeddings \n"
$VecMap $DestDir/emb.$Src.txt $DestDir/emb.$Tgt.txt $DestDir/emb.$Src.mapped.txt $DestDir/emb.$Tgt.mapped.txt --cuda --identical -v
