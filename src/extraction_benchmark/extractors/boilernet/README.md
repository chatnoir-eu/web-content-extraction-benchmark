# BoilerNet

This is stripped-down version of the upstream BoilerNet code by Leonhardt et al. with custom model loader logic.

The model `model.h5` was trained on the Google-Trends-2017 train split according to the instructions by the BoilerNet authors:

```console
python3 net/preprocess.py -s googletrends-2017/50-30-100-split -w 1000 -t 50
python3 net/train.py -l 2 -u 256 -d 0.5 -s 256 -e 50 -b 16 --interval 1
```

The checkpoint with the highest test F1 was selected as the final model.

For more information, see: https://github.com/mrjleo/boilernet/
