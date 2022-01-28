# CDRL

# Requirements
+ Python 2.7.5

+ Tensorflow-gpu 1.2.1

+ numpy 1.13.3

+ nltk 3.2.1 

+ [Google Word2Vec](https://code.google.com/archive/p/word2vec/)

# Environment
+ OS: Ubuntu Linux 18.04
+ GPU: NVIDIA GTX1080
+ CUDA: 8.0


# Running

## Pivot feature extraction learning: 
The goal is to automatically capture pos/neg pivots as a bridge across domains based on PNet, which provides the inputs and labels for NPnet. If the pivots are already obtained, you can ignore this step.

```
python extract_pivots.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```
## Feature extraction model training:
PNet and NPnet are jointly trained for cross-domain sentiment classification. When there exists large domain discrepany, it can demonstrate the efficacy of NPnet.

```
python train_cdrl.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```

## CDRL model learning:
after the feature extraction, learn the CDRL and process cross-domain sentiment analysis 
```
python cdrl.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```


# Results

The results are obtained in this implementation.


| Task | CDRL  |<br>
| books - dvd         | 0.9135 |<br>
| books-electronics   | 0.9015 |<br>
| books-kitchen       | 0.9097 |<br>
| dvd-books           | 0.9198 |<br>
| dvd-electronics     | 0.9150 |<br>
| dvd-kitchen         | 0.9102 |<br>
| electronics-books   | 0.9023 |<br>
| electronics-dvd     | 0.9032 |<br>
| electronics-kitchen | 0.9375 |<br>
| kitchen-books       | 0.9182 |<br>
| kitchen-dvd         | 0.8894 |<br>
| kitchen-electronics | 0.9297 |<br>
| Average		      | 0.9125 |<br>


