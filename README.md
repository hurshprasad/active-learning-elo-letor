# information-retrieval-group  
Project for UCL Information Retrieval 2016  Learning to Rank  (LETOR)

We are comparing two LETOR models adaRank, and LambdaMart, and then observing another approach to LETOR called ELO Active Learning.

## How to Run the code

### Active Learning (ELO)
- Download pre-processed data from [this link](https://github.com/hurshprasad/information-retrieval-group/releases) file called ELO_ACTIVE_LEARNING_PRE_PROCESSED_DATA.zip  
- place the zip file under your project directory under data/MQ2016/active_learning/pre_processed/  
- run the python file active_learning/elo_active_learning.py  
- we used pycharm (which setup the module paths) [screen shot gif](https://github.com/hurshprasad/information-retrieval-group/RecordingWorkingELO.gif) 

### RankLib Models (AdaRank, LambdaMart)

The following runs AdaRank on our dataset, change -ranker to 6 to run LambdaMart

$ java -jar bin/RankLib.jar -train ../data/MQ2016/base1024/Fold1/train.txt -test ../data/MQ2016/active_learning/test.txt -validate ../data/MQ2016/base1024/Fold1/vali.txt -ranker 3 -metric2t DCG@10  

## Data
 - data is MQ2007
 - Segmented into following folders representing record sizes 2^[9 10 11 12 13 14 15 15] for NDCG@10 comparison to ELO Active Learning 
 	- base512 
 	- base1024
 	- base2048  
 	- base4096  
 	- base8192  
 	- base16384
 	- base32768  
 	- base65536  
 - [Data Description](http://research.microsoft.com/en-us/projects/mslr/default.aspx)  ([further reading](http://arxiv.org/ftp/arxiv/papers/1306/1306.2597.pdf))

| Folds | Training Set | Validation Set | Test Set |
|:-----:|:------------:|:--------------:|:--------:|
| Fold1 |  {S1,S2,S3}  |       S4       |    S5    |
| Fold2 |  {S2,S3,S4}  |       S5       |    S1    |
| Fold3 |  {S3,S4,S5}  |       S1       |    S2    |
| Fold4 |  {S4,S5,S1}  |       S2       |    S3    |
| Fold5 |  {S5,S1,S2}  |       S3       |    S4    |

## Frameworks
  * [RankLib](https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.1/)
  * Add ranklib/bin/RankLib.jar to CLASSPATH
  * [Command Line Parameters](https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/)
  * Runing RankLib from command line or terminal  
    `$ java -jar bin/RankLib.jar -train ../data/MQ2008/Fold1/train.txt -test ../data/MQ2008/Fold1/test.txt -validate ../data/MQ2008/Fold1/vali.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel.txt`
  * Letor Framework  
	$ git clone https://bitbucket.org/ilps/lerot.git  
	$ cd lerot  
	$ pip install -r requirements.txt  

## Folder Structure

	.  
	├── data 
	|	 ├── MQ2016  						# segmented MQ2007 data
	|	 │   ├── S1.txt
	|	 │   ├── S2.txt
	|	 │   ├── S3.txt
	|	 │   ├── S4.txt
	|	 │   ├── S5.txt
	|	 │   ├── base512					# segmented data
	|	 │   ├── base1024
	|	 │   ├── base2048
	|	 │   ├── base4096
	|	 │   ├── base8192
	|	 │   ├── base16384
	|	 │   ├── base32768
	|	 │   ├── base65536
	|	 │   └── active_learning/*		# all pre-processed data
	├── literature  
	├── poster  
	├── ranklib  
	├── report  
	├── results
	└── active_learning					# source code for active learning
		├── __init__.py
		├── constants.py  
		├── elo_active_learning.py
		├── pre_processing.py
		└── util.py

