# information-retrieval-group  
Group Project for UCL Information Retrieval 2016  

Learning to Rank  (LETOR)

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
	|	 ├── MQ2016  
	|	 │   ├── S1.txt
	|	 │   ├── S2.txt
	|	 │   ├── S3.txt
	|	 │   ├── S4.txt
	|	 │   ├── S5.txt
	|	 │   ├── base512
	|	 │   ├── base1024
	|	 │   ├── base2048
	|	 │   ├── base4096
	|	 │   ├── base8192
	|	 │   ├── base16384
	|	 │   ├── base32768
	|	 │   └── base65536
	├── literature  
	├── poster  
	├── ranklib  
	├── report  
	└── results  

