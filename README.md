# information-retrieval-group  
Group Project for UCL Information Retrieval 2016  

Learning to Rank  (LETOR)

## Data
 - data is MQ2008, already in repo in data folder

## Frameworks
  * [RankLib](https://sourceforge.net/projects/lemur/files/lemur/RankLib-2.1/)
  * Add ranklib/bin/RankLib.jar to CLASSPATH
  * Runing RankLib from command line or terminal  
  	$ java -jar bin/RankLib.jar -train ../data/MQ2008/Fold1/train.txt -test ../data/MQ2008/Fold1/test.txt -validate ../data/MQ2008/Fold1/vali.txt -ranker 6 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel.txt
  * Letor Framework  
	$ git clone https://bitbucket.org/ilps/lerot.git  
	$ cd lerot  
	$ pip install -r requirements.txt  

## Folder Structure

	.  
	├── data  
	├── literature  
	├── poster  
	├── ranklib  
	├── report  
	└── results  
