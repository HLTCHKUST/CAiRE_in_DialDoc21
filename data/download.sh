mkdir mrqa
mkdir coqa
mkdir quac
mkdir doqa
cd mrqa
mkdir train
mkdir dev

cd train

wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz -O SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O NaturalQuestions.jsonl.gz

gunzip SQuAD.jsonl.gz
gunzip NewsQA.jsonl.gz
gunzip TriviaQA.jsonl.gz
gunzip SearchQA.jsonl.gz
gunzip HotpotQA.jsonl.gz
gunzip NaturalQuestions.jsonl.gz

cd ../dev

# in-domain
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O NaturalQuestions.jsonl.gz

# out of domain
wget http://participants-area.bioasq.org/MRQA2019/ -O BioASQ.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz -O TextbookQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz -O RelationExtraction.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz -O DROP.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz -O DuoRC.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz -O RACE.jsonl.gz

gunzip SQuAD.jsonl.gz
gunzip NewsQA.jsonl.gz
gunzip TriviaQA.jsonl.gz
gunzip SearchQA.jsonl.gz
gunzip HotpotQA.jsonl.gz
gunzip NaturalQuestions.jsonl.gz

gunzip BioASQ.jsonl.gz
gunzip TextbookQA.jsonl.gz
gunzip RelationExtraction.jsonl.gz
gunzip DROP.jsonl.gz
gunzip DuoRC.jsonl.gz
gunzip RACE.jsonl.gz

cd ../../coqa
wget https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json -O coqa-dev-v1.0.json
wget https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json -O coqa-train-v1.0.json

cd ../quac
wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O train_v0.2.json
wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O val_v0.2.json


cd ../doqa
wget http://ixa2.si.ehu.es/convai/doqa-v2.1.zip -O doqa-v2.1.zip
unzip doqa-v2.1.zip
mv doqa-v2.1/doqa_dataset .
mv doqa-v2.1/ir_scenario .