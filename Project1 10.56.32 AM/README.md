### Project 1 
#### Data-Preprocessing
* Four datasets are used: ionosphere, adult, breast-cancer, machines  
* No changes are made to ionosphere and adult datasets after basic data cleaning  
* Some irrelevant features such as sample ID number from breast-cancer data, and vendor and model names from machines dataset are removed  
* Dtype of 7th feature of breast-cancer dataset is changed from 'object'/'str' to 'int64'  
* Arbitrary binary classes are created for machines dataset (published performance > 50, published performance <=50  

Data preprocessing tasks done:
1. generate data report to provide basic overview of datasets (num. of features, dtypes of features, etc.)
2. eliminate missing, duplicates, and malformed instances from datasets

