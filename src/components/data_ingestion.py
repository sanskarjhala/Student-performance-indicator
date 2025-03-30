import os 
import sys
from src.exception import CustomExceptiion
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts" , "train.csv")
    test_data_path: str = os.path.join("artifacts" , "test.csv")
    raw_data_path: str =os.path.join('artifacts' , "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_cofig = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_cofig.train_data_path) , exist_ok=True)
            df.to_csv(self.ingestion_cofig.raw_data_path , index=False , header=True)

            logging.info("Train test split initated")
            train_set , test_set = train_test_split(df , test_size=0.2 , random_state=42)

            train_set.to_csv(self.ingestion_cofig.train_data_path , index=False , header=True)
            test_set.to_csv(self.ingestion_cofig.test_data_path, index=False , header = True)

            logging.info("Ingestion of the data is Completed")

            return (
                self.ingestion_cofig.train_data_path,
                self.ingestion_cofig.test_data_path
            )

        except Exception as e :
            raise CustomExceptiion(e , sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.inittiate_data_tranformation(train_data_path , test_data_path)