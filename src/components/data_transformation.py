import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomExceptiion
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts" , "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        this function is responsible for tranforming the various types of data
        '''
        try:
            numerical_features = ['writing_score' , 'reading_score']
            categorial_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                "test_preparation_course"
            ]
            
            # handling the missing values and applying the stand scaling
            # fit_tranform the traning dataset and transform on the test dataset

            num_pipeline = Pipeline(
                steps=[
                    ('imputer' , SimpleImputer(strategy='median')),
                    ('scaler' , StandardScaler())
                ],
                verbose=True
            )
            logging.info("Numerical columns pipeline exceuted completed")

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer' , SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding' , OneHotEncoder()),
                    ('scaler' , StandardScaler(with_mean=False))
                ],
                verbose=True
            )

            logging.info("Categorical columns encxoding completed")
            
            # combining both the pipeline
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline' , num_pipeline , numerical_features),
                    ('cat_pipeline' , cat_pipeline , categorial_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomExceptiion(e,sys)

    def inittiate_data_tranformation(self , train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining precossing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_features = ['writing_score' , 'reading_score']
            categorial_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                "test_preparation_course"
            ]

            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name] , axis=1)
            target_feature_test_df = test_df[target_column_name]

            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr , np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomExceptiion(e,sys)