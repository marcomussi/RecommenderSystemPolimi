import pandas as pd
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

#from data.Movielens_10M.Movielens10MReader import Movielens10MReader
#from DataReader import dataReader
from DataReaderWithoutValid import dataReader
import time
from HybridRecommenderPARAM import HybridRecommender
from lightfm import LightFM
from lightfm.evaluation import auc_score
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader
from surprise import Dataset
from surprise import SlopeOne
import implicit
from FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

import traceback, os


if __name__ == '__main__':


    dataReader = dataReader()

    URM_train = dataReader.get_URM_train()
    '''coo = URM_train.tocoo(copy=False)
    df = pd.DataFrame({'playlistId': coo.row, 'trackId': coo.col, 'rating': coo.data}
                 )[['playlistId', 'trackId', 'rating']].sort_values(['playlistId', 'trackId']
                 ).reset_index(drop=True)
    reader = Reader(rating_scale=(0,1))
    data = Dataset.load_from_df(df[['playlistId', 'trackId', 'rating']], reader)
    algo = SlopeOne()
    trainset = data.build_full_trainset()
    algo.fit(trainset)'''
    
    URM_test = dataReader.get_URM_test()

    URM_validation = dataReader.get_URM_validation()
    
    ICM_Art = dataReader.get_ICM_Art()
    ICM_Alb = dataReader.get_ICM_Alb()

    recommender_list = [
        HybridRecommender
        ]


    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, [10], exclude_seen=True)
    evaluatorValid = SequentialEvaluator(URM_validation, [10], exclude_seen=True)

    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")


    for recommender_class in recommender_list:

        try:

            print("Algorithm: {}".format(recommender_class))
            recommender = recommender_class(URM_train)
            
            item = ItemKNNCFRecommender(URM_train)
            W_sparse_CF = item.fit(topK=800, shrink=8, similarity='cosine', normalize=True)
            CFW_weithing = CFW_D_Similarity_Linalg(URM_train, ICM_Art, W_sparse_CF)
            CFW_weithing.fit(topK=8, loss_tolerance=1e-04, add_zeros_quota=0)
            recommender.fit(CFW=CFW_weithing)
            results_run, results_run_string = evaluator.evaluateRecommender(recommender)
            print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()# -*- coding: utf-8 -*-

