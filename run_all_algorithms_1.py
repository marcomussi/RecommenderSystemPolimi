
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

from HybridRecommenderBEST import HybridRecommender

import traceback, os


if __name__ == '__main__':


    dataReader = dataReader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()

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

            user = UserKNNCFRecommender(URM_train)

            SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)

            item.fit(topK=800, shrink=10, similarity='cosine', normalize=True)

            user.fit(topK=70, shrink=22, similarity='cosine', normalize=True)
            
            SLIM.fit(l1_penalty=1e-05, l2_penalty=0, positive_only=True, topK=150, alpha=0.00415637376180466)
            
            recommender.fit(ICM_Art, ICM_Alb, item=item, user=user, SLIM=SLIM, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.3, w_cbalb=0.6, w_slim=0.8, w_svd=0.6)

            results_run, results_run_string = evaluator.evaluateRecommender(recommender)
            print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()