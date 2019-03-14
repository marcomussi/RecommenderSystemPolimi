
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

from HybridRecommenderPARAM import HybridRecommender

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
            
            CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=URM_train)

            CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=URM_train)

            #self.SVD = PureSVDRecommender(URM_train=self.URM_train)
    
            #self.p3 = RP3betaRecommender(URM_train=self.URM_train)
    
            #self.p3.fit(alpha=0.7091718304597212, beta=0.264005617987932, min_rating=0, topK=150, implicit=False, normalize_similarity=True)
    
            #self.SVD.fit(num_factors=490)

            CBAlb.fit(topK=160, shrink=5, similarity='cosine', normalize=True, feature_weighting="none")

            CBArt.fit(topK=160, shrink=5, similarity='cosine', normalize=True, feature_weighting="none")

            item = ItemKNNCFRecommender(URM_train)

            user = UserKNNCFRecommender(URM_train)

            SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)

            item.fit(topK=1000, shrink=9, similarity='cosine', normalize=True)

            user.fit(topK=300, shrink=2.5, similarity='cosine', normalize=True)
            
            SLIM.fit(l1_penalty=1e-05, l2_penalty=0, positive_only=True, topK=150, alpha=0.00415637376180466)
            w_slimList = [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
            for w_slim in w_slimList:
                recommender.fit(ICM_Art, ICM_Alb, item=item, user=user, SLIM=SLIM, CBAlb=CBAlb, CBArt=CBArt, w_itemcf=4.5, w_usercf=0.6, w_cbart=0, w_cbalb=0, w_slim=w_slim)
    
                results_run, results_run_string = evaluator.evaluateRecommender(recommender)
                print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
                logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
                logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()