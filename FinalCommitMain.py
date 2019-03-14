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
import traceback, os
from FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg


if __name__ == '__main__':


    dataReader = dataReader()

    #URM_train = dataReader.get_URM_train()
    URM_train = dataReader.get_URM_complete()
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
            
            SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)
            SLIM.fit(l1_penalty=1e-05, l2_penalty=0, positive_only=True, topK=300, alpha=4e-5)
            item = ItemKNNCFRecommender(URM_train)
            W_sparse_CF = item.fit(topK=800, shrink=8, similarity='cosine', normalize=True)
            CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=URM_train)
            CBAlb.fit(topK=10, shrink=16, similarity='cosine', normalize=True, feature_weighting="none")
            CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=URM_train)
            CBArt.fit(topK=12, shrink=16, similarity='cosine', normalize=True, feature_weighting="none")
            user = UserKNNCFRecommender(URM_train)
            user.fit(topK=300, shrink=2, similarity='cosine', normalize=True)
            p3 = RP3betaRecommender(URM_train=URM_train)
            p3.fit(alpha=0.94, beta=0.3, min_rating=0, topK=300, implicit=False, normalize_similarity=True)
            warp_model = LightFM(no_components=100,
                    loss='warp',
                    learning_schedule='adagrad',
                    max_sampled=1000,
                    user_alpha=4e-05,
                    item_alpha=4e-05)
            warp_model.fit(URM_train, epochs=20, num_threads=4, verbose=True)
            CFW_weithing = CFW_D_Similarity_Linalg(URM_train, ICM_Alb, W_sparse_CF)
            CFW_weithing.fit(topK=8, loss_tolerance=1e-04, add_zeros_quota=0)

            recommender.fit(w_CFW = 0.1, CFW=CFW_weithing, ICM_Alb = ICM_Alb, p3=p3, w_p3=1, item=item, w_itemcf=0.05, user=user, w_usercf=0.003, CBArt=CBArt, w_cbart=0.04, CBAlb=CBAlb, w_cbalb=0.13, SLIM=SLIM, w_slim=0.18, warp=warp_model, w_warp=0.05)
            targetPlaylist = pd.read_csv("data/target_playlists.csv")
            targetPlaylistCol = targetPlaylist.playlist_id.tolist()
            file = open("SubmissionWARP.csv","w") 
            file.write("playlist_id,track_ids") 
            for playlist in targetPlaylistCol[0:]:
                file.write("\n{},{}".format(playlist, " ".join(repr(e) for e in (recommender.recommend(playlist, cutoff=10))))) 
            file.close() 

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()