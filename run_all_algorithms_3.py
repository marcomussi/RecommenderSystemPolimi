from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from DataReaderWithoutValid import dataReader
from HybridRecommenderPARAM import HybridRecommender
from lightfm import LightFM
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
            #URM_trainCOO = URM_train.tocoo
            #print(URM_trainCOO.row)
            #SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)
            SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)
            #SLIM.fit(l1_penalty=1e-05, l2_penalty=0, positive_only=True, topK=300, alpha=4e-5)
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
            CFW_weithing = CFW_D_Similarity_Linalg(URM_train, ICM_Alb, W_sparse_CF)
            CFW_weithing.fit(topK=8, loss_tolerance=1e-04, add_zeros_quota=0)

            warp_model = LightFM(no_components=100,
                    loss='warp',
                    learning_schedule='adagrad',
                    max_sampled=1000,
                    user_alpha=4e-05,
                    item_alpha=4e-05)
            #warp_model.fit(URM_train, item_features=ICM_Alb, epochs=2, num_threads=4, verbose=True)
            warp_model.fit(URM_train, epochs=20, num_threads=4, verbose=True)
            #recommender.fit(warp=warp_model, w_warp=1)
            #recommender.fit(modelImplicit=modelImplicit)
            alphaList = [0, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]
            for alpha in alphaList:
                recommender.fit(w_CFW = alpha, CFW=CFW_weithing, ICM_Alb = ICM_Alb, p3=p3, w_p3=1, item=item, w_itemcf=0.05, user=user, w_usercf=0.003, CBArt=CBArt, w_cbart=0.04, CBAlb=CBAlb, w_cbalb=0.13, SLIM=SLIM, w_slim=0.18, warp=warp_model, w_warp=0.05)
                #recommender.fit(modelImplicit=modelImplicit)
                results_run, results_run_string = evaluator.evaluateRecommender(recommender)
                print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
                logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
                logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()