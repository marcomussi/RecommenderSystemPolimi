from sklearn.preprocessing import normalize
import pandas as pd

from Base.Recommender_utils import check_matrix
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from Base.Recommender import Recommender
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

#from data.Movielens_10M.Movielens10MReader import Movielens10MReader
#from DataReader import dataReader
from DataReaderWithoutValid import dataReader

from math import log
import scipy.sparse as sps
import numpy as np


class HybridRecommender(Recommender):
    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train):

        super(HybridRecommender, self).__init__()

        self.URM_train=URM_train
        self.URM_train = check_matrix(URM_train, 'csr')



    def fit(self, w_CFW = 1, CFW=None, modelImplicit=None, algo=None, ICM_Art=None, ICM_Alb=None, item=None, user=None, SLIM=None, CBArt=None, CBAlb=None, p3=None, SVD=None, warp=None, warp2=None, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.5, w_cbalb=1, w_slim=0.7, w_svd=0.6, w_p3=1.1, w_warp=1, w_warp2=1):
        
        self.w_CFW = w_CFW 
        
        self.CFW = CFW
        
        self.modelImplicit = modelImplicit
        
        self.algo = algo
        
        self.item = item

        self.user = user

        self.SLIM = SLIM
        
        self.SVD = SVD
        
        self.ICM_Alb = ICM_Alb
        #self.CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=self.URM_train)

        #self.CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=self.URM_train)

        #self.SVD = PureSVDRecommender(URM_train=self.URM_train)

        #self.p3 = RP3betaRecommender(URM_train=self.URM_train)

        #self.p3.fit(alpha=0.7091718304597212, beta=0.264005617987932, min_rating=0, topK=150, implicit=False, normalize_similarity=True)

        #self.SVD.fit(num_factors=490)

        self.CBAlb = CBAlb
        
        self.CBArt = CBArt
        
        self.warp = warp
                
        self.w_warp = w_warp
                
        self.w_itemcf = w_itemcf

        self.w_usercf = w_usercf

        self.w_cbart = w_cbart

        self.w_cbalb = w_cbalb

        self.w_slim = w_slim

        self.w_svd = w_svd

        self.w_p3 = w_p3
        
        self.p3 = p3
        
        tracks = pd.read_csv("data/tracks.csv")
        self.trackCol = tracks.track_id.tolist()

        # nItems = self.URM_train.shape[1]
        # URMidf = sps.lil_matrix((self.URM_train.shape[0], self.URM_train.shape[1]))
        #
        # for i in range(0, self.URM_train.shape[0]):
        #     IDF_i = log(nItems / np.sum(self.URM_train[i]))
        #     URMidf[i] = np.multiply(self.URM_train[i], IDF_i)
        #
        # self.URM_train = URMidf.tocsr()

    def compute_item_score(self, user_id):
        #return self.item.compute_item_score(user_id) * self.w_itemcf + self.CBAlb.compute_item_score(user_id)*self.w_cbalb + self.CBArt.compute_item_score(user_id)*self.w_cbart + self.user.compute_item_score(user_id)*self.w_usercf + self.p3.compute_item_score(user_id)*self.w_p3 +self.SLIM.compute_item_score(user_id)*self.w_slim
        #return self.item.compute_item_score(user_id) * self.w_itemcf
        #return self.CBAlb.compute_item_score(user_id)*self.w_cbalb
        #return self.CBArt.compute_item_score(user_id)*self.w_cbart
        #return self.user.compute_item_score(user_id)*self.w_usercf
        #return self.p3.compute_item_score(user_id)*self.w_p3
        #return self.SLIM.compute_item_score(user_id)*self.w_slim
        #return self.SLIM2.compute_item_score(user_id)*self.w_slim2
        
        #return self.item.compute_item_score(user_id) * self.w_itemcf + self.p3.compute_item_score(user_id)*self.w_p3 + self.user.compute_item_score(user_id)*self.w_usercf + self.CBArt.compute_item_score(user_id)*self.w_cbart + self.CBAlb.compute_item_score(user_id)*self.w_cbalb + self.SLIM.compute_item_score(user_id)*self.w_slim + self.CFW.compute_item_score(user_id)*self.w_CFW
        #return self.CFW.compute_item_score(user_id) * self.CFW
        #Test:
        '''
        temp=[]
        for user in user_id:
            arr = np.array([])
            for item in self.trackCol:
                x, _, _ = self.modelImplicit.explain(user, self.URM_train, item)
                arr = np.concatenate([arr, [x]])
            temp.append(arr)
        print(temp)
        return np.array(temp)
        
        #for user in user_id:
        #    for item in self.trackCol:
        #        print(self.algo.predict(user,item))
        '''
        # Usa questo:
        temp = []
        for user in user_id:
            temp.append(self.warp.predict(user_ids=user,  item_ids=self.trackCol, num_threads=4))
        warpPrediction = np.array(temp)

        return self.item.compute_item_score(user_id) * self.w_itemcf + self.p3.compute_item_score(user_id)*self.w_p3 + self.user.compute_item_score(user_id)*self.w_usercf + self.CBArt.compute_item_score(user_id)*self.w_cbart + self.CBAlb.compute_item_score(user_id)*self.w_cbalb + self.SLIM.compute_item_score(user_id)*self.w_slim + warpPrediction*self.w_warp + self.CFW.compute_item_score(user_id)*self.w_CFW
        #return warpPrediction*self.w_warp
        
    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        
        dictionary_to_save = {"sparse_weights": self.sparse_weights}
        
        
        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))


        print("{}: Saving complete".format(self.RECOMMENDER_NAME))