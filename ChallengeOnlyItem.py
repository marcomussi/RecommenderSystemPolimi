import numpy as np
import scipy.sparse as sps
import pandas as pd
import time
from DataReaderWithoutValid import dataReader
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from HybridRecommenderPARAM import HybridRecommender
'''
# Any results you write to the current directory are saved as output.
tracks = pd.read_csv("data/tracks.csv")
train = pd.read_csv("data/train.csv")
targetPlaylist = pd.read_csv("data/target_playlists.csv")

targetPlaylistCol = targetPlaylist.playlist_id.tolist()

trackCol = tracks.track_id.tolist()

playlistCol = train.playlist_id.tolist()

tracklistCol = train.track_id.tolist()

albumIdCol = tracks.album_id.tolist()  # column ALBUM_ID from tracks.csv
# albumIdCol.sort()                                   #column ALBUM_ID ordered
albumIdCol_unique = list(set(albumIdCol))  # column ALBUM_ID ordered, without replicated elements

artistIdCol = tracks.artist_id.tolist()  # column ARTIST_ID from tracks.csv
# artistIdCol.sort()                                   #column ARTIST_ID ordered
artistIdCol_unique = list(set(artistIdCol))  # column ARTIST_ID ordered, without replicated elements

durSecCol = tracks.duration_sec.tolist()  # column DURATION_SEC from tracks.csv
# durSecCol.sort()                                      #column DURATION_SEC ordered
durSecCol_unique = list(set(durSecCol))  # column DURATION_SEC ordered, without replicated elements

numTrack = len(trackCol)
numPlayList = len(playlistCol)

albumIdArtistIdCol = albumIdCol + artistIdCol

mat = sps.coo_matrix(((np.ones(numPlayList, dtype=int)), (playlistCol, tracklistCol)))
mat = mat.tocsr()

matTrack_Album = sps.coo_matrix(
    ((np.ones(numTrack, dtype=int)), (trackCol, albumIdCol)))  # sparse matrix ROW: track_id COLUMN: album_id
matTrack_Album = matTrack_Album.tocsr()

matTrack_Artist = sps.coo_matrix(
    ((np.ones(numTrack, dtype=int)), (trackCol, artistIdCol)))  # sparse matrix ROW: track_id COLUMN: artist_id
matTrack_Artist = matTrack_Artist.tocsr()


# matTrack_Dur = sps.coo_matrix(((np.ones(numTrack, dtype=int)), (trackCol, durSecCol)))       #sparse matrix ROW: track_id COLUMN: duration_sec
# matTrack_Dur = matTrack_Dur.tocsr()
'''
# NUOVA AGGIUNTA

reader = dataReader()
URM_train = reader.get_URM_complete()
ICM_Art = reader.get_ICM_Art()
ICM_Alb = reader.get_ICM_Alb()

SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)
start = time.time()
SLIM.fit(l1_penalty=1e-4, l2_penalty=0, positive_only=True, topK=300, alpha=5e-5)
end = time.time()
elapsed = end - start
print("Elapsed time = {}".format(elapsed))
item = ItemKNNCFRecommender(URM_train)
item.fit(topK=1000, shrink=9, similarity='cosine', normalize=True)
CBAlb = ItemKNNCBFRecommender(ICM=ICM_Alb, URM_train=URM_train)
CBAlb.fit(topK=10, shrink=0, similarity='cosine', normalize=True, feature_weighting="none")
CBArt = ItemKNNCBFRecommender(ICM=ICM_Art, URM_train=URM_train)
CBArt.fit(topK=10, shrink=1, similarity='cosine', normalize=True, feature_weighting="none")
user = UserKNNCFRecommender(URM_train)
user.fit(topK=300, shrink=2.5, similarity='cosine', normalize=True)
p3 = RP3betaRecommender(URM_train=URM_train)
p3.fit(alpha=0.975, beta=0.27, min_rating=0, topK=350, implicit=False, normalize_similarity=True)

recommender = HybridRecommender(URM_train)
recommender.fit(item=item, CBAlb=CBAlb, CBArt=CBArt, user=user, p3=p3, SLIM=SLIM, w_itemcf=1, w_cbalb=0.1, w_cbart=0.05, w_usercf=0.125, w_p3=1, w_slim=1.5)
targetPlaylist = pd.read_csv("data/target_playlists.csv")
targetPlaylistCol = targetPlaylist.playlist_id.tolist()
file = open("Submission.csv","w") 
file.write("playlist_id,track_ids") 
for playlist in targetPlaylistCol[0:]:
    file.write("\n{},{}".format(playlist, " ".join(repr(e) for e in (recommender.recommend(playlist, cutoff=10))))) 
file.close() 

