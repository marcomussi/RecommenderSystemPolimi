import numpy as np
import scipy.sparse as sps
import pandas as pd

from DataReaderWithoutValid import dataReader
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import MultiThreadSLIM_ElasticNet

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from HybridRecommenderBEST import HybridRecommender

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

# NUOVA AGGIUNTA

reader = dataReader()
URM_train = reader.mat_complete
ICM_Art = matTrack_Artist
ICM_Alb = matTrack_Album

recommender = HybridRecommender(URM_train)
item = ItemKNNCFRecommender(URM_train)
user = UserKNNCFRecommender(URM_train)
SLIM = MultiThreadSLIM_ElasticNet(URM_train=URM_train)
item.fit(topK=800, shrink=10, similarity='cosine', normalize=True)
user.fit(topK=70, shrink=22, similarity='cosine', normalize=True)
SLIM.fit(l1_penalty=1e-05, l2_penalty=0, positive_only=True, topK=150, alpha=0.00415637376180466)
recommender.fit(ICM_Art, ICM_Alb, item=item, user=user, SLIM=SLIM, w_itemcf=1.1, w_usercf=0.6, w_cbart=0.3, w_cbalb=0.6, w_slim=0.8, w_svd=0.6)
file = open("SubmissionNEW2.csv","w") 
file.write("playlist_id,track_ids") 
for playlist in targetPlaylistCol[0:]:
    file.write("\n{},{}".format(playlist, " ".join(repr(e) for e in (recommender.recommend(playlist, cutoff=10))))) 
file.close() 

