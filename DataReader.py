
import numpy as np
import scipy.sparse as sps
import time, sys
import pandas as pd
import csv

from math import log
import math



class dataReader:

    def __init__(self):
        super(dataReader, self).__init__()

        # Lettura degli input
        tracks = pd.read_csv("data/tracks.csv")
        train = pd.read_csv("data/train.csv")
        targetPlaylist = pd.read_csv("data/target_playlists.csv")
        train_seq = pd.read_csv("data/train_sequential.csv")
        
        # Creo una lista di tutte le target playlist
        targetPlaylistCol = targetPlaylist.playlist_id.tolist()

        # Creo una lista di tutte le playlist applicando la funzione tuple a tutte le 
        playlistColTuples_tot = list(train.apply(tuple, axis=1))
        playlistColTuples_seq = set(train_seq.apply(tuple, axis=1))

        playlistColTuples = list(filter(lambda x: x not in playlistColTuples_seq, playlistColTuples_tot))

        playlistCol_target_notseq = list(filter(lambda x: x[0] in targetPlaylistCol, playlistColTuples))

        trackCol = tracks.track_id.tolist()

        playlistCol = [x[0] for x in playlistColTuples]
        playlistCol_target = [x[0] for x in playlistCol_target_notseq]

        tracklistCol = [x[1] for x in playlistColTuples]
        tracklistCol_target = [x[1] for x in playlistCol_target_notseq]
        albumIdCol = tracks.album_id.tolist()  # column ALBUM_ID from tracks.csv
        albumIdCol_unique = list(set(albumIdCol))  # column ALBUM_ID ordered, without replicated elements

        artistIdCol = tracks.artist_id.tolist()  # column ARTIST_ID from tracks.csv
        artistIdCol_unique = list(set(artistIdCol))  # column ARTIST_ID ordered, without replicated elements

        durSecCol = tracks.duration_sec.tolist()  # column DURATION_SEC from tracks.csv
        durSecCol_unique = list(set(durSecCol))  # column DURATION_SEC ordered, without replicated elements

        numTrack = len(trackCol)
        numPlayList = len(playlistCol)

        albumIdArtistIdCol = albumIdCol + artistIdCol
        albumIdArtistIdCol

        number_of_play = max(train.playlist_id.tolist())
        numPlaylist_notseq = len(playlistColTuples)
        numPlaylist_notseq = np.ones(numPlaylist_notseq, dtype=int)
        mat_notseq = sps.coo_matrix((numPlaylist_notseq, (playlistCol, tracklistCol)),
                                    shape=(number_of_play + 1, len(trackCol)))
        mat_notseq = mat_notseq.tocsr()

        PlaylistColumn = train.playlist_id.tolist()
        trackColumn = train.track_id.tolist()
        numPlaylist = len(PlaylistColumn)
        numPlaylist = np.ones(numPlaylist, dtype=int)
        self.mat_complete = sps.coo_matrix((numPlaylist, (PlaylistColumn, trackColumn)),
                                           shape=(number_of_play + 1, len(trackCol)))
        self.mat_complete = self.mat_complete.tocsr()

        numPlaylist_notseq_target = np.ones(len(playlistCol_target_notseq), dtype=int)
        mat_notseq_target = sps.coo_matrix((numPlaylist_notseq_target, (playlistCol_target, tracklistCol_target)),
                                    shape=(number_of_play + 1, len(trackCol)))

        playlistCol_seq = train_seq.playlist_id.tolist()
        numPlaylist_seq = len(playlistCol_seq)
        tracklistCol_seq = train_seq.track_id.tolist()
        numPlaylist_seq = np.ones(numPlaylist_seq, dtype=int)
        mat_seq = sps.coo_matrix((numPlaylist_seq, (playlistCol_seq, tracklistCol_seq)),
                                 shape=(number_of_play + 1, len(trackCol)))
        mat_seq = mat_seq.tocsr()

        incremental = [i + 1 for i in range(len(playlistCol_seq))]
        incremental = list(reversed(incremental))

        mat_seq_rank = sps.coo_matrix((incremental, (playlistCol_seq, tracklistCol_seq)),
                                      shape=(number_of_play + 1, len(trackCol)))
        mat_seq_rank = mat_seq_rank.tocsr()

        nonempty_seq = set(playlistCol_seq)

        for i in nonempty_seq:
            mask_min = (mat_seq[i] * (mat_seq_rank[i, mat_seq_rank[i].nonzero()[1]].min() - 1))  # the mask with the minimum of each row
            mat_seq_rank[i] = mat_seq_rank[i] - mask_min  # subtract each row, this way the first in playlist will have the highest number

        matTrack_Album = sps.coo_matrix(
            ((np.ones(numTrack, dtype=int)), (trackCol, albumIdCol)))  # sparse matrix ROW: track_id COLUMN: album_id
        matTrack_Album = matTrack_Album.tocsr()

        matTrack_Artist = sps.coo_matrix(
            ((np.ones(numTrack, dtype=int)), (trackCol, artistIdCol)))  # sparse matrix ROW: track_id COLUMN: artist_id
        matTrack_Artist = matTrack_Artist.tocsr()

        URM_train_seq, URM_train, URM_test_seq, URM_test = self.train_test_holdout(mat_notseq_target, mat_seq, mat_seq_rank, nonempty_seq, train_perc=0.8)

        mat = mat_notseq - mat_notseq_target

        #URM_train_seq, URM_train, URM_valid_seq, URM_valid = self.train_valid_holdout(URM_train, URM_train_seq, mat_seq_rank, nonempty_seq, train_perc=0.75, old_perc=0.8)
        URM_train_seq, URM_train, URM_valid_seq, URM_valid = self.train_valid_holdout(URM_train, URM_train_seq, mat_seq_rank, nonempty_seq, train_perc=0.8)

        self.ICM_Art = matTrack_Artist
        self.ICM_Alb = matTrack_Album

        # NUOVA AGGIUNTA
        self.mat_Train = URM_train + URM_train_seq + mat
        self.mat_Test = URM_test+URM_test_seq
        self.mat_Valid = URM_valid + URM_valid_seq

    def get_URM_complete(self):
        return self.mat_complete

    def get_URM_train(self):
        return self.mat_Train

    def get_URM_validation(self):
        return self.mat_Valid

    def get_URM_test(self):
        return self.mat_Test

    def get_ICM_Art(self):
        return self.ICM_Art

    def get_ICM_Alb(self):
        return self.ICM_Alb

    def train_test_holdout(self, URM_all, URM_all_seq, URM_all_seq_rank, nonempty_seq, train_perc=0.8):
        numInteractions = URM_all.nnz
        URM_all = URM_all.tocoo()

        train_mask = np.random.choice([True, False], numInteractions, [train_perc, 1 - train_perc])

        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])),
                                   shape=URM_all.shape)
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)

        URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])),
                                  shape=URM_all.shape)
        URM_test = URM_test.tocsr()

        URM_train_seq = sps.coo_matrix(URM_all.shape, dtype=int)
        URM_train_seq = URM_train_seq.tocsr()

        for i in nonempty_seq:
            perc = int(math.ceil(URM_all_seq_rank[i].max() * (1 - train_perc)))

            URM_train_seq[i] = URM_all_seq_rank[i] > perc

        URM_test_seq = URM_all_seq - URM_train_seq

        return URM_train_seq, URM_train, URM_test_seq, URM_test

    def train_valid_holdout(self, URM_all, URM_all_seq, URM_all_seq_rank, nonempty_seq, train_perc=0.85, old_perc=0.9):
        numInteractions = URM_all.nnz
        URM_all = URM_all.tocoo()

        train_mask = np.random.choice([True, False], numInteractions, [train_perc, 1 - train_perc])

        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])),
                                   shape=URM_all.shape)
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)

        URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])),
                                  shape=URM_all.shape)
        URM_test = URM_test.tocsr()

        URM_train_seq = sps.coo_matrix(URM_all.shape, dtype=int)
        URM_train_seq = URM_train_seq.tocsr()

        for i in nonempty_seq:
            perc = int(math.ceil(URM_all_seq_rank[i].max() * (1 - old_perc)))
            newperc = int(math.ceil((URM_all_seq_rank[i].max() - perc) * (1 - train_perc)))
            URM_train_seq[i] = URM_all_seq_rank[i].multiply(URM_all_seq[i]) - (URM_all_seq[i] * perc) > newperc

        URM_test_seq = URM_all_seq - URM_train_seq

        return URM_train_seq, URM_train, URM_test_seq, URM_test

