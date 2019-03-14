
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

        # Creo una lista di tuple (x,y) così composte:
        # x = indice playlist
        # y = indice track contenuta nella playlist y
        playlistColTuples_tot = list(train.apply(tuple, axis=1))
        # Creo un set di tuple (x,y) così composte:
        # x = indice playlist
        # y = indice track contenuta nella playlist y
        playlistColTuples_seq = set(train_seq.apply(tuple, axis=1))
        
        # Estrai le tuple NON sequenziali
        playlistColTuples = list(filter(lambda x: x not in playlistColTuples_seq, playlistColTuples_tot))
        # Estrai le tuple del target che non sono sequenziali
        playlistCol_target_notseq = list(filter(lambda x: x[0] in targetPlaylistCol, playlistColTuples))
        # Estrai la lista di tutte le tracce
        trackCol = tracks.track_id.tolist()
        # Estrai la lista di tutte le playlist NON sequenziali
        playlistCol = [x[0] for x in playlistColTuples]
        # Estrai la lista di tutti le playlist NON sequenziali nel target
        playlistCol_target = [x[0] for x in playlistCol_target_notseq]
        # Estrai la lista di tutte le tracce contenute in playlist NON sequenziali
        tracklistCol = [x[1] for x in playlistColTuples]
        # Estrai la lista di tutte le tracce contenute in playlist NON sequenziali nel target
        tracklistCol_target = [x[1] for x in playlistCol_target_notseq]
        # Estrai la colonne degli album, degli artisti e delle durate
        albumIdCol = tracks.album_id.tolist()  # column ALBUM_ID from tracks.csv
        artistIdCol = tracks.artist_id.tolist()  # column ARTIST_ID from tracks.csv
        #durSecCol = tracks.duration_sec.tolist()  # column DURATION_SEC from tracks.csv
        numTrack = len(trackCol)
        #numPlayList = len(playlistCol)
        # Combina le colonne con gli id degli album e degli artisti
        #albumIdArtistIdCol = albumIdCol + artistIdCol
        # Ritorna il numero di playlists
        number_of_play = max(train.playlist_id.tolist())
        # Ritorna un'array di uno lungo quanto il numero di playlists NON sequenziali        
        numPlaylist_notseq = np.ones(len(playlistColTuples), dtype=int)
        # Crea la URM di playlist+tracce NON sequenziali
        mat_notseq = sps.coo_matrix((numPlaylist_notseq, (playlistCol, tracklistCol)),
                                    shape=(number_of_play + 1, len(trackCol)))
        # Converte in CSR
        mat_notseq = mat_notseq.tocsr()


        # Ritorna una lista di tutte le playlist
        PlaylistColumn = train.playlist_id.tolist()
        # Ritorna una lista delle tracce di tutte le playlist
        trackColumn = train.track_id.tolist()
        # Ritorna un'array di uno lungo quanto il numero di playlists  
        numPlaylist = np.ones(len(PlaylistColumn), dtype=int)
        # Crea la URM di playlist+tracce COMPLETA
        self.mat_complete = sps.coo_matrix((numPlaylist, (PlaylistColumn, trackColumn)),
                                           shape=(number_of_play + 1, len(trackCol)))
        # Converte in CSR
        self.mat_complete = self.mat_complete.tocsr()

        # Ritorna un array di uno lungo quanto il numero di playlist target NON sequenziali
        numPlaylist_notseq_target = np.ones(len(playlistCol_target_notseq), dtype=int)
        # Crea la URM di playlist NON sequenziali contenute nel target
        mat_notseq_target = sps.coo_matrix((numPlaylist_notseq_target, (playlistCol_target, tracklistCol_target)),
                                    shape=(number_of_play + 1, len(trackCol)))
        
        # Estrai le playlist sequenziali
        playlistCol_seq = train_seq.playlist_id.tolist()
        # Estrai il numero di playlist sequenziali
        numPlaylist_seq = len(playlistCol_seq)
        # Estrai le tracce sequenziali
        tracklistCol_seq = train_seq.track_id.tolist()
        # Ritorna un array di uno lungo quanto il numero di playlist target sequenziali
        numPlaylist_seq = np.ones(numPlaylist_seq, dtype=int)
        # Crea la URM di playlist sequenziali
        mat_seq = sps.coo_matrix((numPlaylist_seq, (playlistCol_seq, tracklistCol_seq)),
                                 shape=(number_of_play + 1, len(trackCol)))
        # Converti in CSR
        mat_seq = mat_seq.tocsr()
        
        # Crea una lista da 1 fino al numero di playlist sequenziali
        incremental = [i + 1 for i in range(len(playlistCol_seq))]
        # Ordina la lista in ordine DECRESCENTE
        incremental = list(reversed(incremental))
        # Crea una matrice speciale in cui assegno i valori decrescenti creati prima
        mat_seq_rank = sps.coo_matrix((incremental, (playlistCol_seq, tracklistCol_seq)),
                                      shape=(number_of_play + 1, len(trackCol)))
        # Converti in CSR
        mat_seq_rank = mat_seq_rank.tocsr()
        # Crea un set delle playlist sequenziali
        nonempty_seq = set(playlistCol_seq)

        # Per ogni playlist sequenziale, assegna un peso maggiore alle tracce inserite per prime
        for i in nonempty_seq:
            mask_min = (mat_seq[i] * (mat_seq_rank[i, mat_seq_rank[i].nonzero()[1]].min() - 1))  # the mask with the minimum of each row
            mat_seq_rank[i] = mat_seq_rank[i] - mask_min  # subtract each row, this way the first in playlist will have the highest number
        
        # Crea matrice track-album
        matTrack_Album = sps.coo_matrix(
            ((np.ones(numTrack, dtype=int)), (trackCol, albumIdCol)))  # sparse matrix ROW: track_id COLUMN: album_id
        matTrack_Album = matTrack_Album.tocsr()
        
        # Crea matrice track-artista
        matTrack_Artist = sps.coo_matrix(
            ((np.ones(numTrack, dtype=int)), (trackCol, artistIdCol)))  # sparse matrix ROW: track_id COLUMN: artist_id
        matTrack_Artist = matTrack_Artist.tocsr()
        
        
        URM_train_seq, URM_train, URM_test_seq, URM_test = self.train_test_holdout(mat_notseq_target, mat_seq, mat_seq_rank, nonempty_seq, train_perc=0.8)
        # mat contiene l'URM delle playlist non sequenziali che non sono contenute nel target
        # NB: mat_notseq non ha avuto nessuno split
        mat = mat_notseq - mat_notseq_target
        
        self.ICM_Art = matTrack_Artist
        self.ICM_Alb = matTrack_Album
        
        # Nel train metti:
        # -> URM_train (basato sulle playlist del target), splittato
        # -> URM_train_seq (basato sulle playlist sequenziali), splittato
        # -> mat (basato su tutte le playlist non sequenziali non contenute nel target), non splittato
        self.mat_Train = URM_train + URM_train_seq + mat
        # Contiene il test + test sequenziale
        self.mat_Test = URM_test+URM_test_seq
        # Vuota, in questo caso non ci interessa il valid
        self.mat_Valid = sps.csr_matrix(mat.shape, dtype=int)

    def get_URM_complete(self):
        return self.mat_complete

    def get_URM_train(self):
        return self.mat_Train

    def get_URM_validation(self):
        return self.mat_Valid

    def get_ICM_Art(self):
        return self.ICM_Art

    def get_ICM_Alb(self):
        return self.ICM_Alb

    def get_URM_test(self):
        return self.mat_Test

    def train_test_holdout(self, URM_all, URM_all_seq, URM_all_seq_rank, nonempty_seq, train_perc=0.8):
        # Numero interazioni totali (=numero di non zero in URM_all)
        numInteractions = URM_all.nnz
        # Trasforma URM_all in COO
        URM_all = URM_all.tocoo()
        # Scegli, a caso, tra true e false con una probabilità di train_perc per True e di 1-train_perc per False
        train_mask = np.random.choice([True, False], numInteractions, [train_perc, 1 - train_perc])
        # Metti in URM_train la matrice di train
        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])),
                                   shape=URM_all.shape)
        # Converti in CSR
        URM_train = URM_train.tocsr()
        # Inverti la train_mask
        test_mask = np.logical_not(train_mask)
        # Metti in URM_test quello che non è in URM_train
        URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])),
                                  shape=URM_all.shape)
        # Converti in CSR
        URM_test = URM_test.tocsr()
        # Inizializza URM_train_seq con le stesse dimensioni di URM_all
        URM_train_seq = sps.coo_matrix(URM_all.shape, dtype=int)
        URM_train_seq = URM_train_seq.tocsr()
        # Per ogni playlist sequenziale, prendi il peso massimo della riga e moltiplicalo per 1-trainperc
        # Es: 25 * 0.2 = 5
        # E metti in URM_train_seq tutte le tracce che hanno un peso maggiore di perc, in questo caso 5, per cui 
        # in questa riga avrò 20 canzoni nel train e 5 nel test
        for i in nonempty_seq:
            perc = int(math.ceil(URM_all_seq_rank[i].max() * (1 - train_perc)))

            URM_train_seq[i] = URM_all_seq_rank[i] > perc
        # Crea URM_test_seq come differenza 
        URM_test_seq = URM_all_seq - URM_train_seq
        # Ritorna le 4 matrici
        return URM_train_seq, URM_train, URM_test_seq, URM_test

    def train_valid_holdout(self, URM_all, URM_all_seq, URM_all_seq_rank, nonempty_seq, train_perc=0.75, old_perc=0.8):
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

