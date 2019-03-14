import numpy as np
import scipy.sparse as sps
import time, sys
import pandas as pd
import csv
import os
from DataReaderWithoutValid import dataReader
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender


from math import log


# EVALUATION FUNCTION
def precision(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(is_relevant, relevant_items):
    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    n_users = URM_test.shape[0]

    for user_id in range(n_users):

        if user_id % 10000 == 0:
            print("Evaluated user {} of {}".format(user_id, n_users))

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict


def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


class Compute_Similarity_Python:

    def __init__(self, dataMatrix, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine", row_weights=None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions

        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.

        """

        super(Compute_Similarity_Python, self).__init__()

        self.TopK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for paramether 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))

        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns, self.n_columns))

        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Col_weights has {} columns, dataMatrix has {}.".format(len(row_weights),
                                                                                         dataMatrix.shape[0]))

            self.use_row_weights = True
            self.row_weights = row_weights.copy()
            self.row_weights_diag = sps.diags(self.row_weights)

            self.dataMatrix_weighted = self.dataMatrix.T.dot(self.row_weights_diag).T

    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')

        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]

        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row = 0

        blockSize = 1000

        while end_row < self.n_rows:
            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize

    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]

        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col = 0

        blockSize = 1000

        while end_col < self.n_columns:
            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize

    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos = 0

        blockSize = 1000

        while end_pos < len(self.dataMatrix.data):
            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos - start_pos)

            start_pos += blockSize

    def compute_similarity(self, start_col=None, end_col=None, block_size=100):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        values = []
        rows = []
        cols = []

        start_time = time.time()
        start_time_print_batch = start_time
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            self.useOnlyBooleanInteractions()

        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            sumOfSquared = np.sqrt(sumOfSquared)

        if self.asymmetric_cosine:
            sumOfSquared_to_1_minus_alpha = sumOfSquared.power(2 * (1 - self.asymmetric_alpha))
            sumOfSquared_to_alpha = sumOfSquared.power(2 * self.asymmetric_alpha)

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')

        start_col_local = 0
        end_col_local = self.n_columns

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        start_col_block = start_col_local

        this_block_size = 0

        # Compute all similarities for each item using vectorization
        while start_col_block < end_col_local:

            # Add previous block size
            processedItems += this_block_size

            end_col_block = min(start_col_block + block_size, end_col_local)
            this_block_size = end_col_block - start_col_block

            if time.time() - start_time_print_batch >= 30 or end_col_block == end_col_local:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / (end_col_local - start_col_local) * 100, columnPerSec,
                                    (time.time() - start_time) / 60))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_print_batch = time.time()

            # All data points for a given item
            item_data = self.dataMatrix[:, start_col_block:end_col_block]
            item_data = item_data.toarray().squeeze()

            if self.use_row_weights:
                # item_data = np.multiply(item_data, self.row_weights)
                # item_data = item_data.T.dot(self.row_weights_diag).T
                this_block_weights = self.dataMatrix_weighted.T.dot(item_data)

            else:
                # Compute item similarities
                this_block_weights = self.dataMatrix.T.dot(item_data)

            for col_index_in_block in range(this_block_size):

                if this_block_size == 1:
                    this_column_weights = this_block_weights
                else:
                    this_column_weights = this_block_weights[:, col_index_in_block]

                columnIndex = col_index_in_block + start_col_block
                this_column_weights[columnIndex] = 0.0

                # Apply normalization and shrinkage, ensure denominator != 0
                if self.normalize:

                    if self.asymmetric_cosine:
                        denominator = sumOfSquared_to_alpha[
                                          columnIndex] * sumOfSquared_to_1_minus_alpha + self.shrink + 1e-6
                    else:
                        denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6

                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)


                # Apply the specific denominator for Tanimoto
                elif self.tanimoto_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.dice_coefficient:
                    denominator = sumOfSquared[columnIndex] + sumOfSquared + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                elif self.tversky_coefficient:
                    denominator = this_column_weights + \
                                  (sumOfSquared[columnIndex] - this_column_weights) * self.tversky_alpha + \
                                  (sumOfSquared - this_column_weights) * self.tversky_beta + self.shrink + 1e-6
                    this_column_weights = np.multiply(this_column_weights, 1 / denominator)

                # If no normalization or tanimoto is selected, apply only shrink
                elif self.shrink != 0:
                    this_column_weights = this_column_weights / self.shrink

                # this_column_weights = this_column_weights.toarray().ravel()

                if self.TopK == 0:
                    self.W_dense[:, columnIndex] = this_column_weights

                else:
                    # Sort indices and select TopK
                    # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                    # - Partition the data to extract the set of relevant items
                    # - Sort only the relevant items
                    # - Get the original item index
                    relevant_items_partition = (-this_column_weights).argpartition(self.TopK - 1)[0:self.TopK]
                    relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                    top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                    # Incrementally build sparse matrix, do not add zeros
                    notZerosMask = this_column_weights[top_k_idx] != 0.0
                    numNotZeros = np.sum(notZerosMask)

                    values.extend(this_column_weights[top_k_idx][notZerosMask])
                    rows.extend(top_k_idx[notZerosMask])
                    cols.extend(np.ones(numNotZeros) * columnIndex)

            start_col_block += block_size

        # End while on columns

        if self.TopK == 0:
            return self.W_dense

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)

            return W_sparse


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
albumIdArtistIdCol

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


class ItemCBFKNNRecommender(object):

    def __init__(self, URM, ICM_art, ICM_Alb):
        self.URM = URM
        self.ICM_art = ICM_art
        self.ICM_Alb = ICM_Alb
        # self.ICM_Dur = ICM_Dur

    def fit(self, topK=160, shrink=22, normalize=True):

        SLIM = SLIMElasticNetRecommender(URM_train=self.URM)


        self.W_sparse_SLIM = SLIM.fit(l1_penalty=1e-5, l2_penalty=0, positive_only=True, topK=150, alpha=0.004156373761804666)

        similarity_object_CF = Compute_Similarity_Python(self.URM, shrink=10,
                                                         topK=800, normalize=normalize,
                                                         similarity="cosine")

        self.W_sparse_CF = similarity_object_CF.compute_similarity()

        similarity_object_CF_user = Compute_Similarity_Python(self.URM.T, shrink=0,
                                                              topK=400, normalize=normalize,
                                                              similarity="cosine")

        self.W_sparse_CF_user = similarity_object_CF_user.compute_similarity()
        # self.W_sparse_CF_user = normalize(self.W_sparse_CF_user)

        similarity_object_artist = Compute_Similarity_Python(self.ICM_art.T, shrink=5,
                                                             topK=topK, normalize=normalize,
                                                             similarity="cosine")

        self.W_sparse_art = similarity_object_artist.compute_similarity()

        similarity_object_album = Compute_Similarity_Python(self.ICM_Alb.T, shrink=5,
                                                            topK=topK, normalize=normalize,
                                                            similarity="cosine")

        self.W_sparse_alb = similarity_object_album.compute_similarity()

        # similarity_object_dur = Compute_Similarity_Python(self.ICM_Dur.T, shrink=shrink,
        #                                            topK=topK, normalize=normalize,
        #                                           similarity = similarity)

        #  self.W_sparse_dur = similarity_object_dur.compute_similarity()



        nItems = self.URM.shape[1]
        URMidf = sps.lil_matrix((self.URM.shape[0], self.URM.shape[1]))

        for i in range(0, self.URM.shape[0]):
            IDF_i = log(nItems / np.sum(self.URM[i]))
            URMidf[i] = np.multiply(self.URM[i], IDF_i)

        self.URM = URMidf.tocsr()

        self.URM_SLIM = self.URM.dot(self.W_sparse_SLIM)
        self.URM_CF = self.URM.dot(self.W_sparse_CF)
        self.URM_art = self.URM.dot(self.W_sparse_art)
        self.URM_alb = self.URM.dot(self.W_sparse_alb)
        self.URM_CF_user = self.W_sparse_CF_user.dot(self.URM)

        self.URM_final_hybrid = self.URM_CF *  1.25 + self.URM_art * 0.6 + self.URM_alb * 0.5 + self.URM_CF_user * 0.6 + self.URM_SLIM * 0.9

        self.pen_mask = np.ones(self.URM_final_hybrid.shape[1], dtype=int)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product

        scores = np.multiply(self.URM_final_hybrid[user_id].toarray().ravel(), self.pen_mask)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        recs = ranking[:at]



        return recs

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


reader = dataReader()
recommender = ItemCBFKNNRecommender(reader.mat_complete, matTrack_Artist, matTrack_Album)
recommender.fit(shrink=22, topK=160)
with open('Hybrid_with_SLIM.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['playlist_id', 'track_ids'])
    for playlist in targetPlaylistCol[0:]:
        thewriter.writerow([playlist, ' '.join(map(str, recommender.recommend(playlist, at=10)))])