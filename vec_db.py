from typing import Dict, List, Annotated
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import os

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db", index_file_path = "representers", new_db = True, db_size = None,cluster_number=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if cluster_number is None:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            elif db_size <= 10000:
                self.cluster_number = 300
            elif db_size <= 1000000:
                self.cluster_number = 3000
            elif db_size <= 10000000:
                self.cluster_number = 9500
            elif db_size <= 15_000_000:
                self.cluster_number = 11500
            else:  # db_size > 15000000
                self.cluster_number = 13500
        else:
            self.cluster_number = cluster_number
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self.insert_records(vectors)
        # self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        # mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        # mmap_vectors[:] = vectors[:]
        # mmap_vectors.flush()
        pass

    def _get_num_records(self) -> int: # get the number of records in the db
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        # # will not be used in our case 
        # num_old_records = self._get_num_records()
        # num_new_records = len(rows)
        # full_shape = (num_old_records + num_new_records, DIMENSION)
        # mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        # mmap_vectors[num_old_records:] = rows
        # mmap_vectors.flush()
        # #TODO: might change to call insert in the index, if you need
        # self._build_index()
        print('begin clustering')
        clustering = MiniBatchKMeans(n_clusters=self.cluster_number, n_init=1, verbose=True,batch_size=4096)
        clustering.fit(rows)
        print('after clustering')
        labels = clustering.predict(rows)
        representers = clustering.cluster_centers_
        self.clusters_files(rows, labels)
        self.representers_index(representers)

    def clusters_files(self, rows, labels):
        os.makedirs(self.db_path, exist_ok=True)
        clusters = [open(f"./{self.db_path}/cluster{i}", "ab") for i in range(self.cluster_number)]
        print('before writing clusters')
        for i in range(len(rows)):
            row_data = np.hstack(([i], rows[i])).astype(np.float32).tobytes()
            clusters[labels[i]].write(row_data)
        for f in clusters:
            f.close()
        print('after writing clusters')

    def representers_index(self,  representers):
        print('before writing representers')
        with open(self.index_path, "ab") as rep_file:
            for i in representers:
                rep_file.write(i)
        print('after writing representers')



    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve_old(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):
        scores = []
        num_records = self._get_num_records()
        # here we assume that the row number is the ID of each vector
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores] # return only the row numbers that has the most similarity 

    def retrieve_one_cluster(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        # First, calculate similarity with all representers
        representers = np.fromfile(self.index_path, dtype=np.float32).reshape(self.cluster_number, DIMENSION)
        similarity_with_representers = []

        for i, representer in enumerate(representers):
            similarity = self._cal_score(query, representer)
            similarity_with_representers.append((similarity, i))

        # Get the index of the most similar representer
        most_similar_representer_index = sorted(similarity_with_representers, reverse=True)[0][1]
        
        print(most_similar_representer_index)
        # Now, open the corresponding cluster file
        cluster_file_path = f"./{self.db_path}/cluster{most_similar_representer_index}"
        
        # similarity score for all the points in the cluster 
        scores = []
        with open(cluster_file_path, "rb") as cluster_file:
            while True:
                row_data = cluster_file.read((DIMENSION+1) * ELEMENT_SIZE)
                if not row_data:
                    break
                row = np.frombuffer(row_data, dtype=np.float32)[1:]  
                score = self._cal_score(query, row)
                scores.append(score)

        # Get the top_k rows with the highest similarity
        top_k_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [idx for idx, _ in top_k_indices] 
    

    def retrieve_10_clusters(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        # Calculate similarity with all representers
        representers = np.fromfile(self.index_path, dtype=np.float32).reshape(self.cluster_number, DIMENSION)
        similarity_with_representers = []

        for i, representer in enumerate(representers):
            similarity = self._cal_score(query, representer)
            similarity_with_representers.append((similarity, i))

        # Get indices of the top 10 most similar representers
        top_clusters_indices = sorted(similarity_with_representers, reverse=True)[:10]
        
        print("Top 10 clusters:", [idx for _, idx in top_clusters_indices])


        all_scores = []
        for _, cluster_index in top_clusters_indices:
            cluster_file_path = f"./{self.db_path}/cluster{cluster_index}"

            try:
                # Read the cluster file and calculate similarity for each row in the cluster
                with open(cluster_file_path, "rb") as cluster_file:
                    while True:
                        row_data = cluster_file.read((DIMENSION + 1) * ELEMENT_SIZE)
                        if not row_data:
                            break
                        row = np.frombuffer(row_data, dtype=np.float32)
                        row_id, row_vector = row[0], row[1:]  
                        score = self._cal_score(query, row_vector)
                        all_scores.append((score, row_id))  
            except FileNotFoundError:
                print(f"Cluster file not found: {cluster_file_path}")
                continue

        # get th e top k scores 
        top_k_results = sorted(all_scores, key=lambda x: x[0], reverse=True)[:top_k]
        
        return [row_id for _, row_id in top_k_results]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # Placeholder for index building logic
        pass

    def get_all_rows_from_file(self, file_path=None) -> np.ndarray:
        file_path = file_path or self.db_path
        num_records = self._get_num_records(file_path)
        vectors = np.memmap(file_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def _get_num_records_for_file(self, file_path=None) -> int:
        file_path = file_path or self.db_path
        file_size = os.path.getsize(file_path)
        return file_size // (DIMENSION * ELEMENT_SIZE)
    
    def _write_vectors_to_file(self, vectors: np.ndarray, file_path=None) -> None:
        ## no mode to use append ?? -> will have to use another representation for the vectors 
        file_path = file_path or self.db_path
        mmap_vectors = np.memmap(file_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]  
        mmap_vectors.flush()  



