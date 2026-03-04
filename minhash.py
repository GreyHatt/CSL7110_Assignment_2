import hashlib
from loguru import logger
from similarities import Similarities
import numpy as np

class MinHashLSH:
    def __init__(self):
        pass
    
    def generate_hash_functions(self, t, m=10005):
        return [
            lambda x, seed=i: int(hashlib.md5(f"{x}_{seed}".encode()).hexdigest(), 16) % m
            for i in range(t)
        ]
    
    def compute_minhash_signature(self, kgrams, hash_functions):
        signature = []
        for hash_func in hash_functions:
            min_hash = min(hash_func(kgram) for kgram in kgrams)
            signature.append(min_hash)
        return signature
    
    def minhash_experiment(self, kgrams_data, t_values):
        doc_names = list(kgrams_data.keys())
        results = {}
        
        kgram_type = "char_3_grams"
        for t in t_values:
            logger.info(f"Running min-hash with t={t}")
            hash_functions = self.generate_hash_functions(t)
            d1_signature = self.compute_minhash_signature(
                kgrams_data["D1.txt"][kgram_type], hash_functions
            )
            d2_signature = self.compute_minhash_signature(
                kgrams_data["D2.txt"][kgram_type], hash_functions
            )
            sim = Similarities()
            approx_sim = sim.approximate_similarity(d1_signature, d2_signature)
            exact_sim = sim.jaccard_similarity(
                kgrams_data["D1.txt"][kgram_type], 
                kgrams_data["D2.txt"][kgram_type]
            )
            results[t] = {
                "approximate_similarity": approx_sim,
                "exact_similarity": exact_sim,
                "relative_error": abs(approx_sim - exact_sim),
            }
        return results
    
    def lsh_parameters(self, t, tau):
        best_r, best_b = None, None
        min_diff = float('inf')
        for r in range(1, t+1):
            if t%r == 0:
                b = t // r
                f_tau = 1 - (1 - tau**b)**r
                diff = abs(f_tau - 0.5)
                if diff < min_diff:
                    min_diff = diff
                    best_r, best_b = r, b
        f_tau = 1 - (1 - tau**best_b)**best_r
        return best_r, best_b, f_tau
    
    def lsh_probability(self, similarity, r, b):
        return 1 - (1 - similarity**b)**r
    
    def lsh_experiment(self, kgrams_data, t=160, tau=0.7):
        r, b, f_tau = self.lsh_parameters(t, tau)
        logger.info(f"Optimal LSH parameters: r={r}, b={b}, f_tau={f_tau}")
        hash_functions = self.generate_hash_functions(t)
        signatures = {}
        for doc_name in kgrams_data.keys():
            signatures[doc_name] = self.compute_minhash_signature(
                kgrams_data[doc_name]["char_3_grams"], hash_functions
            )
        doc_names = list(kgrams_data.keys())
        probabilities = {}
        sim = Similarities()
        for i, doc1 in enumerate(doc_names):
            for j, doc2 in enumerate(doc_names[i+1:], i+1):
                exact_sim = sim.jaccard_similarity(
                    kgrams_data[doc1]["char_3_grams"],
                    kgrams_data[doc2]["char_3_grams"]
                )
                lsh_prob = self.lsh_probability(exact_sim, r, b)
                probabilities[f"{doc1}_vs_{doc2}"] = lsh_prob
        return r, b, f_tau, probabilities
    
    def compute_exact_user_similarities(self, user_movies, threshold):
        users = list(user_movies.keys())
        similar_pairs = []
        sim = Similarities()
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users[i+1:], i+1):
                user_sim = sim.jaccard_similarity(
                    user_movies[user1],
                    user_movies[user2]
                )
                if user_sim >= threshold:
                    similar_pairs.append((user1, user2, user_sim))
        return similar_pairs

    
    def minhash_user_experiment(self, user_movies, t_values, threshold=0.5, runs=5):
        sim = Similarities()
        users = list(user_movies.keys())
        results = {}

        exact_similarities = self.compute_exact_user_similarities(user_movies, threshold)
        exact_pairs = set((u1, u2) for u1, u2, _ in exact_similarities)

        for t in t_values:
            logger.info(f"Running MovieLens minhas with t={t}")
            false_positives = []
            false_negatives = []
            
            for run in range(runs):
                hash_functions = self.generate_hash_functions(t)
                signatures = {}
                for user_id in users:
                    signatures[user_id] = self.compute_minhash_signature(
                        user_movies[user_id],
                        hash_functions
                    )
                approx_pairs = set()
                for i, user1 in enumerate(users):
                    for j, user2 in enumerate(users[i+1:], i+1):
                        approx_sim = sim.approximate_similarity(
                            signatures[user1],
                            signatures[user2]
                        )
                        if approx_sim >= threshold:
                            approx_pairs.add((user1, user2))
                fp = len(approx_pairs - exact_pairs)
                fn = len(exact_pairs - approx_pairs)
                false_positives.append(fp)
                false_negatives.append(fn)
            results[t] = {
                "avg_false_positives": np.mean(false_positives),
                "avg_false_negatives": np.mean(false_negatives),
                "exact_pairs": len(exact_pairs),
                "approx_pairs_avg": np.mean([len(approx_pairs) for _ in range(runs)]),
            }
        return results
    
    def lsh_user_experiment(self, user_movies, configs, threshold=0.6, runs=5):
        users = list(user_movies.keys())
        results = {}
        
        exact_similarities = self.compute_exact_user_similarities(user_movies, threshold)
        exact_pairs = set((u1, u2) for u1, u2, _ in exact_similarities)
        
        for config_name, (r, b, t) in configs.items():
            logger.info(f"Running MovieLens LSH with {config_name}")
            false_positives = []
            false_negatives = []
            
            for run in range(runs):
                hash_functions = self.generate_hash_functions(t)
                signatures = {}
                for user_id in users:
                    signatures[user_id] = self.compute_minhash_signature(
                        user_movies[user_id],
                        hash_functions
                    )
                candidate_pairs = self.lsh_candidate_pairs(signatures, r, b)
                fp = len(candidate_pairs - exact_pairs)
                fn = len(exact_pairs - candidate_pairs)
                false_positives.append(fp)
                false_negatives.append(fn)

            results[config_name] = {
                "avg_false_positives": np.mean(false_positives),
                "avg_false_negatives": np.mean(false_negatives),
                "exact_pairs": len(exact_pairs),
                "candidate_pairs_avg": np.mean([len(candidate_pairs) for _ in range(runs)]),
            }
        return results
    
    def lsh_candidate_pairs(self, signatures, r, b):
        candidate_pairs = set()
        users = list(signatures.keys())
        for i in range(b):
            b_sig = {}
            start = i * r
            end = start + r
            for user_id in users:
                bsig = tuple(signatures[user_id][start:end])
                if bsig not in b_sig:
                    b_sig[bsig] = []
                b_sig[bsig].append(user_id)
            
            for user_ids in b_sig.values():
                if len(user_ids) >= 2:
                    for i in range(len(user_ids)):
                        for j in range(i+1, len(user_ids)):
                            user1, user2 = user_ids[i], user_ids[j]
                            if user1 < user2:
                                candidate_pairs.add((user1, user2))
                            else:
                                candidate_pairs.add((user2, user1))
        return candidate_pairs