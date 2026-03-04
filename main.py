from utils import extract_data, read_documents
from similarities import Similarities
from minhash import MinHashLSH
from movielens import MovieLens
from loguru import logger
from pathlib import Path


if __name__ == "__main__":
    zip_path = Path("data/minhash.zip")
    extract_data(zip_path)
    logger.info("Data extracted successfully")
    documents = read_documents(Path("data/minhash/"))
    logger.info("Documents read successfully")
    for doc_name, text in documents.items():
        logger.info(f"Doc Name: {doc_name}, Doc Length: {len(text)}")
    sim = Similarities()
    kgram_data = sim.create_all_kgrams(documents)
    logger.info("Kgrams created successfully")
    similarities = sim.find_similarities(kgram_data)

    logger.info("Part A : Finding Jaccard Similarities")
    logger.info(f"Similarities : {similarities}")
    for kgram_type, sims in similarities.items():
        logger.info(f"{kgram_type}:")
        for pair, sim in sims.items():
            logger.info(f"{pair}: {sim}")
    
    logger.info("Part B: Min Hash Experiment")
    t_values = [20, 60, 150, 300, 600]
    mhl = MinHashLSH()
    minhash_results = mhl.minhash_experiment(kgram_data, t_values)

    logger.info("Min hash result for D1 vs D2 (3-grams)")
    for t, result in minhash_results.items():
        logger.info(f"t={t}: {result}")
    
    logger.info("Part C: LSH Experiment")
    r, b, f_tau, lsh_probs = mhl.lsh_experiment(kgram_data, t=160, tau=0.7)
    logger.info("Optimal LSH parameters r={r}, b={b}")
    logger.info("f_tau={f_tau}")
    logger.info("LSH probalities for document pairs (3-grams):")
    for pair, prob in lsh_probs.items():
        logger.info(f"{pair}: {prob}")
    
    logger.info("Part D: MovieLens Experiments")
    movielens = MovieLens()
    movielens.download_movielens_data()
    movielens.extract_movielens_data()
    user_movies = movielens.read_movielens_data()
    logger.info(f"Loaded {len(user_movies)} users and their movies ratings")

    t_values = [50, 100, 200]
    movielines_minhash_results = mhl.minhash_user_experiment(user_movies, t_values)
    logger.info("MovieLens minhash results (threshold=0.5)")
    for t, result in movielines_minhash_results.items():
        logger.info(f"t={t}: FP={result['avg_false_positives']}, FN={result['avg_false_negatives']}")
    
    configs = {
        "config1": (5, 10, 50),
        "config2": (5, 20, 100),
        "config3": (5, 40, 200),
        "config4": (10, 20, 200)
    }

    movielens_lsh_res_06 = mhl.lsh_user_experiment(user_movies, configs)
    logger.info("MovieLens LSH results (tau=0.6)")
    for config, result in movielens_lsh_res_06.items():
        logger.info(f"{config}: FP={result['avg_false_positives']}, FN={result['avg_false_negatives']}")
    
    movielens_lsh_res_08 = mhl.lsh_user_experiment(user_movies, configs, threshold=0.8)
    logger.info("MovieLens LSH results (tau=0.8)")
    for config, result in movielens_lsh_res_08.items():
        logger.info(f"{config}: FP={result['avg_false_positives']}, FN={result['avg_false_negatives']}")