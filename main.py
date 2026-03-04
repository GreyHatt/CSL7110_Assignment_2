from utils import extract_data, read_documents
from similarities import Similarities
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