
class Similarities:
    def __init__(self):
        pass
    
    def create_char_kgrams(self, text, k):
        kgrams = set()
        if len(text) >= k:
            for i in range(len(text) - k +1):
                kgrams.add(text[i:i+k])
        return kgrams
    
    def create_word_kgrams(self, text, k):
        words = text.split()
        kgrams = set()
        if len(words) >= k:
            for i in range(len(words) - k +1):
                kgrams.add(" ".join(words[i:i+k]))
        return kgrams
    
    def create_all_kgrams(self, documents):
        kgrams_data = {}
        for doc_name, text in documents.items():
            kgrams_data[doc_name] = {
                "char_2_grams": self.create_char_kgrams(text, 2),
                "char_3_grams": self.create_char_kgrams(text, 3),
                "word_2_grams": self.create_word_kgrams(text, 2),
            }
        return kgrams_data
    
    def jaccard_similarity(self, set1, set2):
        if not set1 and not set2:
            return 1
        if not set1 or not set2:
            return 0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def find_similarities(self, kgram_data):
        doc_names = list(kgram_data.keys())
        kgram_types = ["char_2_grams", "char_3_grams", "word_2_grams"]
        similarities = {}
        for kgram_type in kgram_types:
            similarities[kgram_type] = {}
            for i, doc1 in enumerate(doc_names):
                for j, doc2 in enumerate(doc_names[i+1:], i+1):
                    set1 = kgram_data[doc1][kgram_type]
                    set2 = kgram_data[doc2][kgram_type]
                    sim = self.jaccard_similarity(set1, set2)
                    similarities[kgram_type][f"{doc1}_{doc2}"] = sim
        return similarities
        