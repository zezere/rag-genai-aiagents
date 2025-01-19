import os  # used to defined directory for NLTK data if it needs downloading
import nltk  # pip install nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # pip install scikit-learn
from rank_bm25 import (
    BM25Okapi,
)  # used in step 7, install it like this: pip install rank_bm25


# Function to tokenize and lowercase words and remove stopwords
# Defined for the first time when working on Step 4: Preprocessing
# and then used in some of the following steps as well
def preprocess(text):
    # Tokenize the text into words and convert to lowercase
    lowercase_words = nltk.word_tokenize(text.lower())
    # Get list of alphanumeric words only (removing punctuation and special characters)
    alphanumeric_words = [word for word in lowercase_words if word.isalnum()]
    # get only words that are not stopwords
    result = [
        word
        for word in alphanumeric_words
        if word not in nltk.corpus.stopwords.words("english")
    ]
    return result


# Function to tokenize, lowercase, and remove stopwords
# The difference from the preprocess function is that it excludes logical operators (AND, OR, NOT)
# This preprocessing function is used in boolean search
def preprocess_for_boolean(text):
    # Tokenize the text into words and convert to lowercase
    lowercase_words = nltk.word_tokenize(text.lower())
    # Get list of alphanumeric words only (removing punctuation and special characters)
    alphanumeric_words = [word for word in lowercase_words if word.isalnum()]
    # Define stopwords while excluding locigal operators (AND, OR, NOT)
    stopwords = set(nltk.corpus.stopwords.words("english")) - {"and", "or", "not"}
    # get only words that are not stopwords
    result = [word for word in alphanumeric_words if word not in stopwords]
    return result


# Function to search documents with TF-IDF
# Defined for the first time when working on Step 5: TF-IDF
def search_tfidf(query, vectorizer, tfidf_matrix):
    # Transform query into TF-IDF vector
    query_vec = vectorizer.transform([query])
    # Compute similarity between query vector and each document vector
    similarity = np.dot(tfidf_matrix, query_vec.T).toarray()
    return similarity


# Function to search documents using boolean search
# Defined for the first time when working on Step 6: BOOLEAN
def search_boolean(query, documents):
    # Preprocess query and split it into tokens
    query_tokens = preprocess_for_boolean(query)
    results = []
    # Iterate over each document
    for idx, doc in enumerate(documents):
        # Preprocess  document and convert it into set of unique tokens
        doc_tokens = preprocess(doc)
        # Initialize a flag to determine if document should be included in results
        include_doc = False
        # Iterate over each token in the query
        for i, token in enumerate(query_tokens):
            # Handle the AND operator
            if token == "and" and i + 1 < len(query_tokens):
                # If AND is found, check if next token is present in document
                include_doc = include_doc and query_tokens[i + 1] in doc_tokens
            # Handle the OR operator
            elif token == "or" and i + 1 < len(query_tokens):
                # If OR is found, check if next token is present in document
                include_doc = include_doc or query_tokens[i + 1] in doc_tokens
            # Handle the NOT operator
            elif token == "not" and i + 1 < len(query_tokens):
                # If NOT is found, exclude the document if next token is present in document
                if query_tokens[i + 1] in doc_tokens:
                    include_doc = False
                    break  # Exit loop early to exclude document
            else:
                # For non-operator tokens, check if they are present in the document
                # If this is the first token, it determines initial inclusion status
                # If it's not the first token, it combines with previous logic using 'OR'
                include_doc = token in doc_tokens or include_doc

        # After processing all tokens, if document should be included, add it to results
        if include_doc:
            results.append((idx, doc))

    # Return list of documents that match query
    return results


# Function to search documents using BM25
def search_bm25(query, bm25):
    # Process query, using the same preprocessing as for TF-IDF
    # because here we don't deal with logical operators (AND, OR, NOT)
    query_tokens = preprocess(query)
    # Get BM25 scores for query against indexed documents
    results = bm25.get_scores(query_tokens)
    return results


# Sample documents about sailing in Croatia
documents = [
    "Sailing in Croatia often includes visiting UNESCO World Heritage sites.",
    "The Makarska Riviera is known for its stunning coastline and sailing opportunities.",
    "Sailing in Croatia offers stunning views of the Adriatic Sea.",
    "The city of Sibenik is home to the impressive St. James's Cathedral, a UNESCO World Heritage site.",
    "The island of Brač is known for its beautiful beaches and great sailing conditions.",
    "Sailing to the island of Rab, known for its medieval old town, is a great experience.",
    "The Pakleni Islands near Hvar are a popular spot for sailing yachts.",
    "The island of Cres is one of the largest in Croatia and a great destination for sailing.",
    "The island of Cres is one of the largest in Croatia and has a diverse wildlife.",
    "The city of Zagreb is the capital of Croatia and offers a mix of modern and historic attractions.",
    "Sailors can experience the traditional Dalmatian way of life in many coastal villages.",
    "The city of Knin is known for its historic fortress and beautiful scenery.",
    "The island of Krk is the largest island in the Adriatic Sea.",
    "Croatia has a rich history dating back to the Roman Empire.",
    "The island of Hvar is a popular destination for celebrities and high-end travelers.",
    "The city of Varazdin is known for its baroque buildings and vibrant cultural scene.",
    "Sailors can enjoy fresh seafood at many coastal restaurants in Croatia.",
    "The coastal town of Senj is known for its carnival and Nehaj Fortress.",
    "The Dalmatian Coast is a famous region for sailing in Croatia.",
    "The Diocletian's Palace in Split is one of the most famous Roman ruins in Croatia.",
    "The island of Dugi Otok is known for its dramatic cliffs and beautiful sailing waters.",
    "The island of Rab is famous for its medieval old town and stunning beaches.",
    "The island of Mljet is home to a national park and is ideal for nature lovers.",
    "Sailors in Croatia can visit ancient Roman ruins in Split.",
    "The city of Kotor, just across the border in Montenegro, is a popular extension for Croatian sailing trips.",
    "The island of Korcula is believed to be the birthplace of Marco Polo.",
    "The town of Motovun in Istria is famous for its film festival.",
    "The city of Nin is known for its ancient salt pans and historic church.",
    "The city of Vukovar is known for its role in the Croatian War of Independence.",
    "Sailing to the island of Lošinj, known for its health tourism, is a relaxing experience.",
    "The island of Hvar is known for its vibrant nightlife and sailing opportunities.",
    "The island of Korčula is believed to be the birthplace of Marco Polo and is a popular sailing destination.",
    "Sailing in Croatia often involves stopping at picturesque fishing villages.",
    "The city of Opatija is known for its grand villas and seaside promenade.",
    "The city of Porec is known for the Euphrasian Basilica, a UNESCO World Heritage site.",
    "The island of Brac is home to the famous Zlatni Rat beach, known for its changing shape.",
    "Sailing around the island of Mljet offers a peaceful and scenic experience.",
    "The Blue Cave on Biševo Island is a must-see for sailors.",
    "The island of Pag is famous for its cheese and nightlife, and is a fun stop for sailors.",
    "The island of Lastovo is one of Croatia's most remote and tranquil destinations.",
    "Sailors can enjoy snorkeling in the clear waters of the Adriatic Sea.",
    "The city of Rovinj is a charming starting point for a sailing trip in Croatia.",
    "Croatia's national parks, like Krka and Plitvice, are ideal for hiking and nature lovers.",
    "The city of Zadar is famous for its sunsets, which sailors can enjoy from the sea.",
    "Croatia has over a thousand islands, each with its unique charm.",
    "Croatia's Adriatic coast is dotted with charming fishing villages.",
    "Sailing to the island of Šolta offers a quiet escape from the more touristy areas.",
    "Sailors in Croatia can explore over 1,200 islands.",
    "Sailing in Croatia provides opportunities to visit ancient fortresses and castles along the coast.",
    "Croatia is famous for its beautiful coastline and crystal-clear waters.",
    "The island of Krk is accessible by bridge and is a popular starting point for sailing trips.",
    "Sailing in Croatia is best enjoyed during the summer months.",
    "The island of Pag is famous for its cheese, which is considered a delicacy.",
    "The medieval town of Rovinj is one of the most picturesque places in Croatia.",
    "The island of Losinj is famous for its health tourism and clean air.",
    "The city of Dubrovnik, with its famous city walls, is a top destination for sailors.",
    "Croatia has beautiful islands perfect for sailing.",
    "Sailing in Croatia allows you to explore hidden coves and bays.",
    "The city of Rijeka is an important cultural and economic center in Croatia.",
    "The island of Mljet has a national park that is perfect for exploring by sailboat.",
    "The Istrian Peninsula is famous for its truffles and gourmet food.",
    "The town of Cavtat is a quieter alternative to nearby Dubrovnik.",
    "The Peljesac Peninsula is known for its vineyards and wine production.",
    "The Dubrovnik Summer Festival is a major cultural event featuring theater, music, and dance performances.",
    "The city of Zadar is famous for its unique Sea Organ, an architectural sound art object.",
    "The beaches in Croatia are among the best in Europe, with many receiving Blue Flag status.",
    "Sailing around the island of Murter gives access to the Kornati Islands National Park.",
    "The city of Trogir, with its historic architecture, is a great place to dock.",
    "Sailing from Split to Dubrovnik offers breathtaking coastal scenery.",
    "The Plitvice Lakes National Park is a UNESCO World Heritage site known for its stunning waterfalls and lakes.",
    "The city of Hvar is one of the sunniest places in Europe and a popular sailing hub.",
    "The city of Dubrovnik is often called the 'Pearl of the Adriatic'.",
    "The city of Osijek is located in the eastern part of Croatia and is known for its Baroque style.",
    "The city of Pula, with its Roman amphitheater, is a unique sailing destination.",
    "The town of Trogir is a UNESCO World Heritage site known for its medieval architecture.",
    "Sailors in Croatia can enjoy local wines at many coastal vineyards.",
    "Croatia has a Mediterranean climate, making it a great destination year-round.",
    "Croatia has a diverse cultural heritage, with influences from Italy, Hungary, and Austria.",
    "The city of Rijeka is an important cultural and historical sailing destination.",
    "The best sailing routes in Croatia include Dubrovnik and Split.",
    "The Brijuni Islands are a national park and a former presidential retreat.",
    "Sailing to the Elaphiti Islands offers a mix of natural beauty and cultural sites.",
    "Croatia is known for its delicious seafood cuisine.",
    "The ancient city of Pula is known for its well-preserved Roman amphitheater.",
    "The city of Opatija is a historical seaside resort that welcomes sailors.",
    "The waters around Croatia are known for being calm and clear, ideal for sailing.",
    "The Kornati Islands National Park is a popular sailing destination in Croatia.",
    "Sailing around the Brijuni Islands offers a glimpse of Croatia's natural beauty and wildlife.",
    "The island of Vis was a military base and was closed to tourism until the 1990s.",
    "The city of Split is a major port and gateway to the Dalmatian islands.",
    "Sailing to the island of Vis provides access to the famous Blue Cave.",
    "The city of Buzet in Istria is known as the 'City of Truffles'.",
    "The city of Karlovac is known for its parks and the rivers that flow through it.",
    "Sailing to the island of Vis allows you to experience a more remote part of Croatia.",
    "The city of Šibenik is a UNESCO World Heritage site and a great stop for sailors.",
    "The town of Samobor is famous for its traditional cream cake called 'kremsnita'.",
    "Sailing around the island of Lastovo provides a more secluded experience.",
    "Croatia's wine regions produce some excellent wines, especially in Istria and Dalmatia.",
    "Many sailors start their Croatian adventure from the city of Zadar.",
    "The coastal town of Makarska is known for its beautiful beaches and lively nightlife.",
]


if __name__ == "__main__":

    # ==============================================================
    # Step 1 (optional)
    # DOWNLOAD NLTK DATA
    #
    # Downloading needs to be done only once, so it is commented out
    # because the data comes together with the repository
    # ==============================================================

    # # Define the path to the NLTK data directory
    # project_root = os.path.dirname(os.path.abspath(__file__))
    # nltk_data_dir = os.path.join(os.path.dirname(project_root), "data", "nltk_data")
    # # Download the 'punkt' tokenizer model
    # nltk.download("punkt", download_dir=nltk_data_dir)
    # # NOTE: I Needed to download punkt_tab as well,
    # # otherwise I was getting an error
    # nltk.download("punkt_tab", download_dir=nltk_data_dir)
    # # Download the 'stopwords' corpus from nltk
    # nltk.download("stopwords", download_dir=nltk_data_dir)

    # End of Step 1
    # ==============================================================

    # ==============================================================
    # Step 2
    # DEFINE PATH TO DATA
    # ==============================================================

    # Same code as above to define NLTK data directory, just thought
    # it is tidier to comment out the WHOLE Step 1 and then repeat
    # directory definition separately
    project_root = os.path.dirname(os.path.abspath(__file__))
    nltk_data_dir = os.path.join(os.path.dirname(project_root), "data", "nltk_data")
    # Let nltk know where to look for the data
    nltk.data.path.append(nltk_data_dir)

    # End of Step 2
    # ==============================================================

    # ==============================================================
    # Step 3
    # TOKENIZATION
    # ==============================================================

    print("\nTOKENIZATION\n")

    # Sample text
    text = "Sailing in Croatia offers stunning views of the Adriatic Sea. The city of Sibenik is home to the impressive St. James's Cathedral, a UNESCO World Heritage site."

    # Tokenize the text into sentences
    # Splits the input text into a list of sentences
    sentences = nltk.sent_tokenize(text)
    print(f"Sentences: {sentences}\n")
    # Notice how the tokenizer handles abbreviations like 'St.'
    # and also single and double quotes in the console output!

    # Tokenize the text into words
    # Splits the input text into a list of words and punctuation marks
    words = nltk.word_tokenize(text)
    print(f"Words: {words}\n")
    # Notice how the tokenizer handles the second sentence :/

    # End of Step 3
    # ==============================================================

    # ==============================================================
    # Step 4
    # PREPROCESSING
    # ==============================================================

    print("\nPREPROCESSING\n")

    # Funtion to preprocess step is defined ABOVE (under imports)
    # Here we apply that function to each document
    preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
    print(f"First 5 preprocessed documents:\n{preprocessed_docs[:5]}\n")

    # End of Step 4
    # ==============================================================

    # ==============================================================
    # Step 5
    # TF-IDF
    #
    # Since outputs are long, I cut them to first 5 elements,
    # but feel free to adjust that in the code
    # ==============================================================

    print("\nTF-IDF\n")

    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Fit vectorizer on preprocessed documents and transform them into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
    print(f"TF-IDF matrix:\n{tfidf_matrix.toarray()}\n")

    # Sample query
    query = "Sailing in Croatia is fun."
    # Transform query into TF-IDF vector
    query_vec = vectorizer.transform([query])
    print(
        f"TF-IDF vector of the query (I only display first 5 and they're all zeroes, I know):\n{query_vec.T.toarray()[:5]}\n"
    )
    # Compute similarity between query vector and each document vector
    similarity = np.dot(tfidf_matrix, query_vec.T).toarray()
    print(
        f"Computed similarity between document and query vectors (first 5):\n{similarity[:5]}\n"
    )

    # Now use tfidf_matrix computed a little bit earlier,
    # and search_tfidf() function defined way above (under imports)
    # to search for documents that match the query
    results = search_tfidf(query, vectorizer, tfidf_matrix)
    # sort the results based on their scores
    sorted_results = sorted(enumerate(results), key=lambda x: x[1][0], reverse=True)
    # Iterate over sorted results to print them nicely
    print(f"First 5 results of TF-IDF search for query '{query}':")
    for i, (idx, score) in enumerate(sorted_results):
        print(f"Score: {score[0]:.2f} => Document: {documents[idx]}")
        if i > 3:
            break

    # End of Step 5
    # ==============================================================

    # ==============================================================
    # Step 6
    # BOOLEAN
    # ==============================================================

    print("\n\nBOOLEAN\n")

    # Sample query
    query = "sailing not Croatia"
    # Perform boolean search using function defined above
    # (if you've been following the code so far then you know where)
    results = search_boolean(query, documents)
    # Iterate over results to print them nicely
    # no sorting here, because boolean search doesn't return scores...
    # I also don't clip results at 5, there aren't too many of them anyway
    print(f"All results of boolean search for query '{query}':")
    for idx, doc in results:
        print(f"Document {idx}: {doc}")

    # End of Step 6
    # ==============================================================

    # ==============================================================
    # Step 7
    # PROBABILISTIC RETRIEVAL MODEL
    # ==============================================================

    print("\n\nPROBABILISTIC RETRIEVAL MODEL\n")

    # Sample query
    query = "top 10 things to do in Croatia"
    # Tokenize each document (using function that does not exclude logical operators)
    tokenized_docs = [preprocess(doc) for doc in documents]
    # Initialize model
    bm25 = BM25Okapi(tokenized_docs)
    # Perform BM25 search using function defined above
    # (if you've been following the code so far then you know where)
    bm25_results = search_bm25(query, bm25)
    # Sort results in descending order of relevance scores
    sorted_bm25_results = np.argsort(bm25_results)[::-1]
    # Iterate over sorted results to print them nicely
    print(f"First 5 results of TF-IDF search for query '{query}':")
    for i, idx in enumerate(sorted_bm25_results):
        print(f"Score: {bm25_results[idx]:.4f} => Document: {documents[idx]}")
        if i > 3:
            break

    # End of Step 7
    # ==============================================================

    print("\n\nEnd of the script")
    print("I hope you found it useful and interesting!\n")
