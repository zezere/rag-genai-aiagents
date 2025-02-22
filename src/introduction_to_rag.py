import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)  # Used in Step 1 instead of transformers
import faiss  # Used in Step 2
from transformers import AutoModelForCausalLM  # Used in Step 3
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # Changed model class

# Sample dataset with facts about Berlin
documents = [
    "Berlin is the capital and largest city of Germany by both area and population.",
    "Berlin is known for its art scene and modern landmarks like the Berliner Philharmonie.",
    "The Berlin Wall, which divided the city from 1961 to 1989, was a significant Cold War symbol.",
    "Berlin has more bridges than Venice, with around 1,700 bridges.",
    "The city's Zoological Garden is the most visited zoo in Europe and one of the most popular worldwide.",
    "Berlin's Museum Island is a UNESCO World Heritage site with five world-renowned museums.",
    "The Reichstag building houses the German Bundestag (Federal Parliament).",
    "Berlin is famous for its diverse architecture, ranging from historic buildings to modern structures.",
    "The Berlin Marathon is one of the world's largest and most popular marathons.",
    "Berlin's public transportation system includes buses, trams, U-Bahn (subway), and S-Bahn (commuter train).",
    "The Brandenburg Gate is an iconic neoclassical monument in Berlin.",
    "Berlin has a thriving startup ecosystem and is considered a major tech hub in Europe.",
    "The city hosts the Berlinale, one of the most prestigious international film festivals.",
    "Berlin has more than 180 kilometers of navigable waterways.",
    "The East Side Gallery is an open-air gallery on a remaining section of the Berlin Wall.",
    "Berlin's Tempelhofer Feld, a former airport, is now a public park and recreational area.",
    "The TV Tower at Alexanderplatz offers panoramic views of the city.",
    "Berlin's Tiergarten is one of the largest urban parks in Germany.",
    "Checkpoint Charlie was a famous crossing point between East and West Berlin during the Cold War.",
    "Berlin is home to numerous theaters, including the Berliner Ensemble and the Volksbühne.",
    "The Berlin Philharmonic Orchestra is one of the most famous orchestras in the world.",
    "Berlin has a vibrant nightlife scene, with countless bars, clubs, and music venues.",
    "The Berlin Cathedral is a major Protestant church and a landmark of the city.",
    "Charlottenburg Palace is the largest palace in Berlin and a major tourist attraction.",
    "Berlin's Alexanderplatz is a large public square and transport hub in central Berlin.",
    "Berlin is known for its street art, with many murals and graffiti artworks around the city.",
    "The Gendarmenmarkt is a historic square in Berlin featuring the Konzerthaus, French Cathedral, and German Cathedral.",
    "Berlin has a strong coffee culture, with numerous cafés throughout the city.",
    "The Berlin TV Tower is the tallest structure in Germany, standing at 368 meters.",
    "Berlin's KaDeWe is one of the largest and most famous department stores in Europe.",
    "The Berlin U-Bahn network has 10 lines and serves 173 stations.",
    "Berlin has a population of over 3.6 million people.",
    "The city of Berlin covers an area of 891.8 square kilometers.",
    "Berlin has a temperate seasonal climate.",
    "The Berlin International Film Festival, also known as the Berlinale, is one of the world's leading film festivals.",
    "Berlin is home to the Humboldt University, founded in 1810.",
    "The Berlin Hauptbahnhof is the largest train station in Europe.",
    "Berlin's Tegel Airport closed in 2020, and operations moved to Berlin Brandenburg Airport.",
    "The Spree River runs through the center of Berlin.",
    "Berlin is twinned with Los Angeles, California, USA.",
    "The Berlin Botanical Garden is one of the largest and most important botanical gardens in the world.",
    "Berlin has over 2,500 public parks and gardens.",
    "The Victory Column (Siegessäule) is a famous monument in Berlin.",
    "Berlin's Olympic Stadium was built for the 1936 Summer Olympics.",
    "The Berlin State Library is one of the largest libraries in Europe.",
    "The Berlin Dungeon is a popular tourist attraction that offers a spooky look at the city's history.",
    "Berlin's economy is based on high-tech industries and the service sector.",
    "Berlin is a major center for culture, politics, media, and science.",
    "The Berlin Wall Memorial commemorates the division of Berlin and the victims of the Wall.",
    "The city has a large Turkish community, with many residents of Turkish descent.",
    "Berlin's Mauerpark is a popular park known for its flea market and outdoor karaoke sessions.",
    "The Berlin Zoological Garden is the oldest zoo in Germany, opened in 1844.",
    "Berlin is known for its diverse culinary scene, including many vegan and vegetarian restaurants.",
    "The Berliner Dom is a baroque-style cathedral located on Museum Island.",
    "The DDR Museum in Berlin offers interactive exhibits about life in East Germany.",
    "Berlin has a strong cycling culture, with many dedicated bike lanes and bike-sharing programs.",
    "Berlin's Tempodrom is a multi-purpose event venue known for its unique architecture.",
    "The Berlinische Galerie is a museum of modern art, photography, and architecture.",
    "Berlin's Volkspark Friedrichshain is the oldest public park in the city, established in 1848.",
    "The Hackesche Höfe is a complex of interconnected courtyards in Berlin's Mitte district, known for its vibrant nightlife and art scene.",
    "Berlin's International Congress Center (ICC) is one of the largest conference centers in the world.",
]


def embed_text(text, model):
    document_embeddings = []
    for i, doc in enumerate(text):
        try:
            embedding = model.encode([doc], convert_to_numpy=True)
            document_embeddings.append(embedding[0])
        except Exception as e:
            print(f"Error processing document {i+1}: {e}")
            continue
    return np.array(document_embeddings)


# Function defined for Step 2
def retrieve(query, model, index, documents, top_k=3):
    # Generate the embedding for the query using the provided model
    # NOTE: no tokenizer in our case (different from Diogo's code),
    # because we are using sentence-transformers library which handles
    # the tokenization internally.
    query_embedding = embed_text([query], model)
    # Search the FAISS index for the top_k most similar documents
    distances, indices = index.search(query_embedding, top_k)
    # Return the most similar documents and their corresponding distances
    return [documents[i] for i in indices[0]], distances[0]


# Function defined for Step 3
# Differently from Diogo's code, this function also joins retrieved docs into a context string,
# so there is no need to do that outside of this function.
def generate_text(query, retrieved_docs, tokenizer, generator, max_length=100):
    # Combine context and query into a prompt
    prompt = f"Context: {' '.join(retrieved_docs)}\n\nQuestion: {query}\n\nAnswer:"
    # Tokenize the input - using batch_encode_plus instead
    inputs = tokenizer.batch_encode_plus(
        [prompt], return_tensors="pt", add_special_tokens=True
    )
    outputs = generator.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Function defined for Step 4
def rag(query, embedding_model, index, documents, tokenizer, generator, top_k):
    # Retrieve the most relevant documents
    retrieved_docs, distances = retrieve(
        query, embedding_model, index, documents, top_k
    )
    generated_answer = generate_text(query, retrieved_docs, tokenizer, generator)
    return generated_answer


# Function defined for Step 5
def is_relevant(distance, threshold=40):
    return distance < threshold


# Also function defined for Step 5
def rag_with_relevance(
    query, embedding_model, index, documents, tokenizer, generator, top_k
):
    retrieved_docs, distances = retrieve(
        query, embedding_model, index, documents, top_k
    )
    relevant_docs = [
        doc for doc, distance in zip(retrieved_docs, distances) if is_relevant(distance)
    ]
    if not relevant_docs:
        return "I am sorry, there is no relevant information"
    return generate_text(query, relevant_docs, tokenizer, generator)


# Function defined for Step 6
def generate_text_with_parameters(
    query, retrieved_docs, tokenizer, generator, max_length=100
):
    # Combine context and query into a prompt
    prompt = f"Context: {' '.join(retrieved_docs)}\n\nQuestion: {query}\n\nAnswer:"
    # Tokenize the input - using batch_encode_plus instead
    inputs = tokenizer.batch_encode_plus(
        [prompt], return_tensors="pt", add_special_tokens=True
    )
    outputs = generator.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.1,  # was 0.7, now we turn it down to 0.1
        top_k=50,
        top_p=0.8,
        repetition_penalty=1.2,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":

    # Use sentence-transformers directly instead of raw transformers
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    model = model.cpu()

    # ====================================================================
    # Step 1. Tokenization and Embeddings
    #
    # This part is different from Diogo's code, because we are using
    # sentence-transformers instead of transformers, due to the issue
    # mentioned in the README.
    # ====================================================================

    print("\n\nTOKENIZATION AND EMBEDDINGS\n")

    document_embeddings = []

    for i, doc in enumerate(documents):
        try:
            # Get embeddings for the document and append to the list
            embedding = model.encode([doc], convert_to_numpy=True)
            document_embeddings.append(embedding[0])
        except Exception as e:
            print(f"Error processing document {i+1}: {e}")
            continue

    # Covert to array because it's more convenient to work with
    document_embeddings = np.array(document_embeddings)
    # Check how it looks
    print(f"Done! Shape of embeddings: {document_embeddings.shape[1]}")
    # And now that this code works very well, we turn it into a function
    # which is above the "__main__"

    # ====================================================================
    # Step 2. Build the Retrieval system with FAISS
    #
    # This code is close to Diogo's.
    # ====================================================================

    print("\n\nBUILD THE RETRIEVAL SYSTEM WITH FAISS\n")

    # Initialize FAISS index for L2 (Euclidean) distance
    # Create an index with dimension equal to the size of the document embeddings
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    # Add the document embeddings to the FAISS index for similarity search
    index.add(document_embeddings)

    # Print some information about the index
    print(f"Number of documents in the index: {index.ntotal}")
    print(f"Dimension of the document embeddings: {index.d}")

    # At this point, we can define the retrieve() function,
    # we do it above the "__main__".
    # Here, we test how it works:
    query = "What is the capital of Germany?"
    print(f"\nTesting with query: {query}")
    retrieved_docs, distances = retrieve(query, model, index, documents)
    print("Retrieved documents:")
    for doc, distance in zip(retrieved_docs, distances):
        print(f"Distance: {distance:.2f}, Document: {doc}")

    # ====================================================================
    # Step 3. Integrating the generative system
    #
    # NOTE: This part was coded using AI very heavily.
    # Tokenizers are back in the game...
    # ====================================================================

    print("\n\nINTEGRATING THE GENERATIVE SYSTEM\n")

    # First load the T5 model and tokenizer
    model_name = "google/flan-t5-small"
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )  # renamed from 'model' to 't5_model'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t5_model.to("cpu")

    # Test the RAG pipeline
    query = "What is Berlin's former airport?"
    print(f"Testing with query: {query}")
    retrieved_docs, distances = retrieve(
        query, model, index, documents, top_k=3
    )  # this uses sentence transformer model
    print("Retrieved relevant documents:")
    for doc, distance in zip(retrieved_docs, distances):
        print(f"Distance: {distance:.2f}, Document: {doc}")

    # We define the generate_text() function above the "__main__",
    # then we use retrieved docs as context that we pass into this function.
    response = generate_text(
        query, retrieved_docs, tokenizer, t5_model
    )  # use t5_model here
    print(f"\nGenerated Response: {response}")

    # ====================================================================
    # Step 4. RAG system
    #
    # Merging retrieval and generation into a single system.
    # ====================================================================

    print("\n\nRAG SYSTEM\n")

    # We start by defining function rag() above the "__main__".
    # And then we test this function
    query = "What is Berlin famous for?"
    print(f"Testing with query: {query}")
    answer = rag(query, model, index, documents, tokenizer, t5_model, top_k=3)
    print(f"\nAnswer: {answer}")

    # Test RAG system with different queries
    queries = [
        "What is the capital of Germany?",
        "What is Berlin famous for?",
        "Who discovered the way to Brazil?",
        "What is the most famous person in Lesotho?",
    ]
    print("\nTesting with different queries\n")
    for query in queries:
        print(f"\nTesting with query: {query}")
        answer = rag(query, model, index, documents, tokenizer, t5_model, top_k=3)
        print(f"\nAnswer: {answer}")

    # ====================================================================
    # Step 5. Improve the relevance
    #
    # Improving the definition of relevance and
    # Improving generation (exploring parameters)
    # ====================================================================

    print("\n\nIMPROVE THE RELEVANCE\n")

    # We start by checking a few different queries
    # Yes we did it in previous step, but now we also look at distances
    for query in queries:
        retrieved_docs, distances = retrieve(query, model, index, documents, top_k=3)
        print(f"Query: {query}")
        print(f"Retrieved documents: {retrieved_docs}")
        print(f"Distances: {distances}\n")

    # Now we define function is_relevant() to check document relevance based on distance
    # Look above __main__ for the definition of this function, as always.
    # Then we will use it to discard irrelevant documents based on a threshold

    # Next, we define a new rag() function, let's call it rag_with_relevance()
    # This function will use the is_relevant() function to discard irrelevant documents
    # based on a threshold.
    # Look above __main__ ....

    # And here we test how this new function works
    for query in queries:
        print(f"\nTesting with query: {query}")
        answer = rag_with_relevance(
            query, model, index, documents, tokenizer, t5_model, top_k=3
        )
        print(f"\nAnswer: {answer}")

    # ====================================================================
    # Step 6. Improve the Generation Systems
    #
    # Outputs can be further improved by exploring parameters.
    # In this step we tweak function generate_text() by adding parameters.
    # ====================================================================

    print("\n\nIMPROVE THE GENERATION SYSTEMS\n")

    # We start by redefining the function for generating text.
    # We call it generate_text_with_parameters(), turn down temperature
    # and add other parameters.
    # Look above __main__ for the definition of this function.

    # We test the new function with the same query as before
    for query in queries:
        print(f"\nTesting with query: {query}")
        answer = generate_text_with_parameters(
            query, retrieved_docs, tokenizer, t5_model
        )
        print(f"\nAnswer: {answer}")
    # NOTE: results are terrible, but it's exactly what Diogo was expecting!
