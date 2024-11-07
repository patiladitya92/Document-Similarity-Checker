from flask import Flask, render_template, request
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

app = Flask(__name__)

# Define an empty list to store documents
documents = []

# Initialize the Doc2Vec model
model = Doc2Vec(vector_size=50, window=2, min_count=1, workers=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Get documents from the form
        doc1 = request.form['doc1']
        doc2 = request.form['doc2']

        # Preprocess and tokenize the documents
        tokens1 = doc1.lower().split()
        tokens2 = doc2.lower().split()

        # Create TaggedDocument objects
        tagged_doc1 = TaggedDocument(words=tokens1, tags=[0])
        tagged_doc2 = TaggedDocument(words=tokens2, tags=[1])

        # Append the TaggedDocument objects to the documents list
        documents.append(tagged_doc1)
        documents.append(tagged_doc2)

        # Update the model with the new documents
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

        # Get the similarity score after updating the model
        similarity_score = model.dv.similarity(0, 1)

        # Convert similarity score to percentage
        similarity_percentage = round(similarity_score * 100, 2)

        # Adjust the similarity threshold based on your use case
        similarity_threshold = 0.7

        # Determine the category
        if similarity_score > similarity_threshold:
            similarity_category = 'High Similarity'
        else:
            similarity_category = 'Low Similarity'

        return render_template('result.html', doc1=doc1, doc2=doc2, similarity_percentage=similarity_percentage,
                               similarity_category=similarity_category)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
