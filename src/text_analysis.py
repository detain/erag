import os
import re
from collections import defaultdict
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from PyPDF2 import PdfReader
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from src.settings import settings
from src.api_model import EragAPI
from src.look_and_feel import success, info, warning, error
from src.print_pdf import PDFReportGenerator

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def read_file_content(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    else:  # Assume it's a text file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_text(text):
    # Remove Project Gutenberg license and metadata
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1 and end != -1:
        text = text[start + len(start_marker):end]
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, chunk_size):
    sentences = nlp(text).sents
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_text = sentence.text.strip()
        sentence_size = len(sentence_text.split())
        
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence_text)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_summary(erag_api, chunk):
    prompt = f"""Summarize the following text, focusing on key concepts and main ideas:

Text:
{chunk}

Summary:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates concise summaries."},
        {"role": "user", "content": prompt}
    ]
    response = erag_api.chat(messages, temperature=settings.temperature)
    return response.strip()

def extract_key_concepts(doc, n=30):
    concept_freq = defaultdict(int)
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop:
            concept_freq[token.lemma_] += 1
    
    return sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:n]

def create_concept_graph(text, n=20):
    doc = nlp(text)
    concepts = extract_key_concepts(doc, n)
    
    G = nx.Graph()
    for concept, freq in concepts:
        G.add_node(concept, weight=freq)
    
    window_size = 5
    for i in range(len(doc) - window_size):
        window = doc[i:i+window_size]
        window_concepts = [token.lemma_ for token in window if token.lemma_ in dict(concepts)]
        for j in range(len(window_concepts)):
            for k in range(j+1, len(window_concepts)):
                if G.has_edge(window_concepts[j], window_concepts[k]):
                    G[window_concepts[j]][window_concepts[k]]['weight'] += 1
                else:
                    G.add_edge(window_concepts[j], window_concepts[k], weight=1)
    
    return G

def visualize_concept_graph(G, output_path):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    node_sizes = [G.nodes[node]['weight'] * 100 for node in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=node_sizes,
            font_size=8, font_weight='bold', edge_color='gray', width=edge_weights)
    
    plt.title("Concept Relationship Graph")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_word_cloud(text, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', max_font_size=100, min_font_size=10).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def perform_topic_modeling(text_chunks, num_topics=5):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_chunks)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics.append((topic_idx, top_words))
    
    return topics

def extract_definitions(text):
    patterns = [
        r'(?P<term>\w+)\s+is defined as\s+(?P<definition>[^.]+)',
        r'(?P<term>\w+)\s+refers to\s+(?P<definition>[^.]+)',
        r'(?P<term>\w+):\s+(?P<definition>[^.]+)',
    ]
    
    glossary = defaultdict(list)
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            term = match.group('term').lower()
            definition = match.group('definition').strip()
            glossary[term].append(definition)
    
    return glossary

def run_text_analysis(file_path, erag_api):
    content = read_file_content(file_path)
    content = clean_text(content)
    chunks = chunk_text(content, settings.file_chunk_size)
    
    output_folder = os.path.join(settings.output_folder, "text_analysis_output")
    os.makedirs(output_folder, exist_ok=True)
    
    print(info("Generating summary and analyzing text..."))
    summaries = []
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks)):
        summary = generate_summary(erag_api, chunk)
        summaries.append(summary)
    
    combined_summary = "\n\n".join(summaries)
    
    print(info("Creating concept graph..."))
    G = create_concept_graph(content)
    graph_output = os.path.join(output_folder, "concept_graph.png")
    visualize_concept_graph(G, graph_output)
    
    print(info("Generating word cloud..."))
    word_cloud_output = os.path.join(output_folder, "word_cloud.png")
    generate_word_cloud(content, word_cloud_output)
    
    print(info("Performing topic modeling..."))
    topics = perform_topic_modeling(chunks)
    
    print(info("Extracting glossary..."))
    glossary = extract_definitions(content)
    
    # Prepare results for PDF report
    results = {
        "summary": combined_summary,
        "concept_graph": graph_output,
        "word_cloud": word_cloud_output,
        "topics": topics,
        "glossary": glossary
    }
    
    # Generate PDF report
    pdf_generator = PDFReportGenerator(output_folder, erag_api.model, os.path.basename(file_path))
    pdf_content = []
    
    # Summary
    pdf_content.append(("Summary", [], combined_summary))
    
    # Concept Graph
    pdf_content.append(("Concept Graph", [("Concept Relationship Graph", graph_output)], "The concept graph shows the relationships between key concepts in the text."))
    
    # Word Cloud
    pdf_content.append(("Word Cloud", [("Word Cloud", word_cloud_output)], "The word cloud visualizes the most frequent words in the text."))
    
    # Topics
    topics_text = "\n".join([f"Topic {idx}: " + ", ".join(words) for idx, words in topics])
    pdf_content.append(("Topic Modeling", [], f"The following topics were identified in the text:\n\n{topics_text}"))
    
    # Glossary
    glossary_text = "\n".join([f"{term}: {'; '.join(definitions)}" for term, definitions in glossary.items()])
    pdf_content.append(("Glossary", [], f"The following terms and definitions were extracted from the text:\n\n{glossary_text}"))
    
    # Generate the PDF report
    pdf_file = pdf_generator.create_enhanced_pdf_report(
        [],  # No specific findings for text analysis
        pdf_content,
        [],  # No additional image data
        filename="text_analysis_report",
        report_title=f"Text Analysis Report for {os.path.basename(file_path)}"
    )
    
    if pdf_file:
        print(success(f"PDF report generated successfully: {pdf_file}"))
    else:
        print(error("Failed to generate PDF report"))
    
    return success(f"Text analysis completed. PDF report saved at: {pdf_file}")

if __name__ == "__main__":
    print(warning("This module is not meant to be run directly. Import and use run_text_analysis function in your main script."))