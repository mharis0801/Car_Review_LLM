# Car_Review_LLM
Analyzing Car Reviews with LLMs
This project leverages pre-trained Hugging Face LLMs to analyze car reviews through sentiment classification, translation, question answering, and summarization. Designed for 

**Project Senario:**
Car-ing is Sharing auto dealership, it enhances customer support and human agent efficiency by automating text analysis tasks.

Features
Sentiment Analysis: Classifies reviews as POSITIVE/NEGATIVE using distilbert-base-uncased-finetuned-sst-2-english.

Translation: Translates English reviews to Spanish via Helsinki-NLP/opus-mt-en-es.

Extractive QA: Answers questions about reviews using deepset/minilm-uncased-squad2.

Summarization: Condenses lengthy reviews with cnicu/t5-small-booksum.

Installation
Prerequisites:

Python 3.6+

pip package manager

```
pip install transformers pandas torch evaluate
```
Usage
Data Preparation
Place your car_reviews.csv and reference_translations.txt in a data/ directory. The CSV should include Review and Class columns.

Sentiment Classification
python
```
from transformers import pipeline
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
predicted_labels = classifier(reviews)
```
Translation
python
```
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translated_review = translator(first_review)[0]['translation_text']
```
Question Answering
python
```
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")
# Tokenize and infer answers from context
```
Summarization
python
```
summarizer = pipeline("summarization", model="cnicu/t5-small-booksum")
summarized_text = summarizer(text_to_summarize)[0]['summary_text']
```
Evaluation
Accuracy/F1 Score: Measures sentiment classification performance.

python
```
accuracy.compute(references=references, predictions=predictions)  # Example output: 0.92
```
BLEU Score: Evaluates translation quality against reference texts.

