from transformers import pipeline

# Initialize models
# GPT-2 for text generation
generator = pipeline("text-generation", model="gpt2")

# DistilBERT for extractive question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ------------------------ Question 1 ------------------------
# GPT-2: Continue a sentence about AI
prompt_1 = "Artificial Intelligence is transforming"
result_1 = generator(prompt_1, max_length=30)
print("\nQ1: Text Generation using GPT-2")
print("Prompt:", prompt_1)
print("Generated:", result_1[0]['generated_text'])

# ------------------------ Question 2 ------------------------
# DistilBERT: Answer a basic factual question from a short context
context_2 = "The capital of India is New Delhi. It is known for its historical landmarks."
question_2 = "What is the capital of India?"
result_2 = qa_pipeline(question=question_2, context=context_2)
print("\nQ2: QA using DistilBERT")
print("Question:", question_2)
print("Answer:", result_2['answer'])

# ------------------------ Question 3 ------------------------
# Date-based extraction from context
context_3 = "The World Health Organization was established on April 7, 1948."
question_3 = "When was WHO established?"
result_3 = qa_pipeline(question=question_3, context=context_3)
print("\nQ3: Date Extraction")
print("Question:", question_3)
print("Answer:", result_3['answer'])

# ------------------------ Question 4 ------------------------
# Geographical location extraction
context_4 = "Mount Everest is located in the Himalayas and lies on the border between Nepal and China."
question_4 = "Where is Mount Everest located?"
result_4 = qa_pipeline(question=question_4, context=context_4)
print("\nQ4: Location-based QA")
print("Question:", question_4)
print("Answer:", result_4['answer'])

# ------------------------ Question 5 ------------------------
# Person/organization-based answer
context_5 = "SpaceX is a private aerospace company founded by Elon Musk in 2002."
question_5 = "Who founded SpaceX?"
result_5 = qa_pipeline(question=question_5, context=context_5)
print("\nQ5: Organization QA")
print("Question:", question_5)
print("Answer:", result_5['answer'])

# ------------------------ Question 6 ------------------------
# Numerical data extraction
context_6 = "Apple Inc. had a revenue of 394.3 billion dollars in the fiscal year 2022."
question_6 = "What was the revenue of Apple in 2022?"
result_6 = qa_pipeline(question=question_6, context=context_6)
print("\nQ6: Numerical Answer Extraction")
print("Question:", question_6)
print("Answer:", result_6['answer'])

# ------------------------ Question 7 ------------------------
# Extracting name of a person from a historical context
context_7 = "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences."
question_7 = "Who was the first woman to win a Nobel Prize?"
result_7 = qa_pipeline(question=question_7, context=context_7)
print("\nQ7: Person-based QA")
print("Question:", question_7)
print("Answer:", result_7['answer'])

# ------------------------ Question 8 ------------------------
# GPT-2 generation: fun fact continuation
prompt_8 = "Did you know that octopuses can"
result_8 = generator(prompt_8, max_length=40)
print("\nQ8: Fun Fact Text Generation")
print("Prompt:", prompt_8)
print("Generated:", result_8[0]['generated_text'])
