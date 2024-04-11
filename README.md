# Disease Classification and Medical QA System

This section of the project involves disease classification using machine learning models and a medical question-answering system using the BioGPT language model.

## Introduction

This section performs the following tasks:

1. **Data Preprocessing**: The dataset (Symptom2Disease.csv) containing symptoms and corresponding disease labels is preprocessed. Text cleaning techniques such as removing punctuation and stop words are applied.

2. **Text Classification**: Several machine learning models are trained on the preprocessed text data to classify diseases based on symptoms. The models include Random Forest, XGBoost, AdaBoost, Gradient Boosting, and Support Vector Classifier (SVC). Ensemble learning is also employed to combine the predictions of multiple models.

3. **Recurrent Neural Network (RNN) Models**: Two RNN models, one using GRU and the other using LSTM, are trained on the disease classification task. These models utilize word embeddings to process the text input and predict the disease label.

4. **Medical Question-Answering System**: The BioGPT language model is fine-tuned on medical text data to create a question-answering system. Given a medical query or symptom description, the system generates appropriate responses using the trained model.

5. **Speech Recognition**: The system includes functionality for transcribing audio files containing patient symptom descriptions. This transcribed text is then used as input for disease classification and question-answering.


**Installation of Dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

 

## Usage

1. **Data Preparation**: Prepare your dataset containing symptoms and corresponding disease labels. Ensure the data is cleaned and preprocessed before training the models.

2. **Training Models**: Run the script to train the machine learning models and RNN models on the prepared dataset.

3. **Fine-tuning BioGPT**: Fine-tune the BioGPT language model on medical text data to create the question-answering system.

4. **Testing and Evaluation**: Evaluate the trained models using appropriate metrics. Test the question-answering system with sample medical queries to ensure its functionality.

5. **Integration**: Integrate the disease classification and question-answering components into your medical application or system for real-world use.


# Fine-tuning LLAMA Model on PubMedQA Dataset

This repository contains code for fine-tuning the LLAMA-2 7B model on the PubMedQA dataset, a collection of question-answer pairs derived from PubMed articles, focusing on medical-related queries.

## Introduction

The LLAMA model is a powerful language model developed by NousResearch, specifically tailored for understanding medical text and generating accurate responses to medical queries. Fine-tuning the LLAMA model on a specific dataset like PubMedQA involves training the model on the new dataset to adapt its parameters to the specific task or domain.


## Fine-Tuning Process

1. **Dataset Preparation**: The PubMedQA dataset is loaded using the `datasets` library from Hugging Face. This dataset contains question-answer pairs extracted from PubMed articles.

2. **Model Selection**: The LLAMA model (`NousResearch/Llama-2-7b-chat-hf`) is chosen as the base model for fine-tuning. This model is pre-trained on a large corpus of medical text and has demonstrated strong performance in understanding medical language.

3. **Tokenization**: The selected dataset is tokenized using the LLAMA tokenizer (`AutoTokenizer`) provided by Hugging Face. Tokenization involves breaking down the text into individual tokens, which are then encoded as numerical representations suitable for input to the model.

4. **Data Preparation**: The dataset is processed to merge the question and answer pairs into single text sequences. This step involves concatenating the question and its corresponding long answer from the PubMed article.

5. **Model Configuration**: Various model configurations are set up, including the use of QLoRA (Question-Answering with Logic Relation Attention) architecture, BitsAndBytes quantization, and SFT (Soft Fine-Tuning) training strategy.

6. **Training**: The model is trained using the SFTTrainer provided by the `trl` library. This involves iterating through the training dataset, adjusting the model's parameters based on the loss computed from comparing its predictions to the ground truth labels.

## Quantization Methods

### BitsAndBytes Quantization

- **Use of 4-bit Precision**: The fine-tuning process utilizes 4-bit precision for loading the base model (`use_4bit=True`). This reduces the memory footprint and speeds up inference while maintaining performance.

- **Quantization Type**: The 4-bit quantization type is specified as "nf4" (`bnb_4bit_quant_type="nf4"`), which stands for nested floating-point 4-bit quantization. This method optimizes the quantization process to achieve a balance between model size and accuracy.

- **Nested Quantization**: Nested quantization (`use_nested_quant=False`) is disabled in this implementation. Nested quantization applies double quantization, further reducing the model's precision but potentially sacrificing some accuracy.

### LoRA Implementation

- **Logic Relation Attention**: LoRA is a novel attention mechanism introduced in the LLAMA model. It enhances the model's ability to capture logic relationships between tokens in the input sequence, particularly beneficial for question-answering tasks where understanding context is crucial.

- **LoRA Configuration**: The LoRA configuration (`LoraConfig`) includes parameters such as the attention dimension (`lora_r`), alpha parameter for scaling (`lora_alpha`), and dropout probability (`lora_dropout`). These parameters are carefully chosen to optimize the performance of the LoRA mechanism for the given task.

## Usage

1. **Clone Repository**: Clone this repository to your local machine.

2. **Run Fine-Tuning Script**: Execute the fine-tuning script (`fine_tune_llama.py`) to start the fine-tuning process. This script will load the PubMedQA dataset, configure the model, and train it using the specified parameters.

3. **Generate Answers**: After fine-tuning, the model can generate answers to medical questions. Use the provided `generate_answer` function to input symptoms or medical queries and receive responses from the fine-tuned LLAMA model.


---


## OCR-Based Drug Identification

This project aims to identify drugs from images containing text descriptions using Optical Character Recognition (OCR) and Generative AI. By leveraging OCR technology, the project extracts text from images, processes the text to identify drug names, and then utilizes Google Gemini-pro model to generate responses related to the identified drugs.



**Usage**:

   - **OCR (Optical Character Recognition)**:
     
     The project utilizes PaddleOCR, a deep learning-based OCR tool, to extract text from images. This text extraction step is crucial for identifying drug names present in the images.
     
   - **Drug Identification**:
     
     After extracting text from images, the project processes the text to identify drug names. It concatenates the extracted strings and uses Google Gemini to generate responses related to the identified drugs.
     
   - **Generative AI**:
     
     Gemini-pro is employed to generate responses based on the identified drugs. The API key for Gemini needs to be configured for access to the Generative AI model.

## Instructions

1. **Installation**:

   Ensure that you have Python installed on your system. Then, execute the provided bash command to install the necessary dependencies.

2. **Obtain API Key for Generative AI**:

   Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) to create an API key for Gemini-pro model. This key is required to configure Generative AI for access to the model.

3. **Usage**:

   - **Prepare Images**:
     
     Provide images containing text descriptions of drugs. Ensure that the images are clear and contain legible text for accurate OCR results.

   - **Run the Code**:
     
     Execute the provided Python script to perform OCR on the images, identify drug names, and generate responses using Generative AI.

     
   - **Run the Streamlit Deployment**:

   ```bash
   streamlit run streamlit_prescription_parsing.py
   ```
   
   -**Review Output**:
     
     Review the generated responses to get insights into the drugs identified from the images.

## Notes

- **Image Quality**:
  
  The accuracy of OCR and drug identification heavily relies on the quality of the input images. Ensure that the images provided are clear and contain well-defined text.

- **Gemini-pro Configuration**:
  
  Before using Gemini, make sure to configure the API key obtained from the Google AI Studio for access to the Gemini model.


  ## Disclaimer

This application is for educational and demonstration purposes only. It should not be used as a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare provider for accurate medical information and treatment.


---

# Personalized AI Chat Doctor


The implemented code incorporates RAG (Retrieval-Augmented Generation) architecture to power the personalized AI chat doctor:


1. **Retrieval Phase**: The system initially retrieves relevant medical documents or passages from a dataset (e.g., `iCliniq.json` and `GenMedGPT-5k.json`) based on the user's health issues input.
   

3. **Generation Phase**: After retrieving relevant documents, the system employs a generative model (in this case, the Google GenerativeAI model) to generate detailed responses. These responses are generated using the retrieved documents as contextual information, ensuring that the generated advice aligns with the provided medical context.
   

5. **Personalization**: RAG facilitates personalization by dynamically selecting relevant documents based on the user's health issues input. This ensures that the generated responses are tailored to the specific health concerns expressed by the user.


By leveraging RAG, the system delivers accurate, contextually relevant, and personalized medical advice to users, enhancing the overall effectiveness and user experience of the AI chat doctor application.


## Usage

1. Make sure all dependencies are installed.
2. Run the Streamlit app:

```bash
streamlit run streamlit_Gemini_RAG.py
```

3. The app will launch in your default web browser. Enter your health issues into the chat input.
4. The AI chat doctor will respond with personalized suggestions and advice based on the provided information.

## Files

- `streamlit_Gemini_RAG.py`: Contains the main code for the Streamlit app.
- `iCliniq.json`: JSON file containing medical data.
- `GenMedGPT-5k.json`: Additional JSON file containing medical data.

## Configuration

- Update the `GOOGLE_API_KEY` variable in `streamlit_Gemini_RAG.py` with your own Google API key.
- Ensure that the file paths for `iCliniq.json` and `GenMedGPT-5k.json` are correctly set in the code.




# Evaluation of Fine-tuned LLAMA and Microsoft-BioGPT


The assessment of the fine-tuned LLAMA and Microsoft-BioGPT models entails an examination across various evaluation metrics including BLEU, ROUGE, BERT Score, Novelty, Diversity, and Levenshtein distance. The fine-tuned LLAMA model exhibits comparable performance to BioGPT in BERT Score and Levenshtein distance measures, while demonstrating superior performance in terms of Diversity metric when compared to BioGPT.

---
