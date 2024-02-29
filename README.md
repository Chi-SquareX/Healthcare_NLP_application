## OCR-Based Drug Identification

This project aims to identify drugs from images containing text descriptions using Optical Character Recognition (OCR) and Generative AI. By leveraging OCR technology, the project extracts text from images, processes the text to identify drug names, and then utilizes Google Gemini-pro model to generate responses related to the identified drugs.
```markdown

## Overview

1. **Installation of Dependencies**:
   
   To set up the project environment, you need to install the required dependencies. Use the following commands in your terminal:

   ```bash
   pip install -r requirements.txt
   ```

2. **Usage**:

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

   - **Review Output**:
     
     Review the generated responses to get insights into the drugs identified from the images.

## Notes

- **Image Quality**:
  
  The accuracy of OCR and drug identification heavily relies on the quality of the input images. Ensure that the images provided are clear and contain well-defined text.

- **Gemini-pro Configuration**:
  
  Before using Gemini, make sure to configure the API key obtained from the Google AI Studio for access to the Gemini model.
```
