# Expense Tracker LLM
## Description

This application is used to track your day to day expenses using Natural Language. It uses 2 models for predicting expense details from natural language:
- Fine tuned Palm2 Bison (Hosted using Google API)
- Fine tuned Llama2-7B (Hosted locally).

The data for fine tuning was generated using GPT-3.5 with the following prompt:
```
Below is a input that describes an expense. Write a response in json format that appropriately completes the request.
Response is a json string with fields - account_type (CREDIT or DEBIT), category, sub_category,  reason (Explain detailed reason if available), third_party - person who gave to got the money (Amount in Indian Rupees).
Generate appropriate response json string for the input expense. Response must be in only json string format strictly.

### Input:
I gave 5000 rupees to my friend for a personal loan repayment.

### Response:
```

With this technique, the 1000 data points was generated and both the models are fine tuned in following manner:
- Palm2 Bison was fine tuned on Google AI Studio Platform by importing the generated dataset with following configurations:
  - Max Output Tokens: 256
  - Temperature: 0.4
  - Learning Rate: 0.02
  - Batch Size: 16
  - Epochs: 10
  - Combined Loss: 0.01
- Llama2 was fine tuned using Ludwig AI and Transformers Framework on Tesla T4 Machine (Google Colab) with following configurations:
  - Max Output Tokens: 256
  - Temperature: 0.1
  - Learning Rate: 0.0004
  - Batch Size: 2
  - Epochs: 10
  - Combined Loss: 0.06
