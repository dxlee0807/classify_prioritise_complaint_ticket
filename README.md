
# Classifying and Prioritising Tickets for Optimised Customer Support using Natural Language Processing and Machine Learning

## Introduction
The project aims to classify and prioritise support tickets to speed up the customer support ticket resolution process, thus guaranteeing customer satisfaction and business revenue.

To achieve the project aim, these objectives should be accomplished:
1. To generate categories based on the ticket content by applying topic modelling.
2. To prioritise urgent tickets by conducting a sentiment analysis on the ticket content.
3. To build support ticket classification models based on categories generated from the first objective using supervised machine learning algorithms.
4. To assess the support ticket classification models using suitable evaluation metrics.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Examples](#examples)
- [Dataset](#dataset)
- [Contributors](#contributors)
- [License](#license)

## Installation
To set up the project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/dxlee0807/classify_prioritise_complaint_ticket.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Preprocess the ticket descriptions using the provided notebooks (`1_preprocess_ticket_description_and_eda.ipynb`).
2. Generate and label ticket categories using the `2_1_*` series of notebooks.
3. Train and evaluate models for classification and prioritization with the `3_*` series of notebooks.
4. Use the script `4_support_ticketing_system.py` to deploy the model in a ticketing system.

## Features
- **Data Preprocessing**: Includes data cleaning, exploration, and analysis.
- **Feature Engineering**: Automated generation and labelling of ticket categories and priorities on unlabelled support ticket datasets.
- **Model Training**: Multiple models (e.g., Logistic Regression, Support Vector Machine, Random Forest, Multinomial Naive Bayes, K-Nearest Neighbors, and Decision Tree) are used to classify and prioritize tickets.
- **Hyperparameter Tuning**: Model selection techniques such as grid search and random search are used to search the best combination of hyperparameter values for each model.
- **Deployment**: Supports deploying the models in a real-world ticketing system.

## Project Structure
- **ml_models/**: Directory containing saved machine learning models.
- **Notebooks**: Jupyter notebooks for data preprocessing, model training, and evaluation.
- **Scripts**: Python scripts for deploying models and handling ticket data.

## Dependencies
- Python 3.8.0
- Jupyter Notebook
- Required libraries listed in `requirements.txt`

## Configuration
Adjust model parameters and preprocessing steps in the respective notebooks and scripts according to your dataset and requirements.

## Examples
Refer to the Jupyter notebooks for detailed examples on how to preprocess data, train models, and evaluate performance.

## Dataset
The financial complaint dataset is available at https://www.kaggle.com/datasets/venkatasubramanian/automatic-ticket-classification/data

## Contributors
- [dxlee0807](https://github.com/dxlee0807)

## License
This project is licensed under the MIT License.
