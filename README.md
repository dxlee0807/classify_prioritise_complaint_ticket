
# Complaint Ticket Classification and Prioritization

## Introduction
This project focuses on automating the classification and prioritization of complaint tickets using machine learning models. The goal is to streamline the process of managing customer complaints by categorizing them into appropriate topics and assigning priority levels.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Examples](#examples)
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
- **Category Generation**: Automated generation and labeling of ticket categories.
- **Model Training**: Multiple models (e.g., Logistic Regression, SVM, Random Forest) are used to classify and prioritize tickets.
- **Deployment**: Support for deploying the models in a real-world ticketing system.

## Project Structure
- **ml_models/**: Directory containing saved machine learning models.
- **Notebooks**: Jupyter notebooks for data preprocessing, model training, and evaluation.
- **Scripts**: Python scripts for deploying models and handling ticket data.

## Dependencies
- Python 3.x
- Jupyter Notebook
- Required libraries listed in `requirements.txt`

## Configuration
Adjust model parameters and preprocessing steps in the respective notebooks and scripts according to your dataset and requirements.

## Examples
Refer to the Jupyter notebooks for detailed examples on how to preprocess data, train models, and evaluate performance.

## Contributors
- [dxlee0807](https://github.com/dxlee0807)

## License
This project is licensed under the MIT License.
