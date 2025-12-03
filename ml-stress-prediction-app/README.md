# Machine Learning Stress Prediction Application

This project is a machine learning application designed to predict stress levels and estimate fringe orders from images. It includes an image preprocessing pipeline, two machine learning models, and a user-friendly interface built with Streamlit.

## Project Structure

```
ml-stress-prediction-app
├── src
│   ├── preprocessing          # Image preprocessing functions
│   ├── models                 # Machine learning models
│   ├── training               # Training scripts for models
│   ├── prediction             # Prediction functions
│   └── utils                  # Utility functions
├── streamlit_app              # Streamlit web application
├── config                     # Configuration files
├── requirements.txt           # Project dependencies
├── setup.py                   # Packaging information
└── README.md                  # Project documentation
```

## Features

- **Image Preprocessing**: Functions for resizing, normalization, and augmentation of images.
- **Stress Prediction Model**: A machine learning model that predicts stress levels based on input images.
- **Fringe Order Estimation Model**: A model that estimates fringe orders from images.
- **Training Modules**: Scripts to train both models with appropriate data preparation.
- **Prediction Module**: A function to make predictions using the trained models.
- **Streamlit Interface**: A multi-page web application for training models, making predictions, and analyzing results.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-stress-prediction-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application by editing `config/config.yaml` to set model parameters and paths.

## Usage

To run the Streamlit application, execute the following command:
```
streamlit run streamlit_app/main.py
```

Navigate through the interface to train models, make predictions, and analyze results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.