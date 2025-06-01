# Cat and Dog Image Analysis

This project explores image classification techniques using embeddings from a pretrained ResNet50 model. The goal is to compare multiple machine learning models for classifying images of cats and dogs and to visualize how they behave in a PCA-reduced space.

> This repository contains the exploratory data analysis and model evaluation for the [CatDog AI Classifier Web App](https://github.com/mooogy/catdog-ai), which uses the best-performing models identified here.

## Table of Contents
- [Analysis Recreation](#analysis-recreation)
- [The Data](#the-data)
- [Data Processing](#data-processing)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [License](#license)

## Analysis Recreation
> **Note**: Due to size constraints, the dataset is not included in this repository.  
To use this project:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset).
2. Extract the `cat/` and `dog/` folders from the Kaggle dataset into a `dataset/` directory:
```
catdog-image-analysis/
├── ImageEmbedding.ipynb
├── ModelEvaluation.ipynb
└── dataset/
    ├── cat/
    └── dog/
```
3. Run the .ipynb files in the following order:
- ImageEmbedding.ipynb
- ModelEvaluation.ipynb

## The Data

The image dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset), containing labeled images of cats and dogs. The dataset consists of:
- 1,000+ cat images
- 1,000+ dog images
- JPEG format, standard resolution

## Data Processing

- Each image was transformed using the preprocessing pipeline provided by `torchvision.models.ResNet50_Weights.DEFAULT`.
- A pretrained ResNet50 model was used as a feature extractor by removing its final classification layer.
- 2048-dimensional embeddings were generated for each image and saved into embeddings.csv, along with their respective labels and filenames.

## Model Evaluation

Multiple classifiers were trained and evaluated using both the original and PCA-reduced embeddings:

- PCA was used to reduce the 2048D embeddings to 1000D and 2D to compare accuracy and visualization effectiveness.
- K-Nearest Neighbors (KNN) was tuned over different values of k using 5-fold cross-validation. Best performance was found with k=11 on 1000D PCA.
- Support Vector Machines (SVM) were optimized using a grid search over C, gamma, and kernel. RBF kernel with C=100, gamma=0.1 performed best on the 2D projection.
- Logistic Regression was also tested, with C=0.1 showing strong generalization.

Decision boundaries were visualized in the 2D PCA-reduced space to understand how each model separates the two classes.

Validation techniques:
- Cross-validation
- Accuracy metrics

## Conclusion

- KNN with PCA(1000D) provided the best accuracy with relatively fast training time.
- KNN with PCA(2D) offered strong decision boundary visualization 
- Logistic Regression models were useful for probability-based predictions.

> Final models used in the production classifier [catdog-ai](https://github.com/mooogy/catdog-ai) were selected based on these evaluation results.

## License

This project is licensed under the [MIT License](LICENSE).
