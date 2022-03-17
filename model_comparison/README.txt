## Model Comparison

The model comparison part of the road detection project for Madagascar allows to train and compare the performance of different road segmentation models. The test set are currently images from the district 8 of the Antanarivo Province scraped from mapbox. The framework can easily be extended to further configurations.

## Usage 

The file model_comparison_demo.ipynb gives an example how to use the model comparison api and evaluate the results. The demo shows how to call the "perform_cv" function of the module module_comparison.py. Parameters are the IMAGE_FOLDER with the test_images, the MODEL_NAMES of the models to be evaluated, a model_getter function that determines how to make predictions from a model and a data loader function to load the data. The file test_model_comparison.py contains unit tests for the model comparison part.

## Setup:

The required libraries for the model comparison are specified in the file "requirements.txt". 

## Existing Models 

The following models already exist::

- A unet model trained on Massachusetts . This model was trained using the module "model_training_on_massachussets_data.ipynb". By running this notebook the massachussets data are also downloaded if not already existent. 

- A unet model trained on the Madagascar Data. This model was trained using the module "model_training_on_madagascar_data.ipynb". The images are the ones from district 8 of the antannarivo privinces scraped from mapbox. 

- A unet model that was trained on both the Madagascar and the Massachussets Data. This model was trained using the module "model_training_on_madagascar_and_massachussets_data.ipynb". 

Due to different libraries used, the libraries needed for the training of the models are different and incompatible with the libraries used for the model comparison. The required libraries for the model comparison are specified in the file "requirements_tf1.15.txt". 

To train further models the indicated training scripts modules can be easily adapted. 

## Remarks

- In our tests, the model trained on only Madagascar data achieved the best results

- The modules give a framework to add further modules that use different architectures, training data and hyperparameters. In particular, threshold tuning is still an open issue. 

- Intermediate results, in particular trained models per fold, will be cached to disk to the folder "cached_intermediate_data". 

- GPU support highly accelerates the runtime of the different modules.
