This is the third project for Udacity DevOps nano degree program. The starter code is taken from
https://github.com/udacity/nd0821-c3-starter-code/tree/master

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment by executing the following in the command line
  `conda create --name udacity-fastapi --file requirements.txt -c conda-forge`

      
## Data

* Use census.csv from the data folder in the repository.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.

## Model
* A model card provided for the implemented model.

## Model performance
To see the model performance for each specified categorical value run:
` python model_performance.py > screenshots/slice_output.txt`
This will create the performance metrics of trained model on test set for each gender
and saves the results to slice_output.txt.

## API Creation and Deployment
* RESTful API using FastAPI is implemented with GET and POST methods where POST method performs inference on provided data.
* App created on Render and deployed when checks defined via Github actions passes.
