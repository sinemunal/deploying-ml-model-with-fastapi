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

## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
   * Enable automatic deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Set up DVC on Heroku using the instructions contained in the starter directory.
* Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
* Write a script that uses the requests module to do one POST on your live API.
