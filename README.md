# Crime Pattern Analysis

This project analyzes a crime dataset using Python and basic machine learning techniques. It loads crime records from a CSV file, cleans and transforms the data, visualizes trends, identifies location clusters, trains classification models, and extracts association rules for pattern discovery.

## Features

- Loads crime data from `data/crime_dataset.csv`
- Cleans and preprocesses date, time, location, and crime fields
- Visualizes:
  - Crimes per year
  - Crimes per month
  - Top crime categories
- Detects crime hotspots using K-Means clustering
- Trains and evaluates:
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- Finds relationships between location and crime type using Apriori association rules
- Prints a sample crime prediction and basic dataset insights

## Project Structure

```text
crime_pattern/
|-- app/
|   `-- analysis.py
|-- data/
|   `-- crime_dataset.csv
|-- README.md
`-- requirements.txt
```

## Requirements

- Python 3.9 or newer

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

From the project root:

```bash
python app/analysis.py
```

## Output

Running the script will:

- Print dataset shape before and after cleaning
- Display multiple charts using Matplotlib and Seaborn
- Train machine learning models and print their accuracy
- Generate association rules from the dataset
- Predict a sample crime category
- Print summary insights such as:
  - Most common crime
  - Most dangerous location
  - Peak crime hour

## Main Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `mlxtend`

## Notes

- The script expects the dataset at `data/crime_dataset.csv`.
- Graphs are shown in separate plot windows, so run the script in an environment that supports GUI plotting.
- Some imported modules in the script are currently unused, but the dependency list matches the code as written.

## Future Improvements

- Add a Jupyter notebook for interactive analysis
- Save charts to files instead of only displaying them
- Add command-line options for prediction input
- Split preprocessing, training, and visualization into separate modules

