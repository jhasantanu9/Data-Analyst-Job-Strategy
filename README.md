# Data-Analyst-Job-Market-Analysis

## Project Overview
**Data-Analyst-Job-Market-Analysis** is a data-driven project aimed at simplifying the job hunt for Data Analyst roles in India. By leveraging advanced web scraping techniques, data preprocessing, and analysis, this project provides valuable insights into the job market. The project showcases key skills and tools in data collection, processing, and visualization, including a web application built using Shiny Python.

Explore the live web app here: [Data Analyst Jobs in India](https://6hohxv-santanu-jha.shinyapps.io/data_analyst_jobs_in_india/)

## Skills and Technologies
- **Python**: Core programming language used throughout the project.
- **Web Scraping**: Utilized Selenium to scrape job listings from LinkedIn.
- **Data Preprocessing**: Performed data cleaning and preprocessing using Pandas and Numpy.
- **Natural Language Processing (NLP)**: Employed NLTK for text processing, including tokenization and stopwords removal.
- **Machine Learning**: Applied TF-IDF vectorization and K-means clustering with Scikit-learn for skills analysis.
- **Data Visualization**: Created visualizations using Matplotlib to present insights.
- **Shiny for Python**: Developed an interactive web application using Shiny Python, enabling users to explore the analyzed data.
- **Deployment**: Deployed the Shiny Python app on Shinyapps.io.

## Project Goals
- **Simplify Job Search**: Streamline the search process for Data Analyst positions in India.
- **Comprehensive Analysis**: Provide insights into the skills, experience, and educational requirements for Data Analyst roles.
- **Data-Driven Insights**: Offer data-driven insights to help job seekers understand market trends and requirements.

## Data Dictionary

| Column Name               | Description                                                                                     |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `Link`                    | URL to the job listing.                                                                         |
| `Job Title`               | Title of the job position.                                                                      |
| `Required Skills`         | List of skills required for the job, extracted from the job description.                        |
| `Job Description`         | Full text of the job description.                                                               |
| `Required Experience`     | Level of experience required for the job (e.g., Entry, Mid, Senior).                            |
| `Company Name`            | Name of the company offering the job.                                                           |
| `Location`                | Location of the job, typically formatted as "City, State".                                      |
| `Employee Size`           | Size of the company in terms of number of employees.                                            |
| `Industry`                | Industry in which the company operates.                                                         |
| `Work Mode`               | Mode of work (e.g., On-site, Remote, Hybrid).                                                   |
| `Actively Recruiting`     | Indicator of whether the company is actively recruiting (Yes/No).                               |
| `City`                    | City where the job is located, extracted from Location.                                         |
| `State`                   | State where the job is located, extracted from Location.                                        |
| `Cleaned_Job_Description` | Preprocessed job description used for text analysis (lowercased, tokenized, stopwords removed). |
| `bachelor`                | Binary indicator (0/1) of whether a Bachelor's degree is mentioned as a requirement.            |
| `master`                  | Binary indicator (0/1) of whether a Master's degree is mentioned as a requirement.              |
| `phd`                     | Binary indicator (0/1) of whether a PhD degree is mentioned as a requirement.                   |

## Project Workflow

### Data Collection
1. **Web Scraping**: Utilized Selenium to scrape job listings for Data Analyst roles from LinkedIn.
2. **Saved Data**: The scraped data was saved into a CSV file named `AnalystJobs.csv`.

### Data Preprocessing and Cleaning
1. **Initial Cleaning**: Performed initial cleaning and saved the processed data to `modified_data.csv`.
2. **Advanced Cleaning**: Further processed the data to extract degrees from job descriptions and saved it to `modified_data1.csv`.

### Folder Structure
- `scrapper.ipynb`: Notebook for web scraping.
- `Data preprocessing.ipynb`: Notebook for data cleaning and preprocessing.
- `Shiny_app/app.py`: Python script for running the Shiny web app.
- `AnalystJobs.csv`: Raw scraped data from LinkedIn.
- `modified_data.csv`: Preprocessed and cleaned data.
- `modified_data1.csv`: Further cleaned data with degree information.

## Analysis

### Skills Analysis
- **Token Frequency**: Identified and visualized the most frequently mentioned skills in job listings.
- **Skill Clustering**: Used TF-IDF vectorization and K-means clustering to identify groups of related skills.
- **Skills by Experience Level**: Visualized top skills required for different experience levels.

### Educational Requirements
- **Degree Distribution**: Analyzed and visualized the distribution of job openings based on required educational qualifications.
- **Top Skills by Degree**: Identified top skills required for jobs requiring different educational qualifications.

### Work Mode Analysis
- **Work Mode Distribution**: Analyzed and visualized the distribution of jobs by work mode.
- **Top Skills by Work Mode**: Identified top skills required for different work modes.

### Job Location Analysis
- **Job Locations**: Analyzed the geographic distribution of job openings using city coordinates.
- **Top Cities**: Identified and visualized the top cities with the most job openings.
- **Actively Recruiting Locations**: Identified and visualized locations with the most actively recruiting companies.

### Company Analysis
- **Companies with Most Job Openings**: Identified top companies with the most job openings.
- **Actively Recruiting Companies**: Visualized companies actively recruiting based on job openings.

### Industry Analysis
- **Industry Distribution**: Analyzed and visualized the distribution of job openings across different industries.
- **Actively Recruiting Industries**: Identified top industries actively recruiting for Data Analyst positions.
- **Employee Size Distribution**: Analyzed the distribution of job openings based on the employee size of companies.
- **Work Mode by Employee Size**: Visualized the distribution of work modes across different employee sizes.
- **Industry-Specific Skills**: Identified top skills required for each industry.

## Hypothesis Testing
1. **Recruitment Status**: Tested if recruitment status depends on Industry and Location.
2. **Required Experience/Degree among Industries**: Tested if required experience or degree for jobs differs between industries.

## Key Findings
- **Skills Analysis**: Identified top skills and skill clusters for Data Analyst roles.
- **Educational Requirements**: Analyzed degree distribution and top skills by degree.
- **Work Mode Preferences**: Analyzed work mode distribution and top skills by work mode.
- **Job Location Insights**: Identified top cities and locations with the most actively recruiting companies.
- **Company and Industry Insights**: Identified top companies, industries, and skills required in different industries.

## Web Application
Explore the **Data-Analyst-Job-Market-Analysis** project through an interactive Shiny Python app, deployed on Shinyapps.io: [Data Analyst Jobs in India](https://6hohxv-santanu-jha.shinyapps.io/data_analyst_jobs_in_india/)

## Dependencies
- Python 3.x
- Selenium
- Pandas
- Numpy
- Scikit-learn
- NLTK
- Matplotlib
- Shiny Python
- sklearn

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Data-Analyst-Job-Market-Analysis.git
