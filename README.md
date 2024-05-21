### Data-Analyst-Job-Market-Analysis

# Project Description
This project aims to simplify the job hunt for Data Analyst roles in India by scraping job listings from LinkedIn, preprocessing and cleaning the data, and performing detailed analysis. The project includes skills analysis, educational requirements, work mode preferences, job location distribution, and company and industry insights.

# Goals
Simplify Job Search: Streamline the job search process for Data Analyst positions in India.
Comprehensive Analysis: Provide insights into the skills, experience, and educational requirements for Data Analyst roles.
Data-Driven Insights: Offer data-driven insights to help job seekers understand market trends and requirements.

## Project Workflow

Data Collection
Web Scraping: Utilized Selenium to scrape job listings for Data Analyst roles from LinkedIn.
Saved Data: The scraped data was saved into a CSV file named AnalystJobs.csv.
Data Preprocessing and Cleaning
Initial Cleaning: Performed initial cleaning and saved the processed data to modified_data.csv.
Advanced Cleaning: Further processed the data to extract degrees from job descriptions and saved it to modified_data1.csv.

## Analysis

#Skills Analysis:

Token Frequency: Identified and visualized the most frequently mentioned skills in job listings.
Skill Clustering: Used TF-IDF vectorization and K-means clustering to identify groups of related skills.
Skills by Experience Level: Visualized top skills required for different experience levels.
Educational Requirements:

Degree Distribution: Analyzed and visualized the distribution of job openings based on required educational qualifications.
Top Skills by Degree: Identified top skills required for jobs requiring different educational qualifications.
Work Mode Analysis:

Work Mode Distribution: Analyzed and visualized the distribution of jobs by work mode.
Top Skills by Work Mode: Identified top skills required for different work modes.
Job Location Analysis:

Job Locations: Analyzed the geographic distribution of job openings using city coordinates.
Top Cities: Identified and visualized the top cities with the most job openings.
Actively Recruiting Locations: Identified and visualized locations with the most actively recruiting companies.
Company Analysis:

Companies with Most Job Openings: Identified top companies with the most job openings.
Actively Recruiting Companies: Visualized companies actively recruiting based on job openings.
Industry Analysis:

Industry Distribution: Analyzed and visualized the distribution of job openings across different industries.
Actively Recruiting Industries: Identified top industries actively recruiting for Data Analyst positions.
Employee Size Distribution: Analyzed the distribution of job openings based on the employee size of companies.
Work Mode by Employee Size: Visualized the distribution of work modes across different employee sizes.
Industry-Specific Skills: Identified top skills required for each industry.

# Hypothesis Testing
Recruitment Status:
Tested if recruitment status depends on Industry and Location.
Required Experience/Degree among Industries:
Tested if required experience or degree for jobs differs between industries.
Key Findings
Skills Analysis: Identified top skills and skill clusters for Data Analyst roles.
Educational Requirements: Analyzed degree distribution and top skills by degree.
Work Mode Preferences: Analyzed work mode distribution and top skills by work mode.
Job Location Insights: Identified top cities and locations with the most actively recruiting companies.
Company and Industry Insights: Identified top companies, industries, and skills required in different industries.

# CSV files
AnalystJobs.csv: Raw scraped data from LinkedIn.
modified_data.csv: Preprocessed and cleaned data.
modified_data1.csv: Further cleaned data with degree information.

# Dependencies
Selenium
Pandas
Numpy
Scikit-learn
NLTK
Matplotlib
Seaborn

# Conclusion
This project provides a comprehensive analysis of the Data Analyst job market in India, helping job seekers to understand market trends, required skills, and educational qualifications. The insights gained from this analysis can significantly simplify the job search process and guide job seekers in aligning their skills with market demands.
