import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from wordcloud import WordCloud 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from shiny import App, render, ui, reactive
import plotly.express as px
from shinywidgets import output_widget, render_widget
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Loading the data
data_path = 'data/modified_data1.csv'
df = pd.read_csv(data_path)

data_path_coordinates = 'data/city_coordinates.csv'
city_coordinates_df = pd.read_csv(data_path_coordinates)

df.dropna(subset=['Required Skills', 'Required Experience','Work Mode','Industry'], inplace=True)

df['Required Skills'] = df['Required Skills'].astype(str)
df['Required Skills'] = df['Required Skills'].str.lower()

def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str=True)
    return string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))

    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    string_without_stopwords = ' '.join(filtered_words)

    return string_without_stopwords

df['Required Skills'] = df['Required Skills'].apply(tokenize)
df['Required Skills'] = df['Required Skills'].apply(remove_stopwords)

# Prepare the data for plotting
all_tokens = [token.strip() for tokens_list in df['Required Skills'] for token in tokens_list.split(',')]
token_frequency = pd.Series(all_tokens).value_counts()

# Extract tokens with frequency greater than or equal to a threshold
threshold = 5
selected_tokens = token_frequency[token_frequency >= threshold].index.tolist()
selected_tokens_sorted = token_frequency[selected_tokens].sort_values(ascending=False)
top_N = 10  # Number of top tokens to plot
top_tokens = selected_tokens_sorted.head(top_N).sort_values()

# Prepare the data for the pie chart
experience_counts = df['Required Experience'].value_counts().sort_values()
work_mode_counts = df['Work Mode'].value_counts()
bachelor_count = df['bachelor'].sum()
master_count = df['master'].sum()
phd_count = df['phd'].sum()


# Define the plotting function
def create_plot():
    fig, ax = plt.subplots(figsize=(8, 6))
    top_tokens.plot(kind='barh', ax=ax, color='#0077B5')
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    return fig

# Define the function to create the WordCloud plot
def create_wordcloud():
    wordcloud = WordCloud(width=800, height=600, background_color='white').generate_from_frequencies(token_frequency)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Vectorize tokens using TF-IDF representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(selected_tokens)

# Perform K-means clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Define the function to create the clustering plots
def create_clustering_plot():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Remove spines and ticks for each axis
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])  # Remove ticks
        ax.set_yticks([]) 

    # Plot 1: Clustering of Tokens
    for i in range(k):
        axes[0].scatter(X_pca[kmeans.labels_ == i, 0], X_pca[kmeans.labels_ == i, 1], label=f'Cluster {i+1}')
    axes[0].legend(fontsize=10)  # Increase legend font size
    axes[0].grid(True)


    # Prepare cluster names for Plot 2
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    feature_names = vectorizer.get_feature_names_out()
    cluster_names = []
    for i in range(k):
        top_words = [feature_names[ind] for ind in order_centroids[i, :3]]  # Extract top 5 words from each cluster
        cluster_names.append(' '.join(top_words))

    # Plot 2: Clustering of Skills
    for i in range(k):
        axes[1].scatter(X_pca[kmeans.labels_ == i, 0], X_pca[kmeans.labels_ == i, 1], label=cluster_names[i])
    axes[1].legend(fontsize=10)  # Increase legend font size
    axes[1].grid(True)

    fig.text(0.5, 0.5, 
             'Plot on Left: Visualizes the clustering of tokens using PCA-reduced features.\n'
             'Each color represents a different cluster of tokens.\n\n'
             'Plot on Right: Shows the same clustering but includes labels for the top words in each cluster.\n'
             'These labels help identify the main characteristics of each cluster.',
             ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
    line = Line2D([0.5, 0.5], [0, 1], color='black', linewidth=1.5, linestyle='--')
    fig.add_artist(line)


    # Adjust layout to prevent overlap
    return fig
   
def create_bar_chart_experience(experience_counts):
    colors = ['orange', 'yellow', 'salmon', 'lightgreen','skyblue']
    total = experience_counts.sum()
    percentages = (experience_counts / total) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        experience_counts.index,
        experience_counts.values,
        color=colors
    )
    
    # Annotate bars with percentage values
    for bar, percentage in zip(bars, percentages):
        width = bar.get_width()
        ax.annotate(f'{percentage:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

    # Hide all spines (lines around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='y', which='both', length=0)  # Remove ticks

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    return fig

counts = [bachelor_count, master_count, phd_count]
labels = ['Bachelor', 'Master', 'PhD']
colors = ['skyblue', 'lightgreen', 'salmon']

# Define the function to create the pie chart
def create_degree_pie_chart_degree():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig

def create_work_mode_pie_chart(work_mode_counts):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(work_mode_counts, labels = work_mode_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'lightcoral'])
    ax.axis('equal')
    return fig

# Combine tokens into a single list
all_tokens = [token.strip() for skills_list in df['Required Skills'] for token in skills_list.split(', ')]

# Create a dictionary to store skills for each experience level
skills_by_experience = {}

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    experience_level = row['Required Experience']
    required_skills = row['Required Skills'].split(', ')
    
    # Check if experience level is already in the dictionary
    if experience_level in skills_by_experience:
        # Extend the existing list of skills
        skills_by_experience[experience_level].extend(required_skills)
    else:
        # Create a new list of skills
        skills_by_experience[experience_level] = required_skills

# Count the occurrences of each skill for each experience level
skill_counts_by_experience = {}
for experience_level, skills in skills_by_experience.items():
    skill_counts_by_experience[experience_level] = pd.Series(skills).value_counts().head(5).sort_values()

# Define the function to plot the top skills
def plot_skills(experience_level):
    plt.figure(figsize=(10, 5))
    skill_counts = skill_counts_by_experience.get(experience_level, pd.Series([]))
    plt.barh(skill_counts.index, skill_counts.values, color='#0077B5')
    plt.gca().lines = []
    plt.show()

# Convert the DataFrame to long format
data_long = df.melt(id_vars=['Required Skills'], var_name='Degree Level', value_name='Count')

# Ensure that 'Count' is numeric
data_long['Count'] = pd.to_numeric(data_long['Count'], errors='coerce')

# Filter to include only rows with count > 0
data_long = data_long[data_long['Count'] > 0]

# Function to calculate top skills for each degree level
def get_top_skills(df, top_n=10):
    skills = df['Required Skills'].str.split(', ', expand=True).stack()
    return skills.value_counts().head(top_n).sort_values()

# Calculate top skills for each degree level
top_n = 5
bachelor_skill_counts = get_top_skills(data_long[data_long['Degree Level'] == 'bachelor'], top_n)
master_skill_counts = get_top_skills(data_long[data_long['Degree Level'] == 'master'], top_n)
phd_skill_counts = get_top_skills(data_long[data_long['Degree Level'] == 'phd'], top_n)

# Function to plot the top skills for the selected degree level
def plot_top_skills(degree_level):
    plt.figure(figsize=(10, 6))
    plt.gca().lines = []

    
    if degree_level == 'bachelor':
        top_skills = bachelor_skill_counts
        color = '#0077B5'
    elif degree_level == 'master':
        top_skills = master_skill_counts
        color = '#0077B5'
    elif degree_level == 'phd':
        top_skills = phd_skill_counts
        color = '#0077B5'

    top_skills.plot(kind='barh', color=color)
    plt.show()

# Create a dictionary to store skills for each work mode
skills_by_work_mode = {}

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    work_mode = row['Work Mode']
    required_skills = row['Required Skills'].split(', ')
    
    # Check if work mode is already in the dictionary
    if work_mode in skills_by_work_mode:
        # Extend the existing list of skills
        skills_by_work_mode[work_mode].extend(required_skills)
    else:
        # Create a new list of skills
        skills_by_work_mode[work_mode] = required_skills

# Count the occurrences of each skill for each work mode
skill_counts_by_work_mode = {}
for work_mode, skills in skills_by_work_mode.items():
    skill_counts_by_work_mode[work_mode] = pd.Series(skills).value_counts().head(5).sort_values()


# Filter out actively recruiting companies
company_counts = df['Company Name'].value_counts()
top_10_companies = company_counts.head(10).sort_values()
actively_recruiting_companies = df[df['Actively recruiting'] == 'Yes']
job_openings_count = actively_recruiting_companies.groupby('Company Name').size()
top_5_companies = job_openings_count.nlargest(5).sort_values()
# Filter companies actively recruiting
actively_recruiting = df[df['Actively recruiting'] == 'Yes']
industry_counts = actively_recruiting['Industry'].value_counts()
# Count the occurrences of each industry
industry_counts = df['Industry'].value_counts().head(5)
industry_counts_sorted = industry_counts.sort_values()

df['Required Skills'] = df['Required Skills'].str.split(',')

all_skills = set(skill.strip() for skills in df['Required Skills'] for skill in skills)

# Create a dictionary to hold skill counts for each industry
industry_skill_counts = {industry: {skill: 0 for skill in all_skills} for industry in df['Industry'].unique()}

# Count occurrences of each skill within each industry
for index, row in df.iterrows():
    industry = row['Industry']
    skills = row['Required Skills']
    for skill in skills:
        industry_skill_counts[industry][skill.strip()] += 1

# Create a DataFrame to hold skill counts for each industry
skill_counts_df = pd.DataFrame(industry_skill_counts).T

# Identify the top skills for each industry
top_skills_per_industry = {}
for industry in skill_counts_df.index:
    top_skills = skill_counts_df.loc[industry].sort_values(ascending=False).head(5)  # Get top 5 skills
    top_skills_per_industry[industry] = top_skills

# Filter out actively recruiting companies
actively_recruiting = df[df['Actively recruiting'] == 'Yes']
location_counts = actively_recruiting['City'].value_counts().head(5).sort_values()


# Create a frequency column in city_coordinates_df
city_coordinates_df['job_count'] = city_coordinates_df.groupby('City')['City'].transform('count')

# Define the UI for the Shiny app
app_ui = ui.page_fluid(
    ui.panel_title("Data Analyst Job Trends", "Job Market Overview"),  
    ui.navset_card_tab(
        ui.nav_panel("Dashboard",
            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Most Sought-After Skills", style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot1"),
                        full_screen = True,
                    ),
                    ui.card(
                        ui.card_header("In-Demand Skills for Data Analysts: A Word Cloud", style="text-align: center; font-size: 20px; center; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot2"),
                        full_screen = True,

                    ),
                    ui.card(
                        ui.card_header("Clusters of In-Demand Skills", style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot3"),
                        full_screen = True,
                    ),
                    col_widths={"sm": (6, 6, 12)}
                ),
            ),
            ui.card(
                ui.card_header("Distribution of Job Roles and Required Skills by:", style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("Education",
                                       style="text-align: center; font-family: Arial, font-size: 15px; sans-serif;"),
                        ui.card(
                            ui.output_plot("plot5"),
                            full_screen = True,
                        ),
                        ui.card(
                            ui.input_select("degree_level", "Degree Level", ['bachelor', 'master', 'phd']),
                            ui.output_plot("plot51"),
                            full_screen = True,
                        )
                    ),
                    ui.card(
                        ui.card_header("Experience Level",
                                       style="text-align: center; font-size: 15px; font-family: Arial, sans-serif;"),
                        ui.card(
                            ui.output_plot("plot4"),
                            full_screen = True,
                        ),
                        ui.card(
                            ui.input_select("experience_level", "Experience Level", list(skill_counts_by_experience.keys())),
                            ui.output_plot("plot41"),
                            full_screen = True,
                        )
                    ),

                    ui.card(
                        ui.card_header("Work Mode",
                                       style="text-align: center; font-size: 15px; font-family: Arial, sans-serif;"),
                        ui.card(
                            ui.output_plot("plot6"),
                            full_screen = True,
                        ),
                        ui.card(
                            ui.input_select("work_mode", "Work Mode", list(skill_counts_by_work_mode.keys())),
                            ui.output_plot("plot61"),
                            full_screen = True,
                        )
        
                    )
                )
            ),
            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Actively Recruiting Companies",
                                       style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot7"),
                        full_screen = True,

                    ),
                    ui.card(
                        ui.card_header("Companies with High Job Vacancies",
                                       style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot8"),
                        full_screen = True,
                    ),
                )
            ),

            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Industries Actively Recruiting",
                                       style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot9"),
                        full_screen = True,
                    ),
                    ui.card(
                        ui.card_header("In-Demand Skills Across Different Industries",
                                       style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.input_select("industry", "Select Industry", choices=list(top_skills_per_industry.keys())),
                        ui.output_plot("plot10"),
                        full_screen = True,
                        )
                    )
                ),

            ui.card(
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Regions Hiring Activity",
                                       style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        ui.output_plot("plot11"),
                        full_screen = True,
                    ),
                    ui.card(
                        ui.card_header("Regions with the Highest Openings",
                                       style="text-align: center; font-size: 20px; font-weight: bold; font-family: Arial, sans-serif;"),
                        output_widget("map"),
                        full_screen = True,
                    ),
                    fillable=True,
                )
            )
        ),  # Closing parenthesis for ui.nav_panel("Dashboard")

        ui.nav_panel("Hypothesis Tests",
            ui.card(
                ui.HTML("<p>This page presents the results of hypothesis tests conducted. We use chi-square tests to determine the association between 'Actively recruiting' status and categorical variables, as well as to analyze the required experience and degree among different industries.</p>"),
                
                ui.HTML("<h3>Hypothesis 1: Recruitment Status and Industry</h3>"),
                ui.HTML("<p><strong>Hypothesis:</strong></p>"),
                ui.HTML("<p>H0: The recruitment status (actively recruiting or not) depends on Industry.</p>"),
                ui.HTML("<p>H1: The recruitment status (actively recruiting or not) does not depend on Industry.</p>"),
                ui.HTML("<p><strong>Chi-Square Test Results:</strong></p>"),
                ui.output_text("industry_results"),
                ui.HTML("<p><strong>Explanation:</strong> A chi-square test was conducted to examine the association between recruitment status and industry. The null hypothesis is that the recruitment status is independent of the industry. If the p-value is greater than 0.05, we fail to reject the null hypothesis, indicating no significant association between the two variables. In this case, we see that the p-value is greater than 0.05, suggesting that the recruitment status does not depend on the industry.</p>"),

                ui.HTML("<h3>Hypothesis 2: Recruitment Status and Location</h3>"),
                ui.HTML("<p><strong>Hypothesis:</strong></p>"),
                ui.HTML("<p>H0: The recruitment status (actively recruiting or not) depends on Location.</p>"),
                ui.HTML("<p>H1: The recruitment status (actively recruiting or not) does not depend on Location.</p>"),
                ui.HTML("<p><strong>Chi-Square Test Results:</strong></p>"),
                ui.output_text("location_results"),
                ui.HTML("<p><strong>Explanation:</strong> Similarly, a chi-square test was performed to test the relationship between recruitment status and location. With a p-value greater than 0.05, we fail to reject the null hypothesis, indicating that the recruitment status is not significantly associated with the location.</p>"),

                ui.HTML("<h3>Hypothesis 3: Required Experience Across Industries</h3>"),
                ui.HTML("<p><strong>Hypothesis:</strong></p>"),
                ui.HTML("<p>H0: The required experience differs between industries.</p>"),
                ui.HTML("<p>H1: The required experience in one industry is the same as in another industry.</p>"),
                ui.HTML("<p><strong>Chi-Square Test Results:</strong></p>"),
                ui.output_text("experience_results"),
                ui.HTML("<p><strong>Explanation:</strong> This test examines whether the required experience level varies across different industries. A p-value less than 0.05 would suggest rejecting the null hypothesis, indicating that experience requirements differ by industry. The results should show whether this is the case, and guide decisions regarding experience requirements in different sectors.</p>"),

                ui.HTML("<h3>Hypothesis 4: Required Degree Across Industries</h3>"),
                ui.HTML("<p><strong>Hypothesis:</strong></p>"),
                ui.HTML("<p>H0: The required degree differs between industries.</p>"),
                ui.HTML("<p>H1: The required degree in one industry is the same as in another industry.</p>"),
                ui.HTML("<p><strong>Chi-Square Test Results:</strong></p>"),
                ui.output_text("degree_results"),
                ui.HTML("<p><strong>Explanation:</strong> Lastly, this test checks if the degree requirements vary between industries. A significant p-value (less than 0.05) would indicate that different industries require different levels of degrees. The interpretation will guide understanding of educational requirements across sectors.</p>"),
                
                ui.HTML("<h3>General Interpretation</h3>"),
                ui.HTML("<p>For each hypothesis test, the results help us understand the relationships between recruitment status, experience, degree requirements, and various industries and locations. These insights are useful for making informed decisions about recruitment strategies, educational qualifications, and industry-specific requirements.</p>")
            )
        ),


        ui.nav_panel("About",
            ui.layout_column_wrap(
                ui.card(
                    ui.div(ui.h4("Overview"), class_="get_in_touch"),
                    ui.p(
                        "Welcome to my comprehensive Data Analyst Job Market Analysis app. As a data analyst with a keen focus on actionable insights and data-driven decision-making, I have developed this application to offer a detailed examination of the Data Analyst job market in India. This tool is designed not only to present valuable market trends but also to highlight key skills and expertise that can drive successful career outcomes."
                    ),
                    ui.p("Using data collected from LinkedIn job postings across the country, the app offers valuable insights into the demand for data analysts, the skills employers are seeking, "
                        "and the industries actively hiring. "
                        "Additionally, the app includes hypothesis tests to validate trends and draw data-driven conclusions. "
                        "I developed this app to empower others with the knowledge needed to navigate their career paths effectively. "
                        "By leveraging real-time data and rigorous analysis, this tool aims to support informed decision-making and provide clarity in an ever-evolving job market.")
                ),
            ),         

                    ui.HTML('''
                        <div class="contact-container">
                            <h4 style="text-align: center;">Get in Touch</h4>
                            <div class="social-links">
                                <a href="mailto:jhasantanu9@gmail.com" class="btn btn-primary">Email</a>
                                <a href="https://www.linkedin.com/in/santanu-jha-845510292/" target="_blank" class="btn btn-linkedin">LinkedIn</a>
                                <a href="https://github.com/jhasantanu9" target="_blank" class="btn btn-github">GitHub</a>
                                <a href="https://santanujha.netlify.app" target="_blank" class="btn btn-portfolio">Portfolio</a>
                            </div>
                        </div>
                    '''),
            
            # Custom CSS for styling
            ui.tags.style('''
                        .contact-container {
                            display: flex;
                            flex-direction: column;
                            align-items: left;
                            margin: 20px 0;
                        }
                        .social-links {
                            display: flex;
                            justify-content: center;
                            gap: 10px;
                            margin-top: 20px;
                        }
                        .social-links .btn {
                            padding: 8px 16px;
                            color: white;
                            border-radius: 5px;
                            text-decoration: none;
                            transition: background-color 0.3s, color 0.3s; /* Smooth transition for hover effect */
                        }
                        .btn-linkedin { background-color: #0077B5; }
                        .btn-github { background-color: #333333; }
                        .btn-portfolio { background-color: #61dbbb; }
                        .btn-primary { background-color: #c71610; } 
                        /* Hover effects */
                        .btn-linkedin:hover { background-color: #005582; }
                        .btn-github:hover { background-color: #1a1a1a; }
                        .btn-portfolio:hover { background-color: #5ba49d; }
                        .btn-primary:hover { background-color: #781c1c; } 
                        .footer {
                            text-align: center;
                            margin-top: 40px;
                            font-size: 0.9em;
                            color: #888;
                        }
                        .get_in_touch {
                            margin-top: 20px; /* Adds space before the "Get in Touch" heading */
                        }
                    '''),
                        
            # Footer with copyright
            ui.div(
                ui.p("© 2024 @jhasantanu9"),
                class_="footer"
            )
        )
    )
)

# Define the server logic
def server(input, output, session):

#DASHBOARD
    @render.plot()
    def plot1():
        fig = create_plot()
        return fig
    
    @render.plot()
    def plot2():
        fig = create_wordcloud()
        return fig
    
    @render.plot()
    def plot3():
        fig = create_clustering_plot()
        return fig
    
    @render.plot()
    def plot4():
        fig = create_bar_chart_experience(experience_counts)
        return fig
    
    @render.plot()
    def plot5():
        fig = create_degree_pie_chart_degree()
        return fig
    
    @render.plot()
    def plot6():
        fig = create_work_mode_pie_chart(work_mode_counts)
        return fig
    
    @reactive.Calc
    def selected_experience_level():
        return input.experience_level()

    @render.plot()
    def plot41():
        experience_level = selected_experience_level()
        if experience_level:
            fig, ax = plt.subplots(figsize=(10, 5))
            skill_counts = skill_counts_by_experience.get(experience_level, pd.Series([]))
            ax.barh(skill_counts.index, skill_counts.values, color='#0077B5')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False) 
            return fig
        

    @output
    @render.plot
    def plot51():
        degree_level = input.degree_level()
        if degree_level:
            fig, ax = plt.subplots(figsize=(10, 6))
            if degree_level == 'bachelor':
                top_skills = bachelor_skill_counts
                color = '#0077B5'
            elif degree_level == 'master':
                top_skills = master_skill_counts
                color = '#0077B5'
            elif degree_level == 'phd':
                top_skills = phd_skill_counts
                color = '#0077B5'
            ax.barh(top_skills.index, top_skills.values, color=color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False) 
            return fig
        
    @output
    @render.plot
    def plot61():
        work_mode = input.work_mode()
        if work_mode:
            fig, ax = plt.subplots(figsize=(10, 6))
            skill_counts = skill_counts_by_work_mode.get(work_mode, pd.Series([]))
            ax.barh(skill_counts.index, skill_counts.values, color='#0077B5')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False) 

            return fig
        
    @output
    @render.plot
    def plot7():
        top_5_companies.sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        top_5_companies.plot(kind='barh', color='#0077B5', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')  # Center-align labels
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)   # Remove y-axis label
        return fig

    @output
    @render.plot
    def plot8():
        top_10_companies.sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10_companies.plot(kind='barh', color='#0077B5', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')  # Center-align labels
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False) 
        return fig
    
    @output
    @render.plot
    def plot9():
        fig, ax = plt.subplots(figsize=(10, 6))
        industry_counts_sorted.head(10).plot(kind='barh', color='#0077B5')
        ax.set_xticklabels(ax.get_xticklabels(), ha='center')  # Center-align labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False) 
        return fig
    
    @reactive.Calc
    def selected_industry():
        return input.industry()
    
    @output
    @render.plot
    def plot10():
        industry = selected_industry()
        if industry:
            # Sort the skills in descending order
            skills = top_skills_per_industry[industry].sort_values()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(skills.index, skills.values, color='#0077B5')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            return fig

    @output
    @render.plot
    def plot11():
        fig, ax = plt.subplots(figsize=(10, 6))
        location_counts.plot(kind='barh', color='#0077B5', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right')
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        return fig
    
    @render_widget
    def map():
        # Create the scatter plot with bubble size adjusted based on frequency
        fig = px.scatter_geo(
            city_coordinates_df,
            lat='Latitude',
            lon='Longitude',
            hover_name='City',
            size='job_count',
            projection="mercator",  
            color_continuous_scale='dark',
        )

        # Set the map center to India and adjust the zoom level
        fig.update_geos(
            projection_type="mercator",
            center=dict(lat=20.5937, lon=78.9629), 
            projection=dict(scale=15)
        )

        return fig  # Plotly figure object returned as widget

    output.map = map 

#HYPOTHESIS TEST

    @output()
    @render.text
    def industry_results():
        df['Actively recruiting'].fillna('No', inplace=True)
        contingency_table_industry = pd.crosstab(df['Industry'], df['Actively recruiting'])
        chi2_industry, p_industry, dof_industry, expected_industry = chi2_contingency(contingency_table_industry)
        result = (f"Chi-square statistic: {chi2_industry:.2f}\n"
                  f"p-value: {p_industry:.5f}\n"
                  f"Degrees of freedom: {dof_industry}\n"
                  f"{'Reject the null hypothesis.' if p_industry < 0.05 else 'Fail to reject the null hypothesis.'}")
        return result

    @output()
    @render.text
    def location_results():
        df['Actively recruiting'].fillna('No', inplace=True)
        contingency_table_location = pd.crosstab(df['Location'], df['Actively recruiting'])
        chi2_location, p_location, dof_location, expected_location = chi2_contingency(contingency_table_location)
        result = (f"Chi-square statistic: {chi2_location:.2f}\n"
                  f"p-value: {p_location:.5f}\n"
                  f"Degrees of freedom: {dof_location}\n"
                  f"{'Reject the null hypothesis.' if p_location < 0.05 else 'Fail to reject the null hypothesis.'}")
        return result

    @output()
    @render.text
    def experience_results():
        df['Required Experience'].fillna('Unknown', inplace=True)
        contingency_table_experience = pd.crosstab(df['Required Experience'], df['Industry'])
        chi2_experience, p_experience, dof_experience, expected_experience = chi2_contingency(contingency_table_experience)
        result = (f"Chi-square statistic: {chi2_experience:.2f}\n"
                  f"p-value: {p_experience:.5f}\n"
                  f"Degrees of freedom: {dof_experience}\n"
                  f"{'Reject the null hypothesis.' if p_experience < 0.05 else 'Fail to reject the null hypothesis.'}")
        return result

    @output()
    @render.text
    def degree_results():
        df['Degree'] = df['bachelor'] + df['master'] + df['phd']
        contingency_table_degree = pd.crosstab(df['Degree'], df['Industry'])
        chi2_degree, p_degree, dof_degree, expected_degree = chi2_contingency(contingency_table_degree)
        result = (f"Chi-square statistic: {chi2_degree:.2f}\n"
                  f"p-value: {p_degree:.5f}\n"
                  f"Degrees of freedom: {dof_degree}\n"
                  f"{'Reject the null hypothesis.' if p_degree < 0.05 else 'Fail to reject the null hypothesis.'}")
        return result


# Create and run the Shiny app
app = App(app_ui, server, debug=True)

if __name__ == "__main__":
    app.run(port=8001)