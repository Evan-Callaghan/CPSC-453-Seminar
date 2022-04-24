## Blockbuster Recommendation System Project

## This .py file goes through the recommendation system algorithm and produces the top-five recommendations for every series and movie in the data set

## Importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def recommender_main(blockbuster):
    
    ## Printing signal to indicate the algorithm has started:
    print('-- Algorithm Starting --')
    
    ## Changing the data type of the Genre variable
    for i in range(0, blockbuster.shape[0]):
        blockbuster.at[i, 'Genre'] = blockbuster.at[i, 'Genre'].replace("[", '')
        blockbuster.at[i, 'Genre'] = blockbuster.at[i, 'Genre'].replace("'", '')
        blockbuster.at[i, 'Genre'] = blockbuster.at[i, 'Genre'].replace("]", '')
        blockbuster.at[i, 'Genre'] = list(blockbuster.at[i, 'Genre'].split(", "))
    
    ## Creating a movie and series subset
    blockbuster_movie = blockbuster[blockbuster['Series or Movie'] == 'Movie'].reset_index(drop = True)
    blockbuster_series = blockbuster[blockbuster['Series or Movie'] == 'Series'].reset_index(drop = True)
    
    ## Sorting the data by available language
    blockbuster_movie = blockbuster_movie.sort_values('Languages').reset_index(drop = True)
    blockbuster_series = blockbuster_series.sort_values('Languages').reset_index(drop = True)
    
    ## Calling the recommendation subsettor function
    final_movie_recommendations = recommendation_subsettor(blockbuster_movie)
    final_series_recommendations = recommendation_subsettor(blockbuster_series)
    
    ## Returning the final data
    blockbuster_movie = pd.concat([blockbuster_movie, final_movie_recommendations], axis = 1)
    blockbuster_series = pd.concat([blockbuster_series, final_series_recommendations], axis = 1)
    
    ## Printing signal to indicate recommendations have been completed:
    print('\n-- Recommendations Completed --')
    
    ## Calling the final explode fucntion
    blockbuster_movie = final_explode(blockbuster_movie)
    blockbuster_series = final_explode(blockbuster_series)
    
    ## Combining the movie and series data sets into one
    final_blockbuster = pd.concat([blockbuster_movie, blockbuster_series]).reset_index(drop = True)
    
    ## Keeping only necessary variables
    final_blockbuster = final_blockbuster[['Title', 'Genre', 'Languages', 'Series or Movie', 'View Rating', 'Popularity_Score', 'Netflix Link', 'IMDb Link', 'Summary', 'Image', 'Poster', 'Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5']]
    
    ## Printing signal to indicate the algorithm has finished:
    print('\n-- Algorithm Finished --')
    
    ## Returning the complete data set for user interface usage
    return final_blockbuster
    
    
    
def recommendation_subsettor(blockbuster):
    
    ## Defining the number of rows
    n = blockbuster.shape[0]
    
    ## Defining an empty list for results
    languages = []
    
    ## Defining an empty data-frame to store results
    results = pd.DataFrame(columns = ['Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5'])
    
    ## Recording all languages in the data set
    for i in range(0, n):
        
        if np.isin(blockbuster.at[i, 'Languages'], languages, invert = True):
            languages.append(blockbuster.at[i, 'Languages'])
            languages.sort()
            
    ## Calling the recommendation system function
    for language in languages:
        
        language_subset = blockbuster[blockbuster['Languages'] == language].reset_index(drop = True)
        results = pd.concat([results, get_recommendations(language_subset)], ignore_index = True)
    
    return results
    


def get_recommendations(blockbuster):
    
    ## Defining an empty data-frame for results
    results = pd.DataFrame(columns = ['Rec_1', 'Rec_2', 'Rec_3', 'Rec_4', 'Rec_5'])
    
    ## Removing unnecessary variables for recommendation process
    blockbuster_temp = blockbuster.iloc[:, 12:]
    
    ## Computing the Euclidean distances for all observations
    D = euclidean_distances(blockbuster_temp)
    
    ## Extracting the Top-5 recommendations for each item
    for i in range(0, blockbuster_temp.shape[0]):
        
        top_10 = np.argsort(D[:, i])[1:11]
        temp = pd.DataFrame(columns = ['Indices', 'Popularity'])
        temp['Indices'] = top_10
        temp['Popularity'] = [blockbuster.loc[top_10[0], 'Popularity_Score'], blockbuster.loc[top_10[1], 'Popularity_Score'], blockbuster.loc[top_10[2], 'Popularity_Score'], blockbuster.loc[top_10[3], 'Popularity_Score'], blockbuster.loc[top_10[4], 'Popularity_Score'], blockbuster.loc[top_10[5], 'Popularity_Score'], blockbuster.loc[top_10[6], 'Popularity_Score'], blockbuster.loc[top_10[7], 'Popularity_Score'], blockbuster.loc[top_10[8], 'Popularity_Score'], blockbuster.loc[top_10[9], 'Popularity_Score']]
        
        temp = temp.sort_values(by = 'Popularity', axis = 0, ascending = False)
        temp_index = temp['Indices'].tolist()
        
        results.loc[i] = [blockbuster.loc[temp_index[0], 'Title'], blockbuster.loc[temp_index[1], 'Title'], blockbuster.loc[temp_index[2], 'Title'], blockbuster.loc[temp_index[3], 'Title'],  blockbuster.loc[temp_index[4], 'Title']]
        
    return results



def final_explode(blockbuster):
    
    ## Using the explode function to get a unique observation for each genre
    blockbuster_temp = blockbuster.explode('Genre').reset_index(drop = True)
    
    ## Removing observations with missing genres
    results = blockbuster_temp[blockbuster_temp['Genre'] != 'nan'].reset_index(drop = True)
    
    return results