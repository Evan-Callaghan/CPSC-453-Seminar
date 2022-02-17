import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def recommender_main(blockbuster):
    
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
    
    final_blockbuster = pd.concat([blockbuster_movie, blockbuster_series]).reset_index(drop = True)
    
    return final_blockbuster
    
    
    
def recommendation_subsettor(blockbuster):
    
    ## Defining the number of rows
    n = blockbuster.shape[0]
    
    ## Defining an empty list for results
    languages = []
    
    ## Defining an empty data-frame for results
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
    blockbuster_temp = blockbuster.iloc[:, 11:]
    
    ## Computing the Euclidean distances for all observations
    D = euclidean_distances(blockbuster_temp)
    
    ## Extracting the Top-5 recommendations for each item
    for i in range(0, blockbuster_temp.shape[0]):
        
        top_5 = np.argsort(D[:, i])[1:6]
        results.loc[i] = [blockbuster.loc[top_5[0], 'Title'], blockbuster.loc[top_5[1], 'Title'], blockbuster.loc[top_5[2], 'Title'], blockbuster.loc[top_5[3], 'Title'], blockbuster.loc[top_5[4], 'Title']]
        
        
    return results