import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances


def recommender_main(blockbuster):
    
    ## Creating a movie and series subset
    blockbuster_movie = blockbuster[blockbuster['Series or Movie'] == 'Movie'].reset_index(drop = True)
    blockbuster_series = blockbuster[blockbuster['Series or Movie'] == 'Series'].reset_index(drop = True)
    
    ## Sorting the data by available language
    blockbuster_movie = blockbuster_movie.sort_values('Languages').reset_index(drop = True)
    blockbuster_series = blockbuster_series.sort_values('Languages').reset_index(drop = True)
    
#     ## Removing unnecessary variables for recommendation process
#     blockbuster_movie_temp = blockbuster_movie.iloc[:, 14:62]
#     blockbuster_series_temp = blockbuster_series.iloc[:, 14:62]
    
    ## Calling the recommendation subsettor function
    final_movie_recommendations = recommendation_subsettor(blockbuster_movie)
    final_series_recommendations = recommendation_subsettor(blockbuster_series)
    
#     ## Returning the final data
#     blockbuster_movie = pd.concat([blockbuster_movie, final_movie_recommendations], axis = 1)
#     blockbuster_series = pd.concat([blockbuster_series, final_series_recommendations], axis = 1)
    
#     final_blockbuster = pd.concat([blockbuster_movie, blockbuster_series]).reset_index(drop = True)
    
    return final_movie_recommendations, final_series_recommendations
    # return final_blockbuster
    
    
    
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
            
    # ## Calling the recommendation system function
    # for language in languages:
    #     language_subset = blockbuster[blockbuster['Languages'] == language].reset_index(drop = True)
    #     recommendations = get_recommendations(language_subset)
    #     results = pd.concat([results, get_recommendations(language_subset)])
    return languages 
    # return results
    


# def get_recommendations(blockbuster_temps):
    
#     ## FOR each item in blockbuster