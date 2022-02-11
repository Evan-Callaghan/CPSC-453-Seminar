# Blockbuster Recommendation System Project

## Project Description:
After the several features are engineered, we are ready to start implementing our first recommendation engine. The recommendation system relies on similarity, which can be measured with the Euclidean distance, cosine similarity or Manhattan distance, and transaction activity. 

## Algorithm Design:

```
Recommendations:
  
  FOR each item in blockbuster.csv
  
    IF the item is a movie
      THEN return the five-nearest neighbors as the recommendations from a movie data subset;
      
    ELSE (the item is a series)
      THEN return the five-nearest neighbors as the recommendations from a series data subset;
    
    ENDIF;
  
  ENDFOR;

END.
```
