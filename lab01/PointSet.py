from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, 
                 features: List[List[float]], 
                 labels: List[bool], 
                 types: List[FeaturesTypes], 
                 min_split_points: int = 1):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """

        self.types = types
        self.min_split_points = min_split_points
        self.features = np.array(features)
        self.labels = np.array(labels)
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        gini = 1
        n = self.labels.size
        for label in set(self.labels):
            #Count the number of True label
            n1 = np.count_nonzero(self.labels == label)
            gini -= (n1/n) ** 2 
        
        return gini
        
    
    def get_best_threshold(self) -> float:
        return self.best_feature_value

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        best_split = np.inf
        best_feature = None
        best_feature_value = None
        
        for feature_number in range(self.features[0].size):
            gini_split = np.inf
            
            if self.types[feature_number] == FeaturesTypes.BOOLEAN:
                # Separating the points according to their feature value
                node1 = self.labels[self.features[:, feature_number] == 1]
                node2 = self.labels[self.features[:, feature_number] == 0]
                n1 = node1.size
                n2 = node2.size
                
                if n1 >= self.min_split_points and n2>= self.min_split_points:
                    g1 = PointSet([], node1, []).get_gini()
                    g2 = PointSet([], node2, []).get_gini()
                    
                    gini_split = n1/(n1+n2) * g1 + n2/(n1+n2) * g2
                
            elif self.types[feature_number] == FeaturesTypes.CLASSES:
                #Looking for the best split with one child of size 1
                for feature_value in np.unique(self.features[:,feature_number]):
                    condition = (self.features[:,feature_number] == feature_value)
                    node1 = self.labels[condition]
                    node2 = self.labels[~condition]
                    n1 = node1.size
                    n2 = node2.size
                    
                    #Compute Gini index
                    g1 = PointSet([], node1, []).get_gini()
                    g2 = PointSet([], node2, []).get_gini()
                    
                    inner_gini_split = n1/(n1+n2) * g1 + n2/(n1+n2) * g2
                    
                    #Find the min split between the different binary split
                    if inner_gini_split < gini_split and n1 >= self.min_split_points and n2>= self.min_split_points:
                        gini_split = inner_gini_split
                        best_inner_feature_value = feature_value
            
            #Best split for continuous value      
            elif self.types[feature_number] == FeaturesTypes.REAL :
                for feature_value in np.unique(self.features[:, feature_number]):
                    condition = (self.features[:,feature_number] < feature_value)
                    node1 = self.labels[condition]
                    node2 = self.labels[~condition]
                    n1 = node1.size
                    n2 = node2.size

                    #Compute Gini index
                    g1 = PointSet([], node1, []).get_gini()
                    g2 = PointSet([], node2, []).get_gini()
                    
                    try:
                        L = self.features[condition, feature_number].max()
                    except:
                        L = 0
                    try:
                        R = self.features[~condition, feature_number].min()
                    except:
                        R = 0

                    inner_gini_split = n1/(n1+n2) * g1 + n2/(n1+n2) * g2
                    
                    if inner_gini_split < gini_split and n1 >= self.min_split_points and n2>= self.min_split_points:
                        gini_split = inner_gini_split
                        best_inner_feature_value = (L+R)/2
            
            # Look for the minimum split between the features
            if gini_split < best_split:
                best_split = gini_split
                
                if self.types[feature_number] == FeaturesTypes.CLASSES or self.types[feature_number] == FeaturesTypes.REAL:
                    best_feature = feature_number
                    best_feature_value = best_inner_feature_value
                else:
                    best_feature = feature_number
                    best_feature_value = None
                    
        self.best_gain = self.get_gini() - best_split
    
        self.best_feature = best_feature
        self.best_feature_value = best_feature_value
            
        return self.best_feature, self.best_gain
            
            
            

