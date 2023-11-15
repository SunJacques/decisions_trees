from typing import List
import numpy as np

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.points = PointSet(features, labels, types, min_split_points)
        features = np.array(features)
        labels = np.array(labels)
        self.h = h
        self.types = types
        self.left_node = None
        self.right_node = None
        self.node = None
        self.left_split = self.right_split = []
        feature_left_split = feature_right_split = []
        
        
        self.feature_selected, gain = self.points.get_best_gain()
        self.feature_threshold = self.points.get_best_threshold()
        
        #Test if there exist a split such that the children node are superior to the min_split_points
        if self.feature_selected is None:
            self.node = PointSet(features, labels, types)
        else : 
            if self.types[self.feature_selected] == FeaturesTypes.BOOLEAN:
                condition = (features[:, self.feature_selected] == 1)
                
            elif self.types[self.feature_selected] == FeaturesTypes.CLASSES:
                condition = (features[:, self.feature_selected] == self.feature_threshold)

            elif self.types[self.feature_selected] == FeaturesTypes.REAL:
                condition = (features[:, self.feature_selected] < self.feature_threshold)
                
            self.left_split = labels[condition]
            feature_left_split = features[condition]

            self.right_split = labels[~condition]
            feature_right_split = features[~condition]
            
            if h != 0 and self.points.get_gini() > 0 and len(self.left_split) >= min_split_points and len(self.right_split) >= min_split_points:
                self.left_node = Tree(feature_left_split, self.left_split,types, h - 1, min_split_points)
                self.right_node = Tree(feature_right_split, self.right_split,types, h - 1, min_split_points)
            else:
                self.node = PointSet(features, labels, types, min_split_points)

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        if self.h == 0 or self.left_node is None or self.right_node is None:
            #return majority of label of a node
            return np.argmax(np.bincount(self.node.labels))
                
        else:
            if self.types[self.feature_selected] == FeaturesTypes.CLASSES:
                if features[self.feature_selected] ==  self.feature_threshold:
                    return self.left_node.decide(features)
                else:
                    return self.right_node.decide(features)
                
            elif self.types[self.feature_selected] == FeaturesTypes.REAL:
                if features[self.feature_selected] < self.feature_threshold:
                    return self.left_node.decide(features)
                else:
                    return self.right_node.decide(features)
                
            else:
                if features[self.feature_selected] == 1:
                    return self.left_node.decide(features)
                else:
                    return self.right_node.decide(features)