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
                 h: int = 1):
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
        self.points = PointSet(features, labels, types)
        features = np.array(features)
        labels = np.array(labels)
        self.h = h
        self.types = types
        self.left = None
        self.right = None
        self.node = None
        self.node_left = self.node_right = []
        feature_left = feature_right = []
        
        gain = self.points.get_best_gain()
        self.feature_selected, self.feature_class_value = self.points.get_best_threshold()
        
        
        if self.types[self.feature_selected] == FeaturesTypes.BOOLEAN:
            self.node_left = labels[features[:, self.feature_selected] == 1]
            feature_left = features[features[:, self.feature_selected] == 1]
            
            self.node_right = labels[features[:, self.feature_selected] == 0]
            feature_right = features[features[:, self.feature_selected] == 0]
            
            
        elif self.types[self.feature_selected] == FeaturesTypes.CLASSES:
            self.node_left = labels[features[:, self.feature_selected] == self.feature_class_value]
            feature_left = features[features[:, self.feature_selected] == self.feature_class_value]
            
            self.node_right = labels[features[:, self.feature_selected] != self.feature_class_value]
            feature_right = features[features[:, self.feature_selected] != self.feature_class_value]

        
        if h != 0 and self.points.get_gini() > 0:
            self.left = Tree(feature_left, self.node_left,types, h - 1)
            self.right = Tree(feature_right, self.node_right,types, h - 1)
        else:
            self.node = PointSet(features, labels, types)
            
        

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
        if self.h == 0 or self.left is None or self.right is None:
            if self.types[self.feature_selected] == FeaturesTypes.CLASSES:
                if features[self.feature_selected == self.feature_class_value]:
                    return np.argmax(np.bincount(self.node.labels))
                else:
                    return np.argmax(np.bincount(self.node.labels))
                
            else:
                if features[self.feature_selected]:
                    return np.argmax(np.bincount(self.node.labels))
                else:
                    return np.argmax(np.bincount(self.node.labels))
                
        else:
            if self.types[self.feature_selected] == FeaturesTypes.CLASSES:
                if features[self.feature_selected] ==  self.feature_class_value:
                    return self.left.decide(features)
                else:
                    return self.right.decide(features)
            else:
                if features[self.feature_selected]:
                    return self.left.decide(features)
                else:
                    return self.right.decide(features)