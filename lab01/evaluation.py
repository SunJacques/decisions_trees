from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    TP = FP = FN = 0
    for v in zip(expected_results,actual_results):
        if v == (1,1):
            TP += 1
        elif v == (0,1):
            FP += 1
        elif v == (1,0):
            FN += 1
    
    return TP / (TP + FP), TP / (TP + FN)
    

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    p , r = precision_recall(expected_results, actual_results)
    return 2 * r * p / (r + p)
