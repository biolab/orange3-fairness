"""
Utility functions for testing.
"""

from Orange.evaluation import scoring

from orangecontrib.fairness.evaluation import scoring as bias_scoring


def print_metrics(results, bias=True):
    """
    Print the metrics of the results.

    This function prints the metrics of the results.
    If bias is True, it will also print the bias metrics.

    Args:
        results (Results): The results of the model.
        bias (bool): If True, print the bias metrics.
    """
    print(f"ROC AUC: {scoring.AUC(results)}")
    print(f"CA: {scoring.CA(results)}")
    print(f"F1: {scoring.F1(results)}")
    print(f"Precision: {scoring.Precision(results)}")
    print(f"Recall: {scoring.Recall(results)}")
    if bias:
        print(f"SPD: {bias_scoring.StatisticalParityDifference(results)}")
        print(f"EOD: {bias_scoring.EqualOpportunityDifference(results)}")
        print(f"AOD: {bias_scoring.AverageOddsDifference(results)}")
        print(f"DI: {bias_scoring.DisparateImpact(results)}")


def fairness_attributes(domain):
    """
    Get the fairness attributes from the domain.

    This function gets the favorable class value, protected attribute and
    privileged protected attribute values from the domain.

    Args:
        domain (Domain): The domain of the data.
    """
    favorable_class_value = None
    protected_attribute = None
    privileged_pa_values = None
    if "favorable_class_value" in domain.class_var.attributes:
        favorable_class_value = domain.class_var.attributes["favorable_class_value"]
        for var in domain.attributes:
            if "privileged_pa_values" in var.attributes:
                protected_attribute = var
                privileged_pa_values = var.attributes["privileged_pa_values"]
                break
    return favorable_class_value, protected_attribute, privileged_pa_values
