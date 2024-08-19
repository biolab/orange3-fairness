"""
This module contains classes for computing fairness scores.

Classes:
- StatisticalParityDifference
- EqualOpportunityDifference
- AverageOddsDifference
- DisparateImpact
"""

from abc import abstractmethod
from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.evaluation.scoring import Score

from aif360.metrics import ClassificationMetric

from orangecontrib.fairness.widgets.utils import (
    table_to_standard_dataset,
    contains_fairness_attributes,
)


__all__ = [
    "StatisticalParityDifference",
    "EqualOpportunityDifference",
    "AverageOddsDifference",
    "DisparateImpact",
]


class FairnessScorer(Score, abstract=True):
    """
    Abstract class for computing fairness scores.

    Abstract class which will allow fairness scores to be calculated and displayed.
    Subclasses need to implement the metric method which will return the fairness score.
    """

    class_types = (
        DiscreteVariable,
        ContinuousVariable,
    )

    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        """
        Checks if the scorer is compatible with the domain of the data.
        If not the scores will not be computed.

        Args:
            domain (Domain): The domain of the data.
        """
        return contains_fairness_attributes(domain)

    def compute_score(self, results):
        """
        Creates a ClassificationMetric object used to compute fairness scores

        Args:
            results (Results): The results of the model.
        """

        dataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(
            results.data
        )

        # We need to subset the created dataset so that it will match the shape/order
        # This is needed when/if some of the rows in the data were used multiple times
        dataset = dataset.subset(results.row_indices)
        dataset_pred = dataset.copy()
        dataset_pred.labels = results.predicted

        classification_metric = ClassificationMetric(
            dataset,
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )
        return [self.metric(classification_metric)]

    @abstractmethod
    def metric(self, classification_metric):
        """
        Abstract method that needs to be implemented by subclasses.

        It should return the fairness score.

        Args:
            classification_metric (ClassificationMetric):
                The ClassificationMetric object used to compute fairness scores.
        """
        pass


class StatisticalParityDifference(FairnessScorer):
    """
    A class for computing the Statistical Parity Difference fairness score.
    """

    name = "SPD"
    long_name = str(
        "<p>Statistical Parity Difference (SPD): The difference in favorable "
        "outcomes proportions between groups. An ideal value is 0.0.</p>"
        "<ul>"
        "<li>SPD &lt; 0: The privileged group has a higher rate of favorable outcomes.</li>"
        "<li>SPD &gt; 0: The privileged group has a lower rate of favorable outcomes.</li>"
        "</ul>"
    )

    def metric(self, classification_metric):
        return classification_metric.statistical_parity_difference()


class EqualOpportunityDifference(FairnessScorer):
    """
    A class for computing the Equal Opportunity Difference fairness score.
    """

    name = "EOD"
    long_name = str(
        "<p>Equal Opportunity Difference (EOD): The difference in true positive rates between "
        "groups. An ideal value is 0.0 meaning both groups have the same true positive rate.</p>"
        "<ul>"
        "<li>EOD &lt; 0: The privileged group has a higher true positive rate.</li>"
        "<li>EOD &gt; 0: The privileged group has a lower true positive rate.</li>"
        "</ul>"
    )

    def metric(self, classification_metric):
        return classification_metric.equal_opportunity_difference()


class AverageOddsDifference(FairnessScorer):
    """
    A class for computing the Average Odds Difference fairness score.
    """

    name = "AOD"
    long_name = str(
        "<p>Average Odds Difference (AOD): The average of the differences in true "
        "and false positive rates between privileged and unprivileged groups. "
        "A value of 0.0 indicates equal rates for both groups.</p>"
        "<ul>"
        "<li>AOD &lt; 0: Indicates bias in favor of the privileged group.</li>"
        "<li>AOD &gt; 0: Indicates bias against the privileged group.</li>"
        "</ul>"
    )

    def metric(self, classification_metric):
        return classification_metric.average_odds_difference()


class DisparateImpact(FairnessScorer):
    """
    A class for computing the Disparate Impact fairness score.
    """

    name = "DI"
    long_name = str(
        "<p>Disparate Impact (DI) is the ratio of favorable outcome "
        "proportions between an unprivileged and privileged group. "
        "Value of 1.0 indicates that the ratio is equal for both groups.</p>"
        "<ul>"
        "<li>DI &lt; 1.0: The privileged group has a higher rate of favorable outcomes.</li>"
        "<li>DI &gt; 1.0: The privileged group has a lower rate of favorable outcomes.</li>"
        "</ul>"
    )

    # TODO: When using randomize, models sometimes predict the same class for all instances
    # This can lead to division by zero in the Disparate Impact score
    # and untrue results for the other scores.
    def metric(self, classification_metric):
        return classification_metric.disparate_impact()
