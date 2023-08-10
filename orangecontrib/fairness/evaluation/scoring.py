from numpy import unique
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
    """Abstract class which will allow fairness scores to be calculated and displayed in certain widgets"""

    class_types = (
        DiscreteVariable,
        ContinuousVariable,
    )

    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        """Checks if the scorer is compatible with the domain of the data. If not the scores will not be computed."""
        return contains_fairness_attributes(domain)

    def compute_score(self, results):
        """Method that creates a ClassificationMetric object used to compute fairness scores"""

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
        """Method that needs to be implemented by the subclasses of the FairnessScorer."""
        pass


class StatisticalParityDifference(FairnessScorer):
    """Class for Statistical Parity Difference fairness scoring."""

    name = "SPD"
    long_name = "Statistical Parity Difference"

    def metric(self, classification_metric):
        return classification_metric.statistical_parity_difference()


class EqualOpportunityDifference(FairnessScorer):
    """Class for Equal Opportunity Difference fairness scoring."""

    name = "EOD"
    long_name = "Equal Opportunity Difference"

    def metric(self, classification_metric):
        return classification_metric.equal_opportunity_difference()


class AverageOddsDifference(FairnessScorer):
    """Class for Average Odds Difference fairness scoring."""

    name = "AOD"
    long_name = "Average Odds Difference"

    def metric(self, classification_metric):
        return classification_metric.average_odds_difference()


class DisparateImpact(FairnessScorer):
    """Class for Disparate Impact fairness scoring."""

    name = "DI"
    long_name = "Disparate Impact"

    # TODO: When using randomize, models sometimes predict the same class for all instances
    # This can lead to division by zero in the Disparate Impact score (and untrue results for the other scores)
    # What is the best way to handle this?
    def metric(self, classification_metric):
        return classification_metric.disparate_impact()
            