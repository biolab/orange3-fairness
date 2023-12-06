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
    long_name = str(
        "<p>Statistical Parity Difference (SPD): Measures the difference in ratios of "
        "favorable outcomes. An ideal value is 0.0.</p>"
        "<ul>"
        "<li>SPD &lt; 0: The privileged group has a higher rate of favorable outcomes.</li>"
        "<li>SPD &gt; 0: The privileged group has a lower rate of favorable outcomes.</li>"
        "</ul>"
    )

    def metric(self, classification_metric):
        return classification_metric.statistical_parity_difference()


class EqualOpportunityDifference(FairnessScorer):
    """Class for Equal Opportunity Difference fairness scoring."""

    name = "EOD"
    long_name = str(
        "<p>Equal Opportunity Difference (EOD): It measures the difference in "
        "true positive rates. An ideal value is 0.0, indicating the difference "
        "in true positive rates is the same for both groups.</p>"
        "<ul>"
        "<li>EOD &lt; 0: The privileged group has a higher true positive rate.</li>"
        "<li>EOD &gt; 0: The privileged group has a lower true positive rate.</li>"
        "</ul>"
    )

    def metric(self, classification_metric):
        return classification_metric.equal_opportunity_difference()


class AverageOddsDifference(FairnessScorer):
    """Class for Average Odds Difference fairness scoring."""

    name = "AOD"
    long_name = str(
        "<p>Average Odds Difference (AOD): This metric calculates the average difference "
        "between the true positive rates (correctly predicting a positive outcome) and false "
        "positive rates (incorrectly predicting a positive outcome) for both the privileged "
        "and unprivileged groups. A value of 0.0 indicates equal rates for both groups, "
        "signifying fairness.</p>"
        "<ul>"
        "<li>AOD &lt; 0: Indicates bias in favor of the privileged group.</li>"
        "<li>AOD &gt; 0: Indicates bias against the privileged group.</li>"
        "</ul>"
    )

    def metric(self, classification_metric):
        return classification_metric.average_odds_difference()


class DisparateImpact(FairnessScorer):
    """Class for Disparate Impact fairness scoring."""

    name = "DI"
    long_name = str(
        "<p>Disparate Impact (DI): The ratio of ratios of favorable outcomes for an unprivileged "
        "group to that of the privileged group. An ideal value of 1.0 means the ratio is "
        "the same for both groups.</p>"
        "<ul>"
        "<li>DI &lt; 1.0: The privileged group receives favorable outcomes at a higher rate "
        "than the unprivileged group.</li>"
        "<li>DI &gt; 1.0: The privileged group receives favorable outcomes at a lower rate "
        "than the unprivileged group.</li>"
        "</ul>"
    )

    # TODO: When using randomize, models sometimes predict the same class for all instances
    # This can lead to division by zero in the Disparate Impact score (and untrue results for the other scores)
    # What is the best way to handle this?
    def metric(self, classification_metric):
        return classification_metric.disparate_impact()
            