from abc import abstractmethod
from Orange.data import DiscreteVariable, ContinuousVariable, Domain
from Orange.evaluation.scoring import Score

from aif360.metrics import ClassificationMetric

from orangedemo.utils import table_to_standard_dataset, contains_fairness_attributes


__all__ = [
    "StatisticalParityDifference",
    "EqualOpportunityDifference",
    "AverageOddsDifference",
    "DisparateImpact",
]


class FairnessScorer(Score, abstract=True):
    class_types = (
        DiscreteVariable,
        ContinuousVariable,
    )

    # This is a static method, it is called by the Orange framework to check if the scorer is compatible with the domain of the data
    @staticmethod
    def is_compatible(domain: Domain) -> bool:
        return contains_fairness_attributes(domain)

    # This method is called by the Orange framework, together with the metric method it computes the specific score
    def compute_score(self, results):

        dataset, privileged_groups, unprivileged_groups = table_to_standard_dataset(results.data)
        # We need to subset the created dataset so that it will match the shape/order of the predictions if some of the data was used multiple times
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

    # This indicates that the metric method needs to be implemented by the subclasses
    @abstractmethod
    def metric(self, classification_metric):
        pass


class StatisticalParityDifference(FairnessScorer):
    name = "SPD"
    long_name = "Statistical Parity Difference"

    def metric(self, classificationMetric):
        return classificationMetric.statistical_parity_difference()


class EqualOpportunityDifference(FairnessScorer):
    name = "EOD"
    long_name = "Equal Opportunity Difference"

    def metric(self, classificationMetric):
        return classificationMetric.equal_opportunity_difference()


class AverageOddsDifference(FairnessScorer):
    name = "AOD"
    long_name = "Average Odds Difference"

    def metric(self, classificationMetric):
        return classificationMetric.average_odds_difference()


class DisparateImpact(FairnessScorer):
    name = "DI"
    long_name = "Disparate Impact"

    def metric(self, classificationMetric):
        return classificationMetric.disparate_impact()
