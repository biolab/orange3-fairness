from Orange.data import Table
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_rows
from Orange.evaluation import scoring

from orangedemo.evaluation import scoring as bias_scoring


def as_fairness_setup(self):
    test_data = Table(f"{self.test_data_path}/adult.tab")
    self.send_signal(
        self.as_fairness.Inputs.data,
        test_data,
        widget=self.as_fairness,
    )
    simulate.combobox_activate_item(
        self.as_fairness.controls.favorable_class_value, ">50K"
    )
    simulate.combobox_activate_item(
        self.as_fairness.controls.protected_attribute, "sex"
    )
    select_rows(self.as_fairness.controls.privileged_PA_values, [1])
    output_data = self.get_output(self.as_fairness.Outputs.data)
    return output_data


def print_metrics(results, bias=True):
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
