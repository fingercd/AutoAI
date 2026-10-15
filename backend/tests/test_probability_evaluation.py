import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from backend.app.probability_evaluation import grouped_balanced_accuracy,grouped_probability_predictions

class EvaluationTests(unittest.TestCase):
 def test_single_rows_use_probability_not_native_labels(self):
  rows=[dict(Sample_ID='a',true_label='0',pred_label='1',prob_0=.8,prob_1=.2),dict(Sample_ID='b',true_label='1',pred_label='0',prob_0=.1,prob_1=.9)]
  score,groups=grouped_balanced_accuracy(rows,['0','1'],['a','b']);self.assertEqual(score,1);self.assertEqual([x['predicted_label'] for x in groups],['0','1'])
 def test_repeated_measurements_are_averaged_before_argmax(self):
  rows=[dict(Sample_ID='a',true_label='0',prob_0=.51,prob_1=.49),dict(Sample_ID='a',true_label='0',prob_0=.01,prob_1=.99),dict(Sample_ID='b',true_label='1',prob_0=.1,prob_1=.9)]
  score,groups=grouped_balanced_accuracy(rows,['0','1']);self.assertEqual(score,.5);self.assertEqual(len(groups),2);self.assertAlmostEqual(groups[0]['probabilities']['0'],.26)
 def test_exact_ties_preserve_frozen_label_order(self):
  self.assertEqual(grouped_probability_predictions([dict(Sample_ID='a',true_label='z',prob_z=.5,prob_a=.5)],['z','a'])[0]['predicted_label'],'z')
 def test_partition_coverage_mismatch_rejected(self):
  with self.assertRaises(ValueError):grouped_probability_predictions([dict(Sample_ID='a',true_label='0',prob_0=.5,prob_1=.5)],['0','1'],['b'])
 def test_inconsistent_group_labels_rejected(self):
  with self.assertRaises(ValueError):grouped_probability_predictions([dict(Sample_ID='a',true_label=x,prob_0=.5,prob_1=.5) for x in ['0','1']],['0','1'])
 def test_invalid_external_probabilities_rejected(self):
  for p in [float('nan'),-.1,1.1]:
   with self.assertRaises(ValueError):grouped_probability_predictions([dict(Sample_ID='a',true_label='0',prob_0=p,prob_1=.5)],['0','1'])
 def test_no_changed_test_semantics(self):
  import csv,tempfile
  from backend.app.benchmark_driver import grouped_predictions
  rows=[dict(Sample_ID='a',true_label='0',prob_0=.8,prob_1=.2),dict(Sample_ID='a',true_label='0',prob_0=.2,prob_1=.8),dict(Sample_ID='b',true_label='1',prob_0=.1,prob_1=.9)]
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/'predictions.csv'
   with path.open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
   self.assertEqual(grouped_probability_predictions(rows,['0','1'],['a','b']),grouped_predictions(path,['0','1'],['a','b']))
if __name__=='__main__':unittest.main(verbosity=2)
