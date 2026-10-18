"""Display/export order changes only; sample identity and predictions stay paired."""
from dataclasses import replace
from pathlib import Path
import csv
from io import StringIO

from backend.app.runs.comparison_figures import figure_data, sample_indices
from backend.app.runs.batch_archive import _csv
from backend.app.runs.batch_projection import project_model_comparison
from backend.app.runs.artifacts import RunArtifactWriter
from backend.tests.test_training_batches import _repository, _batch


def test_existing_lexical_payload_is_sorted_without_reassigning_predictions():
    ids=['1','10','11','2','3','S10','S2','S1']
    values=[1,0,None,0,1,0,1,1]
    data={'comparable':True,'models':[{'model_type':'svm','metrics':{}}],
          'sample_correctness':{'sample_ids':ids,'values':[{'model_type':'svm','values':values}]}}
    spec=figure_data(data,kind='samples')
    assert spec['columns']==['1','2','3','10','11','S1','S2','S10']
    assert spec['values']==[[1,0,1,0,None,1,1,0]]
    exported=list(csv.reader(StringIO(_csv(spec).decode('utf-8-sig'))))
    assert exported[0][1:]==spec['columns']
    assert [ids[i] for i in sample_indices(data,errors=True)]==['2','10','S10']
    assert ids==['1','10','11','2','3','S10','S2','S1']


def test_projection_returns_numeric_sample_order_and_matching_details(tmp_path):
    repository=_repository(tmp_path);batch,records=_batch(repository)
    record=records[0];writer=RunArtifactWriter(tmp_path/record.run_id)
    writer.write_json('metrics.json',{'test':{'accuracy':.5,'balanced_accuracy':.5,'macro_f1':.5,'weighted_f1':.5,'confusion_matrix':[[2,2],[0,0]],'classification_report':{}}})
    writer.write_json('label_map.json',{'0':'A','1':'B'})
    writer.write_json('split.json',[{'fold_index':1,'train_sample_ids':['S1'],'valid_sample_ids':['S2'],'test_sample_ids':['1','2','3','10']}])
    writer.write_bytes('predictions.csv',b'Sample_ID,true_label,pred_label,prob_A,prob_B\n1,A,A,0.9,0.1\n10,A,B,0.1,0.9\n2,A,B,0.2,0.8\n3,A,A,0.8,0.2\n')
    writer.finalize(run_id=record.run_id)
    result=project_model_comparison(batch_id=batch.batch_id,records=[replace(record,state='succeeded')],run_dir_for=lambda rid:tmp_path/rid)
    source=result['sample_correctness']
    assert source['sample_ids']==['1','2','3','10']
    assert source['values'][0]['values']==[1,0,1,0]
    assert [row['pred_label'] for row in source['values'][0]['details']]==['A','B','A','B']
