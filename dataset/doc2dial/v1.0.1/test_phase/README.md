# Shared Task of DialDoc 21

For Test Phase, there are three files provided.

**doc2dial_dial_finaltest.json** is the dialogue data, which is the same format as [training data](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/data/doc2dial/v1.0.1/doc2dial_dial_train.json) without the annotations. It shares the same [document data](https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/data/doc2dial/v1.0.1/doc2dial_doc.json).  

**test_subtask1_phase2_ids.json** contains the IDs for prediction in required result format for Subtask 1. You basically need to add the result of `prediction_text` and `no_answer_probability`. The threshold for an answer to be considered as no answer is 1.0. You can set  `no_answer_probability` as either 0 or 1.

**test_subtask2_phase2_ids.json** contains the IDs for prediction in required result format for Subtask 2. You basically need to add the result of `utterance`. 

**doc2dial_doc_with_unseen.json** contains the document data including the documents from unseen domain.



**Note:**

The `id ` in **test_subtask1_phase2_ids.json**  and **test_subtask2_phase2_ids.json** is `{dial_id}_{turn_id}`, where `turn_id` is of the turn right before the next agent turn for grounding prediction (Subtask 1) or generation (Subtask 2).

