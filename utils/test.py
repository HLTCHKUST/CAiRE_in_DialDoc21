            """Load dialog data in the reading comprehension task setup, where context is the grounding document,
            input query is dialog history in reversed order, and output to predict is the next agent turn."""

            logging.info("generating examples from = %s", filepath)
            doc_data = self._load_doc_data_rc(filepath)
            with open(filepath, encoding="utf-8") as f:
                dial_data = json.load(f)["dial_data"]
                for domain, d_doc_dials in dial_data.items():
                    for doc_id, dials in d_doc_dials.items():
                        doc = doc_data[domain][doc_id]
                        for dial in dials:
                            all_prev_utterances = []
                            for idx, turn in enumerate(dial["turns"]):
                                all_prev_utterances.append(
                                    "\t{}: {}".format(turn["role"], turn["utterance"])
                                )
                                if "answers" not in turn:
                                    turn["answers"] = self._get_answers_rc(
                                        turn["references"],
                                        doc["spans"],
                                        doc["doc_text"],
                                    )
                                if turn["role"] == "agent":
                                    continue
                                if idx + 1 < len(dial["turns"]):
                                    if dial["turns"][idx + 1]["role"] == "agent":
                                        turn_to_predict = dial["turns"][idx + 1]
                                    else:
                                        continue
                                else:
                                    continue
                                question_str = " ".join(
                                    list(reversed(all_prev_utterances))
                                ).strip()
                                question = " ".join(question_str.split()[:MAX_Q_LEN])
                                id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"]) # For subtask1, the id should be this format.
                                qa = {
                                    "id": id_, # For subtask1, the id should be this format.
                                    "title": doc_id,
                                    "context": doc["doc_text"],
                                    "question": question,
                                    "answers": [],  # For subtask1, "answers" contains the grounding annotations for evaluation.
                                    "domain": domain,
                                }
                                if "answers" not in turn_to_predict:
                                    turn_to_predict["answers"] = self._get_answers_rc(
                                        turn_to_predict["references"],
                                        doc["spans"],
                                        doc["doc_text"],
                                    )
                                if turn_to_predict["answers"]:
                                    qa["answers"] = turn_to_predict["answers"]
                                yield id_, qa
