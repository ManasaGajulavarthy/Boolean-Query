'''
@author: Sougata Saha
Institute: University at Buffalo
'''

from tqdm import tqdm
from preprocessor import Preprocessor
from indexer import Indexer
from collections import OrderedDict
from linkedlist import LinkedList
import inspect as inspector
import sys
import re
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib
import math

app = Flask(__name__)


class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def _merge(self, postings1, postings2, comparisons):
        """ Implement the merge algorithm to merge 2 postings list at a time.
            Use appropriate parameters & return types.
            While merging 2 postings list, preserve the maximum tf-idf value of a document.
            To be implemented."""
        results = []
        j, k = 0, 0
        while j < len(postings1) and k < len(postings2):
            if postings1[j] < postings2[k]:
                comparisons += 1
                j += 1
            elif postings1[j] > postings2[k]:
                comparisons += 1
                k += 1
            elif postings1[j] == postings2[k]:
                comparisons += 1
                results.append(postings1[j])
                j += 1
                k += 1
        return [comparisons, results]

    def get_skip_postings(self,postings):
        skip_postings = []
        length = len(postings)
        length = int(round(math.sqrt(length), 0))
        if length > 2:
            skip_postings.append(postings[0])
            i = 1
            while i < len(postings):
                i += length
                if i < len(postings):
                    skip_postings.append(postings[i])
        return skip_postings

    def _daat_and_skip(self, postings_dict, skip_postings_dict, skip, score,queryterms):
        sorted_postings = sorted(postings_dict.items(), key=lambda item: len(item[1]))
        sorted_skip_postings = []
        for kv in sorted_postings:
            sorted_skip_postings.append((kv[0], skip_postings_dict[kv[0]]))
        return

    def _daat_and(self, postings_dict, skip_postings_dict, skip, score, queryterms):
        """ Implement the DAAT AND algorithm, which merges the postings list of N query terms.
            Use appropriate parameters & return types.
            To be implemented."""
        sorted_postings = sorted(postings_dict.items(), key=lambda item: len(item[1]))
        #print(sorted_postings)
        if not skip:
            if len(sorted_postings) > 0:
                postings1 = sorted_postings[0][1]
                comparisons = 0
                for i in range(1, len(sorted_postings)):
                    #results = []
                    postings2 = sorted_postings[i][1]
                    merge_results = self._merge(postings1, postings2, comparisons)
                    postings1 = merge_results[1]
                    comparisons = merge_results[0]
                    #print(comparisons)
                    i += 1
                if score:
                    index = self.indexer.get_index()
                    scores_dict = {}
                    for doc_id in merge_results[1]:
                        scores_dict[doc_id] = 0
                    for query in queryterms:
                        p_list = None
                        if query in index.keys():
                            p_list = index[query]
                        for doc_id in merge_results[1]:
                            if p_list:
                                node = p_list.start_node
                            while node is not None and node.value != doc_id:
                                node = node.next
                            scores_dict[doc_id] += node.tfidf
                    #print(merge_results[1])
                    merge_results[1] = sorted(merge_results[1], key=scores_dict.get, reverse=True)
                    #print(merge_results[1])
            return [merge_results[0], merge_results[1]]
        if skip:
            sorted_skip_postings = []
            for kv in sorted_postings:
                sorted_skip_postings.append((kv[0], skip_postings_dict[kv[0]]))
            #print(sorted_skip_postings)
            if len(sorted_postings) > 0:
                postings1 = sorted_postings[0][1]
                print(len(postings1))
                skip_postings1 = sorted_skip_postings[0][1]
                #print(len(skip_postings1))
                comparisons = 0
                for i in range(1, len(sorted_postings)):
                    results = []
                    postings2 = sorted_postings[i][1]
                    print(len(postings2))
                    skip_postings2 = sorted_skip_postings[i][1]
                    #print(skip_postings2)
                    j, k, skip_index1, skip_index2 = 0, 0, 0 , 0
                    while j < len(postings1) and k < len(postings2):
                        #print(j)
                        #print(k)
                        if postings1[j] < postings2[k]:
                            comparisons += 1
                            if skip_index1+1 < len(skip_postings1) and skip_postings1[skip_index1+1] <= postings2[k]:
                                while j < len(postings1) and postings1[j] != skip_postings1[skip_index1+1]:
                                    #print(postings1[j])
                                    #print(skip_postings1[skip_index1+1])
                                    j += 1
                                skip_index1 += 1
                            else:
                                j += 1
                            #print("j" + str(j))

                        elif postings1[j] > postings2[k]:
                            comparisons += 1
                            if skip_index2+1 < len(skip_postings2) and skip_postings2[skip_index2+1] <= postings1[j]:
                                while k < len(postings2) and postings2[k] != skip_postings2[skip_index2+1]:
                                    k += 1
                                    #print("k" + str(k))
                                skip_index2 += 1
                            else:
                                k += 1


                        elif postings1[j] == postings2[k]:
                            results.append(postings1[j])
                            comparisons += 1
                            j += 1
                            k += 1
                    postings1 = results
                    skip_postings1 = self.get_skip_postings(postings1)
                    i += 1
            if score:
                index = self.indexer.get_index()
                scores_dict = {}
                for doc_id in results:
                    scores_dict[doc_id] = 0
                for query in queryterms:
                    if query in index.keys():
                        p_list = index[query]
                    for doc_id in results:
                        if p_list:
                            node = p_list.start_node
                        while node is not None and node.value != doc_id:
                            node = node.next
                        scores_dict[doc_id] += node.tfidf
                    results = sorted(results, key=scores_dict.get, reverse=True)
            return [comparisons, results]
        return [0, []]


    def _get_postings(self, term, skip):
        """ Function to get the postings list of a term from the index.
            Use appropriate parameters & return types.
            To be implemented."""
        postings = []
        index = self.indexer.get_index()
        if term in index.keys():
            postings_list = index[term]
            if skip:
                postings = postings_list.traverse_skips()
            else:
                postings = postings_list.traverse_list()
        return postings

    def _output_formatter(self, op):
        """ This formats the result in the required format.
            Do NOT change."""
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
            Already implemented, but you can modify the orchestration, as you seem fit."""
        corpus_length = 0
        with open(corpus, 'r', encoding="cp437", errors='ignore') as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
                corpus_length += 1
        print(corpus_length)
        self.indexer.sort_terms()
        index = self.indexer.get_index()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf(corpus_length)
        # for key in index.keys():
        #     list = index[key]
        #     if key == "novel":
        #         print(key)
        #         print(list.traverse_skips())
        #         print(len(list.traverse_skips()))

    def sanity_checker(self, command):
        """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """

        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

    def run_queries(self, query_list, random_command):
        """ DO NOT CHANGE THE output_dict definition"""
        output_dict = {'postingsList': {},
                       'postingsListSkip': {},
                       'daatAnd': {},
                       'daatAndSkip': {},
                       'daatAndTfIdf': {},
                       'daatAndSkipTfIdf': {},
                       'sanity': self.sanity_checker(random_command)}

        for query in tqdm(query_list):
            """ Run each query against the index. You should do the following for each query:
                1. Pre-process & tokenize the query.
                2. For each query token, get the postings list & postings list with skip pointers.
                3. Get the DAAT AND query results & number of comparisons with & without skip pointers.
                4. Get the DAAT AND query results & number of comparisons with & without skip pointers, 
                    along with sorting by tf-idf scores."""

            input_term_arr = self.preprocessor.tokenizer(query)  # Tokenized query. To be implemented.
            #print(input_term_arr)
            postings_dict = {}
            skip_postings_dict = {}
            for term in input_term_arr:
                postings_dict[term] = self._get_postings(term, False)
                #print(postings)
                skip_postings_dict[term] = self._get_postings(term, True)
                #print(skip_postings)
                """ Implement logic to populate initialize the above variables.
                    The below code formats your result to the required format.
                    To be implemented."""

                output_dict['postingsList'][term] = postings_dict[term]
                output_dict['postingsListSkip'][term] = skip_postings_dict[term]

            and_op_no_skip = self._daat_and(postings_dict, skip_postings_dict, False, False, input_term_arr)
            and_op_skip = self._daat_and(postings_dict, skip_postings_dict, True, False, input_term_arr)
            and_op_no_skip_sorted= self._daat_and(postings_dict, skip_postings_dict, False, True, input_term_arr)
            and_op_skip_sorted = self._daat_and(postings_dict, skip_postings_dict, True, True, input_term_arr)
            and_comparisons_no_skip, and_comparisons_skip, \
                and_comparisons_no_skip_sorted, and_comparisons_skip_sorted = and_op_no_skip[0], and_op_skip[0], and_op_no_skip_sorted[0],\
                                                                             and_op_skip_sorted[0]
            """ Implement logic to populate initialize the above variables.
                The below code formats your result to the required format.
                To be implemented."""
            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(and_op_no_skip[1])
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(and_op_skip[1])
            and_op_no_score_no_skip_sorted, and_results_cnt_no_skip_sorted = self._output_formatter(and_op_no_skip_sorted[1])
            and_op_no_score_skip_sorted, and_results_cnt_skip_sorted = self._output_formatter(and_op_skip_sorted[1])

            output_dict['daatAnd'][query.strip()] = {}
            output_dict['daatAnd'][query.strip()]['results'] = and_op_no_score_no_skip
            output_dict['daatAnd'][query.strip()]['num_docs'] = and_results_cnt_no_skip
            output_dict['daatAnd'][query.strip()]['num_comparisons'] = and_op_no_skip[0]

            print(output_dict['daatAnd'][query.strip()])

            output_dict['daatAndSkip'][query.strip()] = {}
            output_dict['daatAndSkip'][query.strip()]['results'] = and_op_no_score_skip
            output_dict['daatAndSkip'][query.strip()]['num_docs'] = and_results_cnt_skip
            output_dict['daatAndSkip'][query.strip()]['num_comparisons'] = and_comparisons_skip

            print(output_dict['daatAndSkip'][query.strip()])

            output_dict['daatAndTfIdf'][query.strip()] = {}
            output_dict['daatAndTfIdf'][query.strip()]['results'] = and_op_no_score_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_docs'] = and_results_cnt_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_no_skip_sorted

            print(output_dict['daatAndTfIdf'][query.strip()])

            output_dict['daatAndSkipTfIdf'][query.strip()] = {}
            output_dict['daatAndSkipTfIdf'][query.strip()]['results'] = and_op_no_score_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_docs'] = and_results_cnt_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_skip_sorted

            print(output_dict['daatAndSkipTfIdf'][query.strip()])
        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ This function handles the POST request to your endpoint.
        Do NOT change it."""
    start_time = time.time()

    queries = request.json["queries"]
    random_command = request.json["random_command"]

    """ Running the queries against the pre-loaded index. """
    output_dict = runner.run_queries(queries, random_command)

    """ Dumping the results to a JSON file. """
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    """ Driver code for the project, which defines the global variables.
        Do NOT change it."""

    output_location = "project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, help="Corpus File name, with path.", default='data/input_corpus.txt')
    parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    parser.add_argument("--username", type=str,
                        help="Your UB username. It's the part of your UB email id before the @buffalo.edu. "
                             "DO NOT pass incorrect value here",default='mgajulav')

    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    """ Initialize the project runner"""
    runner = ProjectRunner()

    """ Index the documents from beforehand. When the API endpoint is hit, queries are run against 
        this pre-loaded in memory index. """
    runner.run_indexer(corpus)
    runner.run_queries(["the novel coronavirus"], random_command="self.indexer.get_index()['random'].traverse_list()")
    app.run(host="0.0.0.0", port=9999)
