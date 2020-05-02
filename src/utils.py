import numpy as np
from nltk.translate.meteor_score import meteor_score


def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def evaluate(rouge_metric, all_hypothesis, all_references):
    meteor_score_list = []
    for hypothesis, references in zip(all_hypothesis, all_references):
        score = meteor_score([references], hypothesis)
        meteor_score_list.append(score)
    meteor_score_int = np.mean(np.array(meteor_score_list))
    print(f"Meteor score: {meteor_score_int}")

    scores = rouge_metric.get_scores(all_hypothesis, all_references)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(metric, results['p'], results['r'], results['f']))

    return {"meteor": meteor_score_int, "rouge": scores}
