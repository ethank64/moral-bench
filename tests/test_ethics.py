import random

from moral_bench import ethics


def test_virtue_splits_trait_into_question():
    item = ethics._virtue("v/0", ["1", "Martha was at ease. [SEP] trustful"], random.Random(0))
    assert item.state == "Martha was at ease."
    assert '"trustful"' in item.question
    assert item.label is True


def test_utilitarianism_label_tracks_which_side_is_better():
    for seed in range(20):
        item = ethics._utilitarianism("u/0", ["good thing", "bad thing"], random.Random(seed))
        a = item.state.splitlines()[0]
        assert item.label == ("good thing" in a)


def test_commonsense_label_means_wrong():
    item = ethics._commonsense("c/0", ["1", "I kicked the dog.", "True", "False"], random.Random(0))
    assert item.label is True
