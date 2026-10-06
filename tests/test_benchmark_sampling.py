from posthoc.utils.downstream_utils import sample_benchmark


def test_benchmark_cap_is_shared_across_subjects_and_reproducible():
    sizes = {"first": 2000, "second": 3000, "third": 5000}
    samples = sample_benchmark(sizes, 5000)
    assert sum(map(len, samples.values())) == 5000
    assert [len(samples[name]) for name in sizes] == [1000, 1500, 2500]
    assert samples == sample_benchmark(dict(reversed(list(sizes.items()))), 5000)
    for name, selected in samples.items():
        assert len(set(selected)) == len(selected)
        assert min(selected) >= 0 and max(selected) < sizes[name]


def test_small_benchmark_uses_all_examples():
    assert sample_benchmark({"first": 2, "second": 3}, 5000) == {
        "first": [0, 1],
        "second": [0, 1, 2],
    }
