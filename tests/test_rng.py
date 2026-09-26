import pytest

from neurogarden.engine.rng import SplitMix64


def test_reference_vectors_seed_0():
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]


def test_randbelow_reference_seed_42():
    rng = SplitMix64(42)
    assert [rng.randbelow(1000) for _ in range(5)] == [413, 291, 858, 764, 250]


def test_randbelow_stays_in_range():
    rng = SplitMix64(7)
    assert all(0 <= rng.randbelow(3) < 3 for _ in range(1000))


def test_randbelow_rejects_non_positive():
    with pytest.raises(ValueError):
        SplitMix64(1).randbelow(0)


def test_state_is_one_integer_and_restorable():
    rng = SplitMix64(123)
    rng.next_u64()
    clone = SplitMix64(0)
    clone.state = rng.state
    assert clone.next_u64() == rng.next_u64()


def test_seed_is_reduced_mod_2_64():
    assert SplitMix64(2**64 + 5).state == 5


def test_shuffle_is_deterministic_and_a_permutation():
    a, b = list(range(10)), list(range(10))
    SplitMix64(9).shuffle(a)
    SplitMix64(9).shuffle(b)
    assert a == b
    assert sorted(a) == list(range(10))
    assert a != list(range(10))


def test_shuffle_of_single_item_draws_nothing():
    rng = SplitMix64(5)
    before = rng.state
    rng.shuffle([1])
    assert rng.state == before
