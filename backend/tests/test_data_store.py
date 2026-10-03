import io

import pandas as pd

import data_store


def test_load_returns_valid_history():
    df = data_store.load()
    assert not df.empty
    assert list(df.columns) == data_store.REQUIRED_COLUMNS
    assert df["month"].between(1, 12).all()
    assert (df["price_usd_kg"] > 0).all()


def test_validate_skips_malformed_rows():
    csv = io.StringIO(
        "year,month,region,size_class,price_usd_kg\n"
        "2020,1,Global,Medium,10\n"
        "bad,1,Global,Medium,10\n"
        "2020,13,Global,Medium,10\n"
        "2020,2,Global,Medium,-5\n"
        "2020,3,Global,Medium,11\n"
    )
    df = data_store._validate(pd.read_csv(csv))
    assert len(df) == 2


def test_validate_rejects_missing_columns():
    df = pd.DataFrame({"year": [2020], "month": [1]})
    try:
        data_store._validate(df)
        assert False, "should raise"
    except ValueError as e:
        assert "Missing required columns" in str(e)


def test_dimensions():
    d = data_store.dimensions()
    assert "Global" in d["regions"]
    assert "Medium" in d["size_classes"]
