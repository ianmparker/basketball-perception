from research.ontology.validate import load, validate


def test_play_taxonomy_validates():
    data = load()
    assert validate(data) == []


def test_taxonomy_has_the_three_orthogonal_levels():
    data = load()
    assert len(data["play_types"]) >= 10
    assert len(data["coverages"]) >= 8
    assert len(data["play_families"]) >= 5


def test_ids_are_unique_and_kebab_case():
    data = load()
    ids = [item["id"] for group in ("play_types", "coverages", "play_families") for item in data[group]]
    assert len(ids) == len(set(ids))
    assert all(i == i.lower() and " " not in i for i in ids)


def test_required_fields_present():
    data = load()
    for group in ("play_types", "coverages", "play_families"):
        for item in data[group]:
            for field in ("id", "name", "definition", "signals"):
                assert item.get(field), f"{group}.{item.get('id')}: missing {field}"
            assert isinstance(item["signals"], list) and item["signals"]
