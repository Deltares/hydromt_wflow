from hydromt import DataCatalog


def test_fetch_data(build_data_catalog, build_data):
    assert build_data_catalog.is_file()
    assert build_data.is_dir()
    dc = DataCatalog(build_data_catalog)
    assert dc.get_source("chelsa").name == "chelsa"
