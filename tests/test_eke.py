from datetime import date

from leddy.eke import dated_files


def test_observation_date_not_version_date(tmp_path):
    folder = tmp_path / "1993" / "01"
    folder.mkdir(parents=True)
    first = folder / "dt_global_allsat_phy_l4_19930101_20241015.nc"
    second = folder / "dt_global_allsat_phy_l4_19930102_20241015.nc"
    first.touch()
    second.touch()
    result = dated_files(tmp_path, date(1993, 1, 1), date(1993, 12, 31))
    assert [item[0] for item in result] == [date(1993, 1, 1), date(1993, 1, 2)]

