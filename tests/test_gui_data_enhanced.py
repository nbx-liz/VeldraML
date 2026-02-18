from __future__ import annotations

from veldra.gui import services


def test_inspect_data_includes_quality_profiles(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,target,dt\n1,x,0,2024-01-01\n2,y,1,2024-01-02\n", encoding="utf-8")

    out = services.inspect_data(str(csv_path))
    assert out["success"] is True
    stats = out["stats"]
    assert "column_profiles" in stats
    assert "warnings" in stats
    assert any(p["name"] == "target" for p in stats["column_profiles"])
