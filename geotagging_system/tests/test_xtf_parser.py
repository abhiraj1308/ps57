"""
Regression tests for the XTF parser.

pyxtf may not be installed in every environment, so these tests fake out
the `pyxtf` module the parser depends on rather than requiring a real
XTF file. The goal is specifically to catch the class of bug where
PingMetadata gets built with wrong keyword names (e.g. `heading=` instead
of `heading_deg=`) and pydantic silently drops the value instead of
raising, leaving altitude/depth/heading/cable_out at their 0.0 defaults.
"""
import sys
import types
from datetime import datetime, timezone

import pytest

from src.parsers.xtf_parser import XTFParser
from src.models import QualityFlag


class _FakeChanHeader:
    def __init__(self, slant_range, num_samples):
        self.SlantRange = slant_range
        self.NumSamples = num_samples


class _FakeSonarPacket:
    def __init__(self, i):
        # Non-zero, distinguishable values so a field-mapping bug is obvious.
        self.SensorXcoordinate = -82.5 - i * 0.0001
        self.SensorYcoordinate = 27.8 + i * 0.0001
        self.ShipXcoordinate = -82.5001 - i * 0.0001
        self.ShipYcoordinate = 27.8001 + i * 0.0001
        self.Year, self.Month, self.Day = 2024, 1, 15
        self.Hour, self.Minute, self.Second = 10, 0, i
        self.HSeconds = 0
        self.SensorHeading = 47.5
        self.SensorPrimaryAltitude = 14.2
        self.SensorDepth = 2.3
        self.Layback = 55.0
        self.ping_chan_headers = [_FakeChanHeader(slant_range=148.0, num_samples=2048)]


@pytest.fixture
def fake_pyxtf(monkeypatch):
    """Install a minimal fake `pyxtf` module into xtf_parser's namespace."""
    fake_module = types.SimpleNamespace()
    fake_module.XTFHeaderType = types.SimpleNamespace(sonar="sonar")

    packets = {"sonar": [_FakeSonarPacket(i) for i in range(5)]}

    def fake_xtf_read(filepath, read_packets=True):
        if not read_packets:
            return types.SimpleNamespace(SystemType=1, NavUnits=3, NumberOfChannels=2)
        return types.SimpleNamespace(), packets

    fake_module.xtf_read = fake_xtf_read

    import src.parsers.xtf_parser as xtf_parser_module
    monkeypatch.setattr(xtf_parser_module, "pyxtf", fake_module)
    return fake_module


class TestXTFParser:
    def test_parse_maps_navigation_fields_correctly(self, fake_pyxtf, tmp_path):
        """The core regression check: heading/altitude/depth/cable_out
        must survive parsing, not silently reset to 0.0."""
        parser = XTFParser()
        pings = parser.parse("fake.xtf")

        assert len(pings) == 5
        p = pings[0]

        assert p.heading_deg == pytest.approx(47.5)
        assert p.altitude_m == pytest.approx(14.2)
        assert p.depth_m == pytest.approx(2.3)
        assert p.cable_out_m == pytest.approx(55.0)
        assert p.slant_range_m == pytest.approx(148.0)
        assert p.num_samples == 2048

    def test_parse_uses_sensor_coordinates(self, fake_pyxtf):
        parser = XTFParser()
        pings = parser.parse("fake.xtf")
        p = pings[0]
        assert p.latitude == pytest.approx(27.8)
        assert p.longitude == pytest.approx(-82.5)

    def test_parse_falls_back_to_ship_coordinates_when_sensor_is_zero(self, monkeypatch, fake_pyxtf):
        fake_pyxtf.xtf_read.__wrapped__ = None  # no-op, keeps lint happy

        def xtf_read_with_zeroed_sensor(filepath, read_packets=True):
            pkt = _FakeSonarPacket(0)
            pkt.SensorXcoordinate = 0.0
            pkt.SensorYcoordinate = 0.0
            return types.SimpleNamespace(), {"sonar": [pkt]}

        import src.parsers.xtf_parser as xtf_parser_module
        monkeypatch.setattr(fake_pyxtf, "xtf_read", xtf_read_with_zeroed_sensor)
        monkeypatch.setattr(xtf_parser_module, "pyxtf", fake_pyxtf)

        parser = XTFParser()
        pings = parser.parse("fake.xtf")
        p = pings[0]

        assert p.latitude == pytest.approx(27.8001)
        assert p.longitude == pytest.approx(-82.5001)
        assert QualityFlag.GPS_INTERPOLATED in p.quality_flags

    def test_parse_raises_when_pyxtf_not_installed(self, monkeypatch):
        import src.parsers.xtf_parser as xtf_parser_module
        monkeypatch.setattr(xtf_parser_module, "pyxtf", None)
        from src.exceptions import MetadataParseError

        parser = XTFParser()
        with pytest.raises(MetadataParseError):
            parser.parse("fake.xtf")
