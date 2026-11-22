"""Offline checks for fallback, grid integrity and failure reporting."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import fetch_site as fetch


class FakeFetch:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def get_json(self, url):
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def note(self, message):
        self.messages.append(message)


class FetchTests(unittest.TestCase):
    def test_census_failure_falls_back(self):
        client = FakeFetch([OSError('DNS failure'), [{'lat': '39.3', 'lon': '-77.3', 'display_name': 'matched'}]])
        lat, lon, matched, source = fetch.geocode('input', None, None, client)
        self.assertEqual((lat, lon, matched, source['name']), (39.3, -77.3, 'matched', 'nominatim'))

    def test_explicit_coordinates_do_not_call_network(self):
        self.assertEqual(fetch.geocode('input', 39, -77, FakeFetch([]))[3]['name'], 'coordinates')

    def test_both_geocoders_fail_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, '--lat LAT --lon LON'):
            fetch.geocode('input', None, None, FakeFetch([{'result': {'addressMatches': []}}, []]))

    def test_tiff_values_preserve_row_and_column_order(self):
        values = [float(row * 10 + col) for row in range(120) for col in range(120)]
        image = Image.new('F', (120, 120))
        image.putdata(values)
        buffer = io.BytesIO()
        image.save(buffer, format='TIFF')
        self.assertEqual(fetch.decode_dem(buffer.getvalue()), values)
        image.putpixel((1, 0), float('nan'))
        buffer = io.BytesIO()
        image.save(buffer, format='TIFF')
        with self.assertRaisesRegex(ValueError, 'NoData'):
            fetch.decode_dem(buffer.getvalue())

    def test_rendered_tiff_is_rejected(self):
        buffer = io.BytesIO()
        Image.new('L', (120, 120)).save(buffer, format='TIFF')
        with self.assertRaisesRegex(ValueError, 'Expected F32'):
            fetch.decode_dem(buffer.getvalue())

    def test_terrain_spacing_is_ground_feet_at_latitude(self):
        import math
        center = fetch.mercator(39.40, -77.40)
        west, south, east, north = fetch.terrain_bbox(39.40, center)
        ground_width = (east - west) * math.cos(math.radians(39.40)) / .3048
        self.assertAlmostEqual(ground_width / 120, 5.2868, places=7)
        self.assertAlmostEqual((north + south) / 2, center[1])

    def test_nonzero_cli_when_lidar_step_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            args = ['A US address', '--lat', '39', '--lon', '-77', '--out-root', folder]
            with patch.object(fetch, 'fetch_dem'), patch.object(fetch, 'run_lidar', side_effect=ValueError('bad dataset')):
                self.assertEqual(fetch.main(args), 1)
            site = json.loads((Path(folder) / 'a-us-address' / 'site.json').read_text())
            self.assertEqual(site['status'], 'failed')
            self.assertIn('bad dataset', site['errors'][0])
            self.assertIn('CLI exit code: 1', (Path(folder) / 'a-us-address' / 'fetch.log').read_text())
            with self.assertRaises(SystemExit) as error:
                fetch.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_downloader_zero_exit_with_error_manifest_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'lidar').mkdir()
            (output / 'lidar' / 'manifest.json').write_text(json.dumps({'datasets': {'main': {'error': '403'}}}))
            with patch.object(fetch.subprocess, 'Popen') as popen:
                process = popen.return_value.__enter__.return_value
                process.stdout = []
                process.wait.return_value = 0
                site = {'address': 'x', 'lat': 39, 'lon': -77, 'half_width_mercator': 125}
                with self.assertRaisesRegex(ValueError, 'failed or empty'):
                    fetch.run_lidar(site, output, FakeFetch([]))

    def test_invalid_inputs_exit_before_creating_output(self):
        for args in [['x', '--lat', '39'], ['x', '--half', 'nan'], ['x', '--lat', 'nan', '--lon', '0'], ['!!!']]:
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                fetch.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
