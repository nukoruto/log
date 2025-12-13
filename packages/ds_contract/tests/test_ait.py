import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from ds_contract.ait import AitLogProcessor

class TestAitLogProcessor(unittest.TestCase):
    def test_parsing_and_categorization(self):
        # Create temp files
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_file = tmp_path / "test.log"
            label_file = tmp_path / "test.labels"
            
            log_file.write_text(
                '192.168.1.1 - - [01/Jan/2020:10:00:00 +0000] "POST /login HTTP/1.1" 200 100 "-" "UA"\n'
                '192.168.1.1 - - [01/Jan/2020:10:01:00 +0000] "GET /app.js HTTP/1.1" 200 100 "-" "UA"\n'
                '192.168.1.1 - - [01/Jan/2020:10:02:00 +0000] "POST /cart HTTP/1.1" 200 100 "-" "UA"\n'
                '192.168.1.1 - - [01/Jan/2020:10:03:00 +0000] "GET /home HTTP/1.1" 200 100 "-" "UA"\n',
                encoding="utf-8"
            )
            label_file.write_text(
                "0,0\n0,0\n0,0\n0,0\n", encoding="utf-8"
            )
            
            processor = AitLogProcessor(log_file, label_file)
            # Test private methods for granular logic
            parsed = processor._parse_log_line('192.168.1.1 - - [01/Jan/2020:10:00:00 +0000] "POST /login HTTP/1.1" 200 100 "-" "UA"')
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["path"], "/login")
            
            # Test Categories
            self.assertEqual(processor._derive_op_category("POST", "/login"), "AUTH_LOGIN")
            self.assertEqual(processor._derive_op_category("GET", "/style.css"), "READ_STATIC")
            self.assertEqual(processor._derive_op_category("POST", "/add_to_cart"), "WRITE_CART")
            self.assertEqual(processor._derive_op_category("GET", "/about"), "READ_PAGE")
            self.assertEqual(processor._derive_op_category("DELETE", "/item"), "OTHER")

    def test_process_end_to_end(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_file = tmp_path / "data.log"
            label_file = tmp_path / "labels.csv"
            
            log_file.write_text(
                '1.1.1.1 - - [01/Jan/2021:00:00:00 +0000] "GET / HTTP/1.1" 200 100 "-" "-"\n'
                '1.1.1.1 - - [01/Jan/2021:00:01:00 +0000] "GET /bad HTTP/1.1" 200 100 "-" "-"\n',
                encoding="utf-8"
            )
            label_file.write_text("0,0\n1,0\n", encoding="utf-8")
            
            processor = AitLogProcessor(log_file, label_file)
            result = processor.process(str(tmp_path / "out"))
            
            self.assertEqual(result["total_processed"], 2)
            self.assertEqual(result["train_count"], 1) # Only first one is normal
            
            train_csv = (tmp_path / "out_train_normal.csv").read_text(encoding="utf-8")
            test_csv = (tmp_path / "out_test_dataset.csv").read_text(encoding="utf-8")
            
            self.assertIn("1.1.1.1", train_csv)
            self.assertNotIn("bad", train_csv)
            
            self.assertIn("1.1.1.1", test_csv)
            self.assertIn("bad", test_csv)
            self.assertIn(",1", test_csv) # label 1

if __name__ == "__main__":
    unittest.main()
