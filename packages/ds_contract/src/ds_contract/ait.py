"""AIT Log Preprocessing Module.

Handles parsing of Apache Combined Log Format and Label files,
and generation of training and testing datasets for LSTM models.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

# Apache Combined Log Format Regex
# Ref: https://httpd.apache.org/docs/2.4/logs.html#combined
# "%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-agent}i\""
LOG_PATTERN = re.compile(
    r'(?P<ip>[\d\.]+) - - \[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<path>.*?) HTTP/.*?" (?P<status>\d+) (?P<size>\d+|-) ".*?" "(?P<user_agent>.*?)"'
)


class LogEntry(NamedTuple):
    timestamp_utc: str
    uid: str
    session_id: str
    method: str
    path: str
    op_category: str
    label_anomaly: str
    label_attack: str


class AitLogProcessor:
    def __init__(self, log_file: Path, label_file: Path):
        self.log_file = log_file
        self.label_file = label_file

    def process(self, output_prefix: str) -> dict[str, int]:
        """Process logs and labels to generate train and test datasets.
        
        Args:
            output_prefix: Prefix for output CSV files.
            
        Returns:
            Dictionary with counts of generated rows.
        """
        train_file = Path(f"{output_prefix}_train_normal.csv")
        test_file = Path(f"{output_prefix}_test_dataset.csv")
        
        # Ensure output directory exists
        train_file.parent.mkdir(parents=True, exist_ok=True)

        cols = ["timestamp_utc", "uid", "session_id", "method", "path", "op_category"]
        cols_test = cols + ["label"]

        count_total = 0
        count_train = 0

        with self.log_file.open("r", encoding="utf-8") as f_log, \
             self.label_file.open("r", encoding="utf-8") as f_labels, \
             train_file.open("w", newline="", encoding="utf-8") as f_out_train, \
             test_file.open("w", newline="", encoding="utf-8") as f_out_test:
            
            writer_train = csv.DictWriter(f_out_train, fieldnames=cols)
            writer_test = csv.DictWriter(f_out_test, fieldnames=cols_test)
            
            writer_train.writeheader()
            writer_test.writeheader()
            
            log_iter = iter(f_log)
            label_reader = csv.reader(f_labels)
            
            for i, (log_line, label_row) in enumerate(zip(log_iter, label_reader), start=1):
                # 1. Parse Log (Left)
                parsed_log = self._parse_log_line(log_line)
                if not parsed_log:
                     print(f"Warning: Failed to parse log line {i}: {log_line.strip()}")
                     continue
                
                # 2. Parse Label (Right)
                if not label_row:
                    continue
                # Skip header if present and not numeric
                if i == 1 and not label_row[0].isdigit():
                     continue

                # 3. Glue and Transform
                # Check label: 0,0 means Normal. Anything else is Anomaly.
                # Adjust index logic if label_row length varies? 
                # User mentioned "Normal, Abnormal label only" -> assuming standard logic holds.
                # If label_row has 1 col, use it. If 2, use both.
                if len(label_row) >= 2:
                    is_normal = (label_row[0].strip() == "0") and (label_row[1].strip() == "0")
                elif len(label_row) == 1:
                    is_normal = (label_row[0].strip() == "0")
                else:
                    is_normal = False # Fallback

                final_label = "0" if is_normal else "1"
                
                entry = self._transform_entry(parsed_log, final_label)
                
                row_dict = {
                    "timestamp_utc": entry.timestamp_utc,
                    "uid": entry.uid,
                    "session_id": entry.session_id,
                    "method": entry.method,
                    "path": entry.path,
                    "op_category": entry.op_category,
                }
                
                # 4. Write to Test CSV (Combined) - "Based on that CSV"
                row_test = row_dict.copy()
                row_test["label"] = entry.label_anomaly
                writer_test.writerow(row_test)
                count_total += 1
                
                # 5. Filter for Train CSV
                if is_normal:
                    writer_train.writerow(row_dict)
                    count_train += 1

        return {
            "total_processed": count_total,
            "train_count": count_train,
            "test_count": count_total
        }

    def _parse_log_line(self, line: str) -> dict[str, str] | None:
        match = LOG_PATTERN.search(line)
        if not match:
            return None
        return match.groupdict()

    def _transform_entry(self, raw: dict[str, str], label: str) -> LogEntry:
        # timestamp conversion
        # "[29/Feb/2020:00:00:02 +0000]" -> 2020-02-29T00:00:02+00:00
        ts_str = raw["timestamp"]
        # Python's strptime %z requires +HHMM, Apache log usually has +0000. 
        dt = datetime.strptime(ts_str, "%d/%b/%Y:%H:%M:%S %z")
        # Ensure UTC and ISO format
        timestamp_utc = dt.astimezone(timezone.utc).isoformat()

        uid = raw["ip"] # Mapping Rule: uid <= Client IP
        session_id = uid # Mapping Rule: session_id <= uid (simplified)
        
        method = raw["method"]
        path = raw["path"]
        
        op_category = self._derive_op_category(method, path)
        
        return LogEntry(
            timestamp_utc=timestamp_utc,
            uid=uid,
            session_id=session_id,
            method=method,
            path=path,
            op_category=op_category,
            label_anomaly=label,
            label_attack="",
        )

    def _derive_op_category(self, method: str, path: str) -> str:
        # AUTH_LOGIN: Method=POST かつ Pathに login を含む
        if method == "POST" and "login" in path:
            return "AUTH_LOGIN"
        
        # WRITE_CART: Method=POST かつ Pathに cart を含む
        if method == "POST" and "cart" in path:
            return "WRITE_CART"
        
        # READ_STATIC: Method=GET かつ 拡張子が画像/CSS/JS等 (.png, .css, .js...)
        # Simple extension check
        static_exts = (".png", ".jpg", ".jpeg", ".gif", ".css", ".js", ".ico", ".svg", ".woff", ".woff2", ".ttf")
        if method == "GET":
             # Remove query string for extension check
             clean_path = path.split("?")[0]
             if clean_path.lower().endswith(static_exts):
                 return "READ_STATIC"
             # READ_PAGE: 上記以外の GET リクエスト (implied)
             return "READ_PAGE"

        # OTHER: それ以外
        return "OTHER"


