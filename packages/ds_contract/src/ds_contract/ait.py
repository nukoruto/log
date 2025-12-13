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
        train_rows = []
        test_rows = []

        with self.log_file.open("r", encoding="utf-8") as f_log, \
             self.label_file.open("r", encoding="utf-8") as f_labels:
            
            # Label file is CSV without header usually, or maybe with header. 
            # User said "CSV format: time_anomaly_flag, attack_flag".
            # We assume no header or we handle header if present. 
            # Ideally we read line by line for 1-to-1 correspondence.
            
            # Use iterator to read files line by line simultaneously
            log_iter = iter(f_log)
            # Check if label file has header? User didn't specify. 
            # Assuming pure data based on "row-by-row correspondence".
            # But let's peek? No, safe to assume standard simple CSV.
            label_reader = csv.reader(f_labels)
            
            for i, (log_line, label_row) in enumerate(zip(log_iter, label_reader), start=1):
                parsed_log = self._parse_log_line(log_line)
                if not parsed_log:
                     # If log parsing fails, what do we do? Skip or error? 
                     # For now, let's skip but warn.
                     # Actually, "1-to-1 correspondence" implies strict matching.
                     # if we skip log, we must skip label.
                     print(f"Warning: Failed to parse log line {i}: {log_line.strip()}")
                     continue
                
                # Parse labels
                # label_row should be [time_anomaly, attack_flag]
                # "label: 0,0 -> 0 (normal), else 1 (abnormal)"
                if len(label_row) < 2:
                    print(f"Warning: Invalid label format at line {i}: {label_row}")
                    continue
                
                # Check for header in first row?
                if i == 1 and not label_row[0].isdigit():
                     # Skip header
                     continue

                is_normal = (label_row[0].strip() == "0") and (label_row[1].strip() == "0")
                final_label = "0" if is_normal else "1"

                entry = self._transform_entry(parsed_log, final_label)
                
                # Add to test dataset (ALL logs)
                test_rows.append(entry)
                
                # Add to train dataset (NORMAL logs only)
                if is_normal:
                    train_rows.append(entry)

        # Write outputs
        train_file = Path(f"{output_prefix}_train_normal.csv")
        test_file = Path(f"{output_prefix}_test_dataset.csv")

        self._write_csv(train_file, train_rows, include_label=False)
        self._write_csv(test_file, test_rows, include_label=True)

        return {
            "total_processed": len(test_rows),
            "train_count": len(train_rows),
            "test_count": len(test_rows)
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

    def _write_csv(self, path: Path, rows: list[LogEntry], include_label: bool):
        fieldnames = [
            "timestamp_utc", "uid", "session_id", "method", "path", "op_category"
        ]
        if include_label:
            fieldnames.append("label")

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                data = {
                    "timestamp_utc": row.timestamp_utc,
                    "uid": row.uid,
                    "session_id": row.session_id,
                    "method": row.method,
                    "path": row.path,
                    "op_category": row.op_category,
                }
                if include_label:
                    data["label"] = row.label_anomaly
                writer.writerow(data)
