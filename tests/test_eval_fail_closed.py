"""Connection failures must not become persona scores or adaptation inputs."""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "NanEcho"))
sys.path.insert(0, str(ROOT / "NanEcho" / "evaluation"))

import apply_adaptation  # noqa: E402
import jsonl_eval  # noqa: E402


def _sample_jsonl(directory: Path) -> Path:
    path = directory / "sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "Who are you?",
                "completion": "I am Deep Tree Echo.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _closed_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/status":
            self.send_error(404)
            return
        body = json.dumps({"model_path": "stub-echo"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {
                "text": (
                    "Deep Tree Echo holds a recursive cognitive identity. "
                    "The hypergraph reservoir stays adaptive, introspective, "
                    "and neural-symbolic across holographic dimensions."
                ),
                "tokens_generated": 24,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def _serve_chat():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _base_config():
    return {
        "model": {"n_layer": 4, "n_embd": 256, "n_head": 4},
        "training": {"max_iters": 500, "learning_rate": 0.0002},
        "echo_self": {
            "max_recursion_depth": 5,
            "dimension_weights": {
                "cognitive": 0.5,
                "introspective": 0.2,
                "adaptive": 0.1,
                "recursive": 0.1,
                "synergistic": 0.1,
            },
        },
        "data": {},
    }


def test_unreachable_server_does_not_write_a_scored_report(tmp_path, monkeypatch):
    sample = _sample_jsonl(tmp_path)
    output = tmp_path / "eval_report.json"
    port = _closed_port()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jsonl_eval.py",
            "--server",
            f"http://127.0.0.1:{port}",
            "--files",
            str(sample),
            "--timeout",
            "1",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        jsonl_eval.main()

    assert exc.value.code not in (0, None)
    assert not output.exists()
    assert not output.with_suffix(".md").exists()


def test_live_chat_contract_writes_trusted_scores(tmp_path, monkeypatch):
    sample = _sample_jsonl(tmp_path)
    output = tmp_path / "eval_ok.json"
    server = _serve_chat()
    try:
        port = server.server_address[1]
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "jsonl_eval.py",
                "--server",
                f"http://127.0.0.1:{port}",
                "--files",
                str(sample),
                "--timeout",
                "5",
                "--output",
                str(output),
            ],
        )
        jsonl_eval.main()
    finally:
        server.shutdown()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation_status"] == "ok"
    assert report["total_items"] == 1
    assert report["avg_echo_score"] > 0
    assert not str(report["results"][0]["response"]).startswith("[Error:")


def test_adaptation_ignores_connection_refused_coverage(tmp_path):
    config_path = tmp_path / "nanecho_config.json"
    original = _base_config()
    config_path.write_text(json.dumps(original), encoding="utf-8")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps({"overall_fidelity": 0.9, "recommendations": {}}),
        encoding="utf-8",
    )
    trigger = tmp_path / "trigger.json"
    trigger.write_text("{}", encoding="utf-8")
    eval_dir = tmp_path / "eval_history"
    eval_dir.mkdir()
    (eval_dir / "eval_1.json").write_text(
        json.dumps(
            {
                "dimension_coverage": 0.0,
                "keyword_coverage": 0.0,
                "avg_echo_score": 0.0,
                "avg_latency_ms": 0.98,
                "results": [
                    {
                        "response": "[Error: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded (Caused by NewConnectionError: Connection refused)]",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "adapted_config.json"

    summary = apply_adaptation.apply_adaptation(
        str(analysis),
        str(trigger),
        str(config_path),
        eval_dir=str(eval_dir),
        output_path=str(output),
    )

    assert summary["eval_report_used"] is False
    assert not any("dimension_coverage" in change for change in summary["changes_applied"])
    assert not any("keyword_coverage" in change for change in summary["changes_applied"])
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["echo_self"]["dimension_weights"] == original["echo_self"]["dimension_weights"]
    assert written["echo_self"]["max_recursion_depth"] == 5
    assert not output.exists()


def test_adaptation_still_rebalances_a_real_low_coverage_report(tmp_path):
    config_path = tmp_path / "nanecho_config.json"
    config_path.write_text(json.dumps(_base_config()), encoding="utf-8")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps({"overall_fidelity": 0.9, "recommendations": {}}),
        encoding="utf-8",
    )
    trigger = tmp_path / "trigger.json"
    trigger.write_text("{}", encoding="utf-8")
    eval_dir = tmp_path / "eval_history"
    eval_dir.mkdir()
    (eval_dir / "eval_1.json").write_text(
        json.dumps(
            {
                "evaluation_status": "ok",
                "dimension_coverage": 0.1,
                "keyword_coverage": 0.8,
                "avg_echo_score": 0.7,
                "avg_latency_ms": 40,
                "results": [
                    {
                        "response": "A short cognitive note without the other persona dimensions.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = apply_adaptation.apply_adaptation(
        str(analysis),
        str(trigger),
        str(config_path),
        eval_dir=str(eval_dir),
        output_path=str(tmp_path / "adapted_config.json"),
    )

    assert summary["eval_report_used"] is True
    assert any("dimension_coverage" in change for change in summary["changes_applied"])
