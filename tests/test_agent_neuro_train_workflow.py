from pathlib import Path
import unittest


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github/workflows/agent-neuro-train.yml"


class AgentNeuroTrainWorkflowTests(unittest.TestCase):
    def test_persona_controls_use_supported_precision(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn('echo "persona_reinforcement=0.9" >> $GITHUB_OUTPUT', workflow)
        self.assertIn("--persona_weight=0.9 \\", workflow)

    def test_fallback_dataset_script_uses_heredoc_python(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("python - <<'PY'", workflow)
        self.assertIn('with open("data/nanecho/metadata.json", "w") as f:', workflow)


if __name__ == "__main__":
    unittest.main()
