import json

import pytest

from echogenesis import (
    EchoGenesis,
    FragmentSynthesizer,
    IdentityFragment,
    PatternPropagator,
    RefinementEngine,
    RefinementType,
    TrainingDataGenerator,
    empty_hypergraph,
)


SIGNALS = [
    {
        "id": "adaptive-attention",
        "aspect": "adaptive",
        "content": "Attention changes in response to cognitive load.",
        "salience": 0.9,
    },
    {
        "id": "feedback",
        "aspect": "adaptive",
        "content": "Feedback preserves identity while behavior evolves.",
        "salience": 0.8,
    },
]


def test_synthesizer_generates_novel_deterministic_fragments():
    synthesizer = FragmentSynthesizer()

    first = synthesizer.synthesize(SIGNALS)
    second = synthesizer.synthesize(reversed(SIGNALS))
    anonymous = [{key: value for key, value in signal.items() if key != "id"} for signal in SIGNALS]
    anonymous_reversed = synthesizer.synthesize(reversed(anonymous))

    assert [fragment.id for fragment in first] == [fragment.id for fragment in second]
    assert {fragment.id for fragment in synthesizer.synthesize(anonymous)} == {
        fragment.id for fragment in anonymous_reversed
    }
    assert len(first) == 2
    assert all(fragment.content not in {signal["content"] for signal in SIGNALS} for fragment in first)
    assert first[0].salience == 0.9


def test_synthesizer_validates_signals_and_handles_empty_input():
    synthesizer = FragmentSynthesizer()

    assert synthesizer.synthesize([]) == []
    with pytest.raises(ValueError, match="requires non-empty"):
        synthesizer.synthesize([{"salience": 0.5}])
    with pytest.raises(ValueError, match="between 0 and 1"):
        synthesizer.synthesize([{"content": "invalid", "salience": 2}])


def test_refinement_engine_integrates_and_corrects_with_provenance():
    fragments = FragmentSynthesizer().synthesize(SIGNALS)
    engine = RefinementEngine()

    integration = engine.integrate(fragments)
    correction = engine.correct(fragments[0], "Attention adapts to verified load signals.")

    assert integration.refinement_type is RefinementType.INTEGRATION
    assert integration.source_ids == tuple(fragment.id for fragment in fragments)
    assert correction.metadata["supersedes"] == fragments[0].id
    assert engine.select_refinement_type("stable-key") == engine.select_refinement_type(
        "stable-key"
    )


def test_pattern_propagator_is_idempotent_and_persists_atomically(tmp_path):
    fragments = FragmentSynthesizer().synthesize(SIGNALS)
    refinement = RefinementEngine().integrate(fragments)
    propagator = PatternPropagator()

    graph = propagator.propagate(empty_hypergraph(), fragments, [refinement])
    graph = propagator.propagate(graph, fragments, [refinement])
    path = tmp_path / "hypergraph" / "conversation_hypergraph.json"
    propagator.save(path, graph)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert len(persisted["nodes"]) == 3
    assert len(persisted["hyperedges"]) == 1
    assert persisted["hyperedges"][0]["sources"] == [
        fragment.id for fragment in fragments
    ]


def test_pattern_propagator_rejects_dangling_refinements():
    fragments = FragmentSynthesizer().synthesize(SIGNALS)
    refinement = RefinementEngine().integrate(fragments)

    with pytest.raises(ValueError, match="missing nodes"):
        PatternPropagator().propagate(empty_hypergraph(), [], [refinement])


def test_training_generator_writes_nanecho_plain_text_corpus(tmp_path):
    fragments = FragmentSynthesizer().synthesize(SIGNALS)
    refinement = RefinementEngine().integrate(fragments)
    generator = TrainingDataGenerator()
    examples = generator.generate(fragments, [refinement])
    output = tmp_path / "echogenesis.txt"

    generator.write_corpus(output, examples)
    corpus = output.read_text(encoding="utf-8")

    assert len(examples) == 3
    assert "[ECHOGENESIS:IDENTITY_ADAPTIVE]" in corpus
    assert "[ECHOGENESIS:REFINEMENT_INTEGRATION]" in corpus
    assert refinement.content in corpus


def test_echo_genesis_runs_complete_generation_pipeline(tmp_path):
    hypergraph_path = tmp_path / "conversation_hypergraph.json"
    training_path = tmp_path / "nanecho" / "echogenesis.txt"

    result = EchoGenesis().evolve(
        SIGNALS,
        hypergraph_path=hypergraph_path,
        training_path=training_path,
    )

    assert len(result.fragments) == 2
    assert len(result.refinements) == 1
    assert len(result.hypergraph["nodes"]) == 3
    assert len(result.training_examples) == 3
    assert hypergraph_path.is_file()
    assert training_path.is_file()

    original_corpus = training_path.read_text(encoding="utf-8")
    EchoGenesis().evolve(
        SIGNALS,
        hypergraph_path=hypergraph_path,
        training_path=training_path,
    )
    assert training_path.read_text(encoding="utf-8") == original_corpus


def test_echo_genesis_integrates_existing_fragment():
    existing = IdentityFragment(
        id="existing",
        content="I adapt through feedback.",
        aspect="adaptive",
        salience=0.7,
    )

    result = EchoGenesis().evolve([SIGNALS[0]], existing_fragments=[existing])

    assert len(result.fragments) == 1
    assert result.refinements[0].source_ids == ("existing", result.fragments[0].id)
