"""A program-level device, and what it reaches."""

import pytest
from proto_tools.utils.tool_pool import ToolPool
from pydantic import BaseModel

from proto_language.constraint import ConstraintRegistry
from proto_language.core import Construct, Program, Segment
from proto_language.generator import RandomNucleotideGenerator, RandomNucleotideGeneratorConfig
from proto_language.optimizer import RejectionSamplingOptimizer, RejectionSamplingOptimizerConfig
from proto_language.utils.base import apply_device


def _program(device: str | None) -> Program:
    """Smallest program that exercises the compute decision."""
    segment = Segment(sequence="ACGT" * 5, sequence_type="dna", label="test")
    construct = Construct([segment])
    generator = RandomNucleotideGenerator(RandomNucleotideGeneratorConfig())
    generator.assign(segment)
    constraint = ConstraintRegistry.create(
        key="gc-content", segments=[segment], config_dict={"min_gc": 0, "max_gc": 100}
    )
    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=[constraint],
        config=RejectionSamplingOptimizerConfig(num_samples=2, num_results=1),
    )
    return Program(optimizers=[optimizer], num_results=1, compute=None, device=device)


class _ToolConfig(BaseModel):
    device: str = "cuda"


class _ComponentConfig(BaseModel):
    """A component whose own config has no device, only a nested tool config."""

    tool_config: _ToolConfig = _ToolConfig()
    threshold: float = 0.5


class _DirectConfig(BaseModel):
    device: str = "cuda"


def test_device_reaches_a_nested_tool_config() -> None:
    """The component a caller configures often has no device of its own.

    ``mmseqs_similarity_constraint`` is the real case: it exposes ``mmseqs_config`` and
    ``prodigal_config`` rather than a device, so a walk that stopped at the top level would
    leave the tools running locally while reporting success.
    """
    config = _ComponentConfig()
    apply_device(config, "modal")
    assert config.tool_config.device == "modal"


def test_a_direct_device_field_is_set() -> None:
    """The simple case still works."""
    config = _DirectConfig()
    apply_device(config, "modal")
    assert config.device == "modal"


def test_an_explicit_device_is_left_alone() -> None:
    """A device the caller chose outranks the program's, so one component can stay local.

    This is where device parts ways with seed: the program seed owns run determinism and
    overwrites unconditionally, but pinning one constraint to local CUDA is a legitimate ask.
    """
    config = _DirectConfig(device="cuda:1")
    apply_device(config, "modal")
    assert config.device == "cuda:1"


def test_an_unset_device_is_replaced_even_when_it_equals_the_default() -> None:
    """Not setting a device means no opinion, whatever the default happens to be."""
    config = _DirectConfig()
    assert "device" not in config.model_fields_set
    apply_device(config, "modal")
    assert config.device == "modal"


def test_dict_configs_are_walked() -> None:
    """Constraints accept dict configs, which never carry ``model_fields_set``."""
    config = {"device": "cuda", "nested": {"device": "cuda"}}
    apply_device(config, "modal")
    assert config["device"] == "modal"
    assert config["nested"]["device"] == "modal"


def test_lists_of_configs_are_walked() -> None:
    """A config field holding several tool configs must not be skipped."""
    configs = [_DirectConfig(), _DirectConfig()]
    apply_device(configs, "modal")
    assert [c.device for c in configs] == ["modal", "modal"]


def test_a_config_without_a_device_is_untouched() -> None:
    """Walking must not invent fields on models that have none."""
    config = _ComponentConfig()
    apply_device(config, "modal")
    assert not hasattr(config, "device")
    assert config.threshold == 0.5


@pytest.mark.parametrize("device", ["modal", "proto"])
def test_a_remote_program_starts_no_local_tool_pool(device: str) -> None:
    """A run whose tools execute elsewhere must not allocate local GPU workers.

    Before this, ``Program`` only skipped the pool when a dispatch backend was installed, which
    ``device='modal'`` does not do — so a fully remote run still held GPUs it never used, and
    failed outright on a CPU-only host.
    """
    program = _program(device=device)
    assert not isinstance(program.compute, ToolPool)


def test_a_local_program_still_gets_its_tool_pool() -> None:
    """The default path is unchanged: a local run needs its pool."""
    program = _program(device=None)
    assert isinstance(program.compute, ToolPool)


def test_the_device_reaches_every_optimizer() -> None:
    """Optimizers apply it to their components at run time, so they must carry it."""
    program = _program(device="modal")
    assert [opt.device for opt in program.optimizers] == ["modal"]


def test_a_generator_that_snapshots_device_is_refreshed() -> None:
    """Generators copy ``config.device`` onto the instance at construction, then read it back.

    Eight of the nine device-aware generators do this, so updating the config alone would be
    silently ignored on exactly the components a remote run most wants to move.
    """
    from proto_language.generator import ESM2Generator, ESM2GeneratorConfig

    generator = ESM2Generator(ESM2GeneratorConfig())
    generator._set_program_device("modal")
    assert generator.config.device == "modal"
    assert generator.device == "modal", "the snapshot the generator actually reads must be updated"


def test_a_generator_pinned_to_a_device_keeps_it() -> None:
    """An explicit device survives, on the snapshot as well as the config."""
    from proto_language.generator import ESM2Generator, ESM2GeneratorConfig

    generator = ESM2Generator(ESM2GeneratorConfig(device="cuda:1"))
    generator._set_program_device("modal")
    assert generator.config.device == "cuda:1"
    assert generator.device == "cuda:1"
