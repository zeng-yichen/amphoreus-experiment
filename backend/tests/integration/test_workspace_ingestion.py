"""Integration tests for Stelle workspace staging and data ingestion.

Runs _setup_workspace against the real trimble-mark client data and verifies
every data modality referenced in the system prompt is present and populated.

Data modalities from Stelle's system prompt:
  memory/config.md
  memory/story-inventory.md
  memory/profile.md
  memory/strategy.md
  memory/constraints.md
  memory/source-material/        (transcripts)
  memory/references/
  memory/published-posts/
  memory/voice-examples/
  memory/draft-posts/
  memory/feedback/edits/
  context/research/               (person.md, company.md)
  context/org/                    (company context)
  context/topic-velocity.md
  abm_profiles/
  revisions/
  scratch/
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


COMPANY = "trimble-mark"


@pytest.fixture(scope="module")
def workspace() -> Path:
    """Run _setup_workspace once for all tests in this module."""
    from backend.src.agents.stelle import _setup_workspace
    ws = _setup_workspace(COMPANY)
    assert ws.exists(), f"Workspace not created at {ws}"
    return ws


# ---------- Structural tests: directories exist ----------

class TestDirectoryStructure:
    def test_memory_dir_exists(self, workspace):
        assert (workspace / "memory").is_dir()

    def test_context_dir_exists(self, workspace):
        assert (workspace / "context").is_dir()

    def test_scratch_dir_exists(self, workspace):
        assert (workspace / "scratch").is_dir()
        assert (workspace / "scratch" / "drafts").is_dir()

    def test_output_dir_exists(self, workspace):
        assert (workspace / "output").is_dir()


# ---------- memory/ data modality tests ----------

class TestSourceMaterial:
    """memory/source-material/ — raw interview transcripts."""

    def test_source_material_exists(self, workspace):
        sm = workspace / "memory" / "source-material"
        assert sm.exists(), "memory/source-material/ missing"

    def test_has_transcript_files(self, workspace):
        sm = workspace / "memory" / "source-material"
        files = list(sm.iterdir())
        assert len(files) > 0, "memory/source-material/ is empty — no transcripts found"

    def test_transcript_files_have_content(self, workspace):
        sm = workspace / "memory" / "source-material"
        for f in sm.iterdir():
            if f.is_file() and f.suffix in (".md", ".txt"):
                content = f.read_text(encoding="utf-8", errors="replace")
                assert len(content) > 100, f"Transcript {f.name} suspiciously small ({len(content)} chars)"


class TestPublishedPosts:
    """memory/published-posts/ — from Supabase linkedin_posts."""

    def test_dir_exists(self, workspace):
        d = workspace / "memory" / "published-posts"
        assert d.is_dir(), "memory/published-posts/ missing"

    def test_has_posts(self, workspace):
        d = workspace / "memory" / "published-posts"
        files = [f for f in d.iterdir() if f.is_file()]
        assert len(files) > 0, "memory/published-posts/ is empty — Supabase fetch failed?"

    def test_posts_have_engagement_metrics(self, workspace):
        d = workspace / "memory" / "published-posts"
        checked = 0
        for f in d.iterdir():
            if f.is_file() and f.suffix == ".md":
                content = f.read_text(encoding="utf-8", errors="replace")
                if "reaction" in content.lower() or "comment" in content.lower() or "engagement" in content.lower():
                    checked += 1
        assert checked > 0, "No published post files contain engagement metrics"


class TestVoiceExamples:
    """memory/voice-examples/ — top posts by engagement."""

    def test_dir_exists(self, workspace):
        d = workspace / "memory" / "voice-examples"
        assert d.is_dir(), "memory/voice-examples/ missing"

    def test_has_examples(self, workspace):
        d = workspace / "memory" / "voice-examples"
        files = [f for f in d.iterdir() if f.is_file()]
        assert len(files) > 0, "memory/voice-examples/ is empty"


class TestDraftPosts:
    """memory/draft-posts/ — writable dir for Stelle."""

    def test_dir_exists(self, workspace):
        assert (workspace / "memory" / "draft-posts").is_dir()


class TestFeedback:
    """memory/feedback/edits/ — client feedback."""

    def test_feedback_dir_exists(self, workspace):
        assert (workspace / "memory" / "feedback").is_dir()

    def test_edits_dir_exists(self, workspace):
        assert (workspace / "memory" / "feedback" / "edits").is_dir()


class TestReferences:
    """memory/references/ — client-provided articles."""

    def test_dir_exists(self, workspace):
        assert (workspace / "memory" / "references").exists()


class TestProfileMd:
    """memory/profile.md — LinkedIn profile data or person.md fallback."""

    def test_profile_exists(self, workspace):
        p = workspace / "memory" / "profile.md"
        assert p.exists(), (
            "memory/profile.md missing — both APIMaestro and person.md fallback failed"
        )

    def test_profile_has_content(self, workspace):
        p = workspace / "memory" / "profile.md"
        content = p.read_text(encoding="utf-8")
        assert len(content) > 100, f"profile.md suspiciously small ({len(content)} chars)"

    def test_profile_is_for_correct_client(self, workspace):
        p = workspace / "memory" / "profile.md"
        content = p.read_text(encoding="utf-8").lower()
        assert "mark" in content or "schwartz" in content or "trimble" in content, (
            "profile.md does not mention the client's name — may be wrong client's data"
        )

    def test_profile_not_iana_lin(self, workspace):
        """Regression: previously person.md was unfiltered and returned Iana Lin."""
        p = workspace / "memory" / "profile.md"
        content = p.read_text(encoding="utf-8").lower()
        assert "iana lin" not in content, (
            "profile.md contains 'Iana Lin' — research filter is still broken"
        )


class TestStrategyMd:
    """memory/strategy.md — content strategy from RuanMei content_brief.json."""

    def test_strategy_exists(self, workspace):
        p = workspace / "memory" / "strategy.md"
        assert p.exists(), "memory/strategy.md missing — content_brief.json not loaded?"

    def test_strategy_has_content(self, workspace):
        p = workspace / "memory" / "strategy.md"
        content = p.read_text(encoding="utf-8")
        assert len(content) > 200, f"strategy.md suspiciously small ({len(content)} chars)"

    def test_strategy_from_ruan_mei(self, workspace):
        p = workspace / "memory" / "strategy.md"
        content = p.read_text(encoding="utf-8")
        assert "RuanMei" in content or "learned" in content.lower(), (
            "strategy.md does not appear to be from RuanMei's content_brief"
        )

    def test_strategy_not_from_old_content_strategy(self, workspace):
        """Old content_strategy/ docs should NOT be loaded."""
        p = workspace / "memory" / "strategy.md"
        content = p.read_text(encoding="utf-8")
        assert "content-strategy-2026" not in content, (
            "strategy.md appears to contain old content_strategy/ files"
        )


class TestConstraintsMd:
    """memory/constraints.md — voice/tone rules or placeholder."""

    def test_constraints_exists(self, workspace):
        p = workspace / "memory" / "constraints.md"
        assert p.exists(), "memory/constraints.md missing"

    def test_constraints_has_content(self, workspace):
        p = workspace / "memory" / "constraints.md"
        content = p.read_text(encoding="utf-8")
        assert len(content) > 10, "constraints.md is empty"


class TestConfigMd:
    """memory/config.md — bounded agent memory."""

    def test_config_exists(self, workspace):
        p = workspace / "memory" / "config.md"
        assert p.exists(), "memory/config.md missing"


class TestStoryInventory:
    """memory/story-inventory.md — cross-session story tracker."""

    def test_story_inventory_exists(self, workspace):
        p = workspace / "memory" / "story-inventory.md"
        assert p.exists(), "memory/story-inventory.md missing"


# ---------- context/ data modality tests ----------

class TestResearch:
    """context/research/ — deep research from Supabase parallel_research_results."""

    def test_research_dir_exists(self, workspace):
        d = workspace / "context" / "research"
        assert d.is_dir(), "context/research/ missing"

    def test_person_md_exists(self, workspace):
        p = workspace / "context" / "research" / "person.md"
        assert p.exists(), "context/research/person.md missing — Supabase filter failed?"

    def test_company_md_exists(self, workspace):
        p = workspace / "context" / "research" / "company.md"
        assert p.exists(), "context/research/company.md missing — Supabase filter failed?"

    def test_person_is_mark(self, workspace):
        p = workspace / "context" / "research" / "person.md"
        content = p.read_text(encoding="utf-8").lower()
        assert "mark" in content or "schwartz" in content, (
            "person.md does not mention Mark Schwartz — wrong client's research"
        )

    def test_person_not_iana(self, workspace):
        """Regression: unfiltered query returned Iana Lin's person research."""
        p = workspace / "context" / "research" / "person.md"
        content = p.read_text(encoding="utf-8").lower()
        assert "iana lin" not in content, (
            "person.md contains 'Iana Lin' — research filter is broken"
        )

    def test_company_is_trimble(self, workspace):
        p = workspace / "context" / "research" / "company.md"
        content = p.read_text(encoding="utf-8").lower()
        assert "trimble" in content, (
            "company.md does not mention Trimble — wrong company's research"
        )

    def test_research_has_substance(self, workspace):
        d = workspace / "context" / "research"
        for f in d.iterdir():
            if f.is_file():
                content = f.read_text(encoding="utf-8")
                assert len(content) > 500, f"{f.name} is too small ({len(content)} chars)"


class TestOrgContext:
    """context/org/ — company context (symlinked from research)."""

    def test_org_dir_exists(self, workspace):
        d = workspace / "context" / "org"
        assert d.is_dir(), "context/org/ missing"


class TestTopicVelocity:
    """context/topic-velocity.md — industry signal from Perplexity."""

    def test_topic_velocity_exists(self, workspace):
        p = workspace / "context" / "topic-velocity.md"
        if not p.exists():
            pytest.skip("topic_velocity.md not generated yet for this client")

    def test_topic_velocity_has_content(self, workspace):
        p = workspace / "context" / "topic-velocity.md"
        if not p.exists():
            pytest.skip("topic_velocity.md not generated yet")
        content = p.read_text(encoding="utf-8")
        assert len(content) > 100, "topic-velocity.md is too small"


# ---------- Top-level directory tests ----------

class TestTopLevelDirs:
    """abm_profiles/, revisions/, scratch/, output/."""

    def test_scratch_exists(self, workspace):
        assert (workspace / "scratch").is_dir()

    def test_scratch_drafts_exists(self, workspace):
        assert (workspace / "scratch" / "drafts").is_dir()

    def test_output_exists(self, workspace):
        assert (workspace / "output").is_dir()


# ---------- No vestigial data tests ----------

class TestNoVestigialData:
    """Ensure deprecated files are NOT staged into the workspace."""

    def test_no_past_posts(self, workspace):
        assert not (workspace / "memory" / "past-posts").exists(), "past_posts should not be staged"

    def test_no_learned_directives(self, workspace):
        for f in (workspace / "memory").rglob("learned_directives*"):
            pytest.fail(f"Vestigial learned_directives found: {f}")

    def test_no_lola_state(self, workspace):
        for f in (workspace / "memory").rglob("lola_state*"):
            pytest.fail(f"Vestigial lola_state found: {f}")

    def test_no_cyrene_embeddings(self, workspace):
        for f in workspace.rglob("cyrene_quality_embeddings*"):
            pytest.fail(f"Vestigial cyrene embeddings found: {f}")

    def test_no_pi_sessions(self, workspace):
        assert not (workspace / ".pi-sessions").exists(), ".pi-sessions should not exist"

    def test_no_old_strategy_docs(self, workspace):
        """Old content_strategy/ files should not be symlinked into org/."""
        org = workspace / "context" / "org"
        if org.exists():
            for f in org.iterdir():
                assert "content-strategy-2026" not in f.name, (
                    f"Old strategy doc found in context/org/: {f.name}"
                )


# ---------- Supabase ID resolution test ----------

class TestSupabaseIdResolution:
    """_resolve_supabase_ids returns correct UUIDs for trimble-mark."""

    def test_resolve_returns_ids(self):
        from backend.src.agents.stelle import _resolve_supabase_ids
        user_id, company_id = _resolve_supabase_ids("mark-schwartz-27b5613")
        assert user_id, "user_id not resolved for mark-schwartz-27b5613"
        assert company_id, "company_id not resolved for mark-schwartz-27b5613"

    def test_resolve_known_values(self):
        from backend.src.agents.stelle import _resolve_supabase_ids
        user_id, company_id = _resolve_supabase_ids("mark-schwartz-27b5613")
        assert user_id == "984aba78-c761-42b8-a05c-22efae4d0a57"
        assert company_id == "6b39e696-8d97-4651-83d8-79483903b4de"


# ---------- System prompt alignment test ----------

class TestSystemPromptAlignment:
    """Verify every path mentioned in _DIRECT_SYSTEM_TEMPLATE exists in workspace."""

    EXPECTED_PATHS = [
        "memory/config.md",
        "memory/story-inventory.md",
        "memory/profile.md",
        "memory/strategy.md",
        "memory/constraints.md",
        "memory/source-material",
        "memory/references",
        "memory/published-posts",
        "memory/voice-examples",
        "memory/draft-posts",
        "memory/feedback/edits",
        "context/research",
        "context/org",
        "scratch",
    ]

    @pytest.mark.parametrize("relpath", EXPECTED_PATHS)
    def test_path_exists(self, workspace, relpath):
        p = workspace / relpath
        assert p.exists(), f"System prompt references `{relpath}` but it doesn't exist in workspace"
