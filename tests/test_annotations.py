"""Tests for beacon.annotations."""
from __future__ import annotations

import pytest

from beacon.annotations import (
    annotate,
    clear_annotations,
    collect_notes,
    get_annotations,
    note,
    pop_annotation,
    push_annotation,
)


class TestAnnotationStack:
    def setup_method(self) -> None:
        clear_annotations()

    def teardown_method(self) -> None:
        clear_annotations()

    def test_push_and_get(self) -> None:
        push_annotation("test note")
        annotations = get_annotations()
        assert "test note" in annotations

    def test_pop_returns_last(self) -> None:
        push_annotation("first")
        push_annotation("second")
        popped = pop_annotation()
        assert popped == "second"

    def test_pop_empty_returns_none(self) -> None:
        assert pop_annotation() is None

    def test_get_annotations_lifo_order(self) -> None:
        push_annotation("first")
        push_annotation("second")
        push_annotation("third")
        annotations = get_annotations()
        # get_annotations returns in reverse (most recent first)
        assert annotations[0] == "third"
        assert annotations[-1] == "first"

    def test_clear(self) -> None:
        push_annotation("a")
        push_annotation("b")
        clear_annotations()
        assert get_annotations() == []

    def test_stack_isolation_after_clear(self) -> None:
        push_annotation("a")
        clear_annotations()
        push_annotation("b")
        assert get_annotations() == ["b"]


class TestAnnotateContextManager:
    def setup_method(self) -> None:
        clear_annotations()

    def teardown_method(self) -> None:
        clear_annotations()

    def test_note_visible_inside_block(self) -> None:
        with annotate("inside note"):
            annotations = get_annotations()
            assert "inside note" in annotations

    def test_note_cleared_after_block(self) -> None:
        with annotate("inside note"):
            pass
        assert "inside note" not in get_annotations()

    def test_note_cleared_on_exception(self) -> None:
        try:
            with annotate("exception note"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert "exception note" not in get_annotations()

    def test_nested_annotate(self) -> None:
        with annotate("outer"):
            with annotate("inner"):
                annotations = get_annotations()
                assert "outer" in annotations
                assert "inner" in annotations
            # inner should be gone
            assert "inner" not in get_annotations()
            assert "outer" in get_annotations()
        assert get_annotations() == []


class TestNoteDecorator:
    def test_note_decorator_sets_attribute(self) -> None:
        @note("This is a test note")
        def my_test() -> None:
            pass

        assert hasattr(my_test, "_beacon_notes")
        assert "This is a test note" in my_test._beacon_notes

    def test_multiple_notes(self) -> None:
        @note("first note")
        @note("second note")
        def my_test() -> None:
            pass

        assert "first note" in my_test._beacon_notes
        assert "second note" in my_test._beacon_notes

    def test_decorated_function_still_callable(self) -> None:
        @note("some note")
        def my_fn(x: int) -> int:
            return x * 2

        assert my_fn(3) == 6

    def test_note_preserves_function_name(self) -> None:
        @note("test note")
        def my_named_function() -> None:
            pass

        assert my_named_function.__name__ == "my_named_function"


class TestCollectNotes:
    def setup_method(self) -> None:
        clear_annotations()

    def teardown_method(self) -> None:
        clear_annotations()

    def test_collect_from_decorator(self) -> None:
        @note("decorator note")
        def my_test() -> None:
            pass

        notes = collect_notes(my_test)
        assert any(n.message == "decorator note" for n in notes)
        assert any(n.source == "decorator" for n in notes)

    def test_collect_from_stack(self) -> None:
        push_annotation("stack note")
        notes = collect_notes(None)
        assert any(n.message == "stack note" for n in notes)
        assert any(n.source == "context_manager" for n in notes)

    def test_collect_both_sources(self) -> None:
        @note("decorator note")
        def my_test() -> None:
            pass

        push_annotation("stack note")
        notes = collect_notes(my_test)
        sources = {n.message for n in notes}
        assert "decorator note" in sources
        assert "stack note" in sources

    def test_collect_none_function(self) -> None:
        push_annotation("only stack note")
        notes = collect_notes(None)
        assert len(notes) == 1

    def test_collect_empty(self) -> None:
        notes = collect_notes(None)
        assert notes == []
