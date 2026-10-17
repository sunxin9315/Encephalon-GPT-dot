import unittest
from dataclasses import replace
from encephalon.async_requests import RequestRegistry
from encephalon.contracts import AsyncRequest, AsyncResult, ContractError, Scope
from tests.helpers import SCOPE, header


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.registry = RequestRegistry(SCOPE)
        self.request = AsyncRequest(header(request_id="r", state=1), 0, 0, ("c",), {"input": 1}, 200)
        self.context = dict(state_version=1, question_version=0, premise_versions={"input": 1}, active_claim_ids={"c"})
        self.registry.register(self.request, now_ms=0, **self.context)
        self.registry.dispatch("r", 10)

    def result(self, **kwargs):
        result = AsyncResult(header(at=100, request_id="r", state=1), 0, 0, ("c",), {"input": 1}, 200, {"text": "candidate"})
        return replace(result, **kwargs)

    def test_applicable_result_and_transport_not_evidence(self):
        decision = self.registry.receive(self.result(), now_ms=100, **self.context)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "applicable_not_fact_verified")
        self.assertEqual(self.registry.status_view("r", 100)["status"], "succeeded")

    def test_unrelated_state_change_allowed_when_dependencies_unchanged(self):
        self.context["state_version"] = 3
        self.assertTrue(self.registry.receive(self.result(), now_ms=100, **self.context).accepted)

    def test_changed_premise_rejects_stale_result(self):
        self.context["premise_versions"] = {"input": 2}
        decision = self.registry.receive(self.result(), now_ms=100, **self.context)
        self.assertEqual(decision.reason, "premise_changed")

    def test_question_change_rejects(self):
        self.context["question_version"] = 1
        self.assertEqual(self.registry.receive(self.result(), now_ms=100, **self.context).reason, "question_changed")

    def test_expiry_and_unavailable_future(self):
        self.assertEqual(self.registry.receive(self.result(), now_ms=90, **self.context).reason, "not_yet_available")
        self.assertEqual(self.registry.receive(self.result(), now_ms=201, **self.context).reason, "expired")

    def test_persona_version_session_and_request_version(self):
        for scope in (Scope("other", "0.1", "s-a"), Scope("demo-a", "0.2", "s-a"), Scope("demo-a", "0.1", "s-other")):
            result = self.result(header=header(at=100, request_id="r", state=1, scope=scope))
            self.assertEqual(self.registry.receive(result, now_ms=100, **self.context).reason, "scope_mismatch")
        self.assertEqual(self.registry.receive(self.result(request_version=1), now_ms=100, **self.context).reason, "request_version_mismatch")

    def test_unrequested_inactive_claims_and_dependencies(self):
        self.assertEqual(self.registry.receive(self.result(claim_ids=("other",)), now_ms=100, **self.context).reason, "claim_inactive_or_unrequested")
        self.assertEqual(self.registry.receive(self.result(premise_versions={"input": 0}), now_ms=100, **self.context).reason, "result_premise_mismatch")
        self.context["active_claim_ids"] = set()
        self.assertFalse(self.registry.receive(self.result(), now_ms=100, **self.context).accepted)

    def test_duplicate_return_and_cancelled_request(self):
        self.assertTrue(self.registry.receive(self.result(), now_ms=100, **self.context).accepted)
        self.assertEqual(self.registry.receive(self.result(), now_ms=100, **self.context).reason, "request_not_running")
        with self.assertRaises(ContractError):
            self.registry.dispatch("r", 100)

    def test_cancel_and_failure_do_not_make_success(self):
        self.registry.fail("r", "unavailable", 20)
        self.assertEqual(self.registry.status_view("r", 30)["status"], "unavailable")
        self.assertFalse(self.registry.receive(self.result(), now_ms=100, **self.context).accepted)

    def test_no_dependency_proof_defaults_to_reject_on_changed_state(self):
        registry = RequestRegistry(SCOPE)
        request = replace(self.request, premise_versions={})
        registry.register(request, now_ms=0, **self.context)
        registry.dispatch("r", 0)
        context = {**self.context, "state_version": 2}
        decision = registry.receive(self.result(premise_versions={}), now_ms=100, **context)
        self.assertEqual(decision.reason, "state_changed_without_dependencies")

    def test_status_does_not_expose_future_completion_or_eta(self):
        status = self.registry.status_view("r", 50)
        self.assertEqual(status["elapsed_ms"], 50)
        self.assertEqual(set(status), {"request_id", "status", "elapsed_ms", "request_version", "request_state_version", "claim_ids"})

    def test_original_provenance_required(self):
        result = self.result(header=header(at=100, request_id="r", state=1, source_ids=("returned-source",)))
        self.assertEqual(self.registry.receive(result, now_ms=100, **self.context).reason, "missing_request_provenance")

    def test_request_reuse_requires_higher_version_and_terminal_old_request(self):
        with self.assertRaises(ContractError):
            self.registry.register(replace(self.request, request_version=1), now_ms=10, **self.context)
        self.registry.fail("r", "cancelled", 10)
        self.registry.register(replace(self.request, request_version=1), now_ms=10, **self.context)
        self.registry.dispatch("r", 10)
        self.assertEqual(self.registry.receive(self.result(), now_ms=100, **self.context).reason, "request_version_mismatch")
