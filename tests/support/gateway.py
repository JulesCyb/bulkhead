"""One shared gateway admin fake (issue #98 / spec #90, "A6-T3"): a `GatewayAdminClient` built on
`httpx.MockTransport`, replacing the three per-file copies that used to live in
`tests/test_operator_tool_integration.py`, `tests/test_operator_erase_integration.py`, and
`tests/test_gateway_provisioning_integration.py` (a fourth, in
`tests/test_operator_create_dedicated_integration.py`, duplicated the no-delete-handling variant
of the same thing). The gateway's administrative HTTP interface is never reached over a real
network anywhere in the suite -- every one of those files, and `tests.support.seeding.seed_tenant`
(which now provisions every seeded tenant's gateway credential through the operator's own `create`
command), builds its `GatewayAdminClient` from `fake_gateway_admin_client` below instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from app.gateway_provisioning import GatewayAdminClient


@dataclass
class FakeGatewayAdmin:
    """Records every `/key/generate` and `/key/delete` request made against the client this
    backs (`GatewayAdminClient.fake`, a test-only attribute -- not part of that class's real
    contract), and can be told to fail some of them, for the partial-failure/re-run tests.

    `fail_generate`: every `/key/generate` call returns a 500 instead of minting a key.
    `fail_deletes_until`: the first this-many `/key/delete` calls return a 500, simulating a
    transient gateway failure; 0 (the default) always succeeds.
    """

    key: str = "sk-minted"
    fail_generate: bool = False
    fail_deletes_until: int = 0
    generate_requests: list[dict] = field(default_factory=list)
    delete_requests: list[dict] = field(default_factory=list)

    @property
    def generate_calls(self) -> int:
        return len(self.generate_requests)

    @property
    def delete_calls(self) -> int:
        return len(self.delete_requests)

    @property
    def deleted_keys(self) -> list[str]:
        """The credential string named by every `/key/delete` request's own `keys` list, in call
        order -- the convenience `tests/test_gateway_provisioning_integration.py`'s own fake used
        to expose as `revoke_calls`."""
        return [key for body in self.delete_requests for key in body.get("keys", [])]

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/key/generate":
            self.generate_requests.append(body)
            if self.fail_generate:
                return httpx.Response(500, text="gateway temporarily unavailable")
            return httpx.Response(200, json={"key": self.key})
        if request.url.path == "/key/delete":
            self.delete_requests.append(body)
            if self.delete_calls <= self.fail_deletes_until:
                return httpx.Response(500, text="gateway temporarily unavailable")
            return httpx.Response(200, json={"deleted_keys": body.get("keys", [])})
        return httpx.Response(404)


def fake_gateway_admin_client(
    *,
    key: str = "sk-minted",
    fail_generate: bool = False,
    fail_deletes_until: int = 0,
    base_url: str = "http://litellm.internal:4000",
    master_key: str = "sk-master-test",
) -> GatewayAdminClient:
    """A `GatewayAdminClient` over `httpx.MockTransport`, handling both `/key/generate` (`create`)
    and `/key/delete` (`erase`'s revoke step, `revoke_gateway_credential`) -- never a real network
    call. The recorder backing it is reachable as `client.fake` (a `FakeGatewayAdmin`): every
    request body it received, and how many calls of each kind it has seen.
    """
    fake = FakeGatewayAdmin(
        key=key, fail_generate=fail_generate, fail_deletes_until=fail_deletes_until
    )
    http_client = httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(fake.handle))
    client = GatewayAdminClient(base_url=base_url, master_key=master_key, http_client=http_client)
    client.fake = fake  # test-only attribute, not part of GatewayAdminClient's real contract
    return client
