# ai-app-starter

Multi-tenant backend for AI agents: several customers share one deployment, each with their own
users and data, and agents reach that data only through tools.

## Language

### Tenancy

**Tenant**:
A customer organization: the contracting party. One customer, one tenant. The tenant is the
boundary for data isolation, billing, and deletion; departments, subsidiaries, or test areas of
a customer are structure inside the tenant, never a second tenant.
_Avoid_: customer, client, account, organization, workspace

**Isolation tier**:
The degree of data separation a tenant is entitled to: *pooled* (shares a database with other
tenants) or *dedicated* (its own database). Every tenant has exactly one tier.
_Avoid_: silo, pool model, physical separation, own instance

**Residency**:
The jurisdiction in which a tenant's content may be processed and stored, set per tenant. Every
path that carries content out of the process (model, embeddings, tracing) honours it; a tenant's
data never crosses it.
_Avoid_: region (as a synonym), data location, hosting, EU flag

**Budget**:
How much model usage a tenant may consume in a period, enforced outside the application at the
gateway with credentials issued per tenant. An exhausted budget stops that tenant, never the
deployment.
_Avoid_: quota, spend cap, plan limit, rate limit (a budget is not a rate)

**Run limit**:
The ceiling of model requests, tool calls, and wall-clock time a single agent run may consume,
enforced inside the application. It protects against loops, including those a poisoned document provokes, not
against cost.
_Avoid_: usage limit (the library's term), timeout, budget

**Operator**:
The party that runs the deployment and owns the control plane: creates, suspends, and erases
tenants, holds the owner database role and the gateway master credential. Never a tenant.
_Avoid_: platform admin, superuser, root, provider

**Suspension**:
A tenant state in which no request is served and nothing is deleted. Entered by the operator,
reversible, and always the step before erasure.
_Avoid_: deactivation, lock, freeze, soft delete

**Erasure**:
The removal of a tenant from every place its data lives: databases, gateway credentials,
conversations, traces, and, after the backup horizon has passed, backups. Produces a record of
what was removed where.
_Avoid_: deletion (as the whole), purge, offboarding, cleanup

**Control plane**:
The operator's own records about tenants and identities: which exist, their isolation tier,
residency, where their data lives. Owned by the operator, never by a tenant; a tenant's request
can read what it needs of it and change nothing.
_Avoid_: admin data, master data, tenant catalog

**Tenant secret**:
A secret the operator holds and uses on one tenant's behalf: the gateway credential, access to a
dedicated database. Delivered to the deployment, never stored in a database, and rotated by the
operator with the tenant's admins told.
_Avoid_: API key (as a generic), config value, credential (that word is for agent identities)

### Identity

**Identity**:
A person or an agent as authenticated by an identity provider, known by issuer and subject.
Global: an identity exists once and is not owned by any tenant.
_Avoid_: user, account, login, principal

**Membership**:
The relation of one identity to one tenant, carrying that identity's role in the tenant.
Everything a tenant knows about a person is its membership; a person in two tenants has two
memberships and one identity.
_Avoid_: user, user account, seat, tenant user

**Role**:
What a membership may do inside its tenant. Exactly four: *admin* (manages members, settings,
and deletion), *member* (uses the tenant's content and agents), *support* (an operator's
membership for troubleshooting, always visible to the tenant's admins), *agent* (an agent
identity's membership; reading tools only unless the tenant's admin grants more). Roles never
decide what a member can see, only what it can do.
_Avoid_: permission, group, scope, owner

### Agents

**Tool**:
The only way an agent reaches data or acts, always in the context of a membership. A *reading*
tool returns what is needed and nothing more; a *writing* tool changes something outside the
conversation and runs only after an approval or under a standing grant.
_Avoid_: function, action, skill, plugin, integration

**Agent**:
A program that answers questions or acts through tools. Interactively it acts by delegation;
autonomously it acts through an agent identity.
_Avoid_: bot, assistant (as a concept; "assistant" is only the name of the default agent), copilot

**Delegation**:
An agent acting with the membership of the person who asked. The audit trail records both: the
person as the actor, the agent as the means.
_Avoid_: impersonation, on-behalf-of (in prose), service call

**Agent identity**:
An identity that belongs to an agent rather than a person, with its own membership and the role
*agent*, and credentials issued per tenant and revocable by the tenant's admin. Used only when no
person is present, such as scheduled jobs or a customer's own automation.
_Avoid_: service account, API user, bot user, machine user, global key

**Approval**:
A member's confirmation of one specific writing action in their own conversation, bound to the
exact arguments they saw and valid for a short time. There is no "always allow".
_Avoid_: consent, confirmation dialog, permission, allow-list

**Standing grant**:
A tenant admin's permission for one agent identity to use one writing tool without a person
present. Listed for the tenant's admins, revocable, and the only way autonomous work can write.
_Avoid_: auto-approve, always allow, whitelist, scope

**Identity provider**:
The external system that authenticates an identity and issues its tokens. A tenant may bring its
own; whether that is required is still open.
_Avoid_: IdP (in prose), auth server, SSO

### Content

**Document**:
A piece of a tenant's content that agents may search on behalf of members. Every member of the
tenant can see every document of the tenant; there is no per-member visibility.
_Avoid_: file, record, private document, shared document

**Conversation**:
One member's exchange with an agent inside a tenant, kept by the server as the only trusted
record of what was said, searched, and approved. A client contributes new messages, never the
past. Kept for the tenant's retention period and deleted with the tenant.
_Avoid_: chat history, thread, session, transcript
