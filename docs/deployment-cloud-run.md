# Cloud Run deployment

## Decision

GitHub Pages is not a safe deployment target for this application. Pages can only
serve exported static files, whereas this app has dynamic Node routes:

- `/api/images/[id]` reads local image files selected from a manifest;
- `/api/inference` accepts an upload, writes a temporary directory, and launches
  Pixi/CUDA inference; and
- the server component loads local manifests and prediction shards on every
  server start/request path.

`next build` confirms this: both API routes are dynamic. The current
`output: "standalone"` setting produces a compact Node server, but it does not
turn those routes into static files. A Pages export would either fail or remove
those core behaviors. There is no existing GitHub Actions workflow or Pages
configuration in the repository.

Cloud Run is the smallest credible Google Cloud host for the web application:
it runs the standalone Next.js server and its filesystem-backed gallery routes.
Vertex AI is not a general web host. If online inference is later separated from
the app, Vertex AI can host the detector as an endpoint, while Cloud Run remains
the browser/API host and calls that endpoint.

## Included configuration

- `Dockerfile` builds `web/` and runs Next's standalone server on port 8080.
- `.dockerignore` and `.gcloudignore` exclude caches and mutable upload folders.
  A clean checkout builds an empty `/app/work`; approved gallery artifacts must
  be staged separately before a populated deployment.
- `.github/workflows/deploy-cloud-run.yml` is manual-only and uses GitHub OIDC
  Workload Identity Federation. It cannot deploy until the repository variables
  below are configured.

The files under `work/` are deliberately gitignored. The container creates an
empty `/app/work`, so the manual workflow builds successfully but shows the
application empty state until an approved artifact-delivery step is configured.
Before a real deployment, use a secure, approved artifact-delivery
mechanism (for example a restricted Cloud Storage download during the image
build) or an approved release process that stages the exact public gallery
subset into the build context. Do not add datasets, model weights, tokens, or
prediction caches to Git merely to deploy them.

## One-time Google Cloud and GitHub setup

Create a Google Cloud project and Cloud Run service in the chosen region. Create
a dedicated deploy service account with least-privilege roles sufficient to
build from source and update the selected Cloud Run service (typically Cloud Run
Admin, Cloud Build Editor, Service Account User, and Artifact Registry Writer,
scoped as tightly as practical). Configure a Workload Identity Pool/provider
that trusts this repository and grants that identity access to the deploy service
account.

Set these GitHub **Actions variables** (not secrets) before running the manual
workflow:

| Variable | Example |
| --- | --- |
| `GCP_PROJECT_ID` | `my-project` |
| `GCP_REGION` | `asia-southeast1` |
| `CLOUD_RUN_SERVICE` | `trace-lens` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/123/locations/global/workloadIdentityPools/github/providers/github` |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | `github-deployer@my-project.iam.gserviceaccount.com` |

Use a protected GitHub environment named `production` if approval is required.
The workflow has no push trigger and no credentials embedded in the repository.

## Runtime limits and follow-up

The gallery works only when its manifest, canonical images/derivatives, and
prediction shards are included in the image (or otherwise made available at
`TRACE_PROJECT_ROOT/work`). Without them, the existing application intentionally
shows its empty-gallery state.

The present inference route is not Cloud Run-ready: it hard-codes Pixi's
`linux-aarch64-cuda` platform, `cuda:0`, local Hugging Face cache, and a
filesystem-backed Python environment. Cloud Run is x86_64 and its local disk is
ephemeral. Inference therefore continues to return its existing `503` fallback
until a separately authorized inference migration changes the model/runtime
code. That migration should provision model artifacts outside Git and either
use a GPU-capable service or a Vertex AI endpoint; it is intentionally outside
this hosting-only change.

After configuring identity and staging only authorized public gallery artifacts,
run **Deploy evidence browser to Cloud Run** from the Actions tab. This operation
builds and deploys a billable Google Cloud service, so it is not run by this
repository configuration.
