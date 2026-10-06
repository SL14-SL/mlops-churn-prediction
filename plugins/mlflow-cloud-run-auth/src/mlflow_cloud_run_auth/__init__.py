import os
from threading import Lock
from urllib.parse import urlsplit

import google.auth
from google.auth import impersonated_credentials
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from mlflow.tracking.request_auth.abstract_request_auth_provider import (
    RequestAuthProvider,
)
from requests.auth import AuthBase


class CloudRunAuth(AuthBase):
    """Attach renewable identity tokens only to the configured MLflow origin."""

    def __init__(self):
        audience = os.environ["MLFLOW_CLOUD_RUN_AUDIENCE"].rstrip("/")
        parsed = urlsplit(audience)

        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".run.app")
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise ValueError(
                "MLFLOW_CLOUD_RUN_AUDIENCE must be a Cloud Run HTTPS service URL without a path."
            )

        self._origin = (parsed.scheme, parsed.hostname, parsed.port or 443)
        self._request = Request()
        self._lock = Lock()

        service_account = os.getenv("MLFLOW_AUTH_SERVICE_ACCOUNT")

        if service_account:
            source_credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            target_credentials = impersonated_credentials.Credentials(
                source_credentials=source_credentials,
                target_principal=service_account,
                target_scopes=[
                    "https://www.googleapis.com/auth/cloud-platform",
                ],
            )
            self._credentials = impersonated_credentials.IDTokenCredentials(
                target_credentials,
                target_audience=audience,
                include_email=True,
            )
        else:
            self._credentials = id_token.fetch_id_token_credentials(
                audience,
                request=self._request,
            )

    def __call__(self, request):
        parsed = urlsplit(request.url)
        origin = (parsed.scheme, parsed.hostname, parsed.port or 443)

        if origin != self._origin:
            raise ValueError("Refusing to send a Cloud Run identity token to another origin.")

        with self._lock:
            if not self._credentials.valid:
                self._credentials.refresh(self._request)

            request.headers["X-Serverless-Authorization"] = f"Bearer {self._credentials.token}"

        return request


class CloudRunAuthProvider(RequestAuthProvider):
    """MLflow authentication plugin for a private Cloud Run tracking server."""

    def __init__(self):
        self._auth = None
        self._lock = Lock()

    def get_name(self):
        return "cloud_run"

    def get_auth(self):
        with self._lock:
            if self._auth is None:
                self._auth = CloudRunAuth()

            return self._auth
