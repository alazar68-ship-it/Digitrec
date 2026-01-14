from __future__ import annotations

import os
from pathlib import Path


# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent  # repo root


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


SECRET_KEY = _env("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    # Fejlesztői fallback: nem production célra.
    SECRET_KEY = "dev-insecure-secret-key-change-me"

DEBUG = _env("DJANGO_DEBUG", "0") not in {"0", "false", "False"}

ALLOWED_HOSTS = [h.strip() for h in _env("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "digits",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "digitrec_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "digitrec_web.wsgi.application"
ASGI_APPLICATION = "digitrec_web.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": _env("DJANGO_SQLITE_PATH", str(REPO_ROOT / "digitrec.sqlite3")),
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "hu-hu"
TIME_ZONE = "Europe/Budapest"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
#STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "static"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Digit recognition model artifact directory
DIGITREC_ARTIFACT_DIR = Path(_env("DIGITREC_ARTIFACT_DIR", str(REPO_ROOT / "artifacts"))).resolve()
ALLOWED_HOSTS = [
    "lazarsoft.hu",
    "www.lazarsoft.hu",
]
