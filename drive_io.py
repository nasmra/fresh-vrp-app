# drive_io.py
import io
from typing import Optional
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# Scopes et MIME par défaut pour Excel
SCOPES = ["https://www.googleapis.com/auth/drive"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_SERVICE = None  # cache du service Drive


def _get_folder_id() -> str:
    try:
        return st.secrets["drive"]["folder_id"]
    except Exception:
        raise RuntimeError(
            "Le secrets [drive].folder_id est introuvable. "
            "Ajoute ton folder_id dans .streamlit/secrets.toml sous la section [drive]."
        )


def _get_service():
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE

    try:
        info = st.secrets["gcp_service_account"]
    except Exception:
        raise RuntimeError(
            "Le secrets [gcp_service_account] est introuvable. "
            "Colle le JSON du compte de service dans .streamlit/secrets.toml sous [gcp_service_account]."
        )

    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    _SERVICE = build("drive", "v3", credentials=creds)
    return _SERVICE


def _find_file_id_by_name(filename: str, folder_id: Optional[str] = None) -> Optional[str]:
    """Retourne l'ID du premier fichier portant ce nom dans le dossier donné (non supprimé)."""
    service = _get_service()
    folder_id = folder_id or _get_folder_id()

    # Attention à l’échappement des quotes pour le nom
    name_escaped = filename.replace("'", "\\'")
    query = f"name = '{name_escaped}' and '{folder_id}' in parents and trashed = false"

    resp = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        pageSize=10
    ).execute()
    files = resp.get("files", [])
    if not files:
        return None
    return files[0]["id"]


def drive_download(filename: str) -> bytes:
    """
    Télécharge le fichier `filename` (dans le dossier Drive configuré) et renvoie son contenu en bytes.
    """
    service = _get_service()
    folder_id = _get_folder_id()
    file_id = _find_file_id_by_name(filename, folder_id)
    if not file_id:
        raise FileNotFoundError(f"Fichier introuvable sur Drive dans ce dossier: {filename}")

    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        # (optionnel) progression: status.progress()

    buf.seek(0)
    return buf.getvalue()


def drive_upload(filename: str, data: bytes, mime_type: str = XLSX_MIME) -> str:
    """
    Uploade (ou met à jour si déjà existant) `filename` dans le dossier Drive configuré.
    Renvoie l'ID du fichier.
    """
    service = _get_service()
    folder_id = _get_folder_id()

    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)
    file_id = _find_file_id_by_name(filename, folder_id)

    try:
        if file_id:
            # Met à jour le contenu du fichier existant
            updated = service.files().update(
                fileId=file_id,
                media_body=media,
                body={"name": filename},
                fields="id, name"
            ).execute()
            return updated["id"]
        else:
            # Crée le fichier s’il n’existe pas
            created = service.files().create(
                media_body=media,
                body={"name": filename, "parents": [folder_id]},
                fields="id, name"
            ).execute()
            return created["id"]
    except HttpError as e:
        raise RuntimeError(f"Echec upload/update Drive pour {filename}: {e}")
