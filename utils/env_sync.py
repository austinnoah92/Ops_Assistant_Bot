from pathlib import Path

def sync_env_to_secrets():
    """
    Automatically sync .env to .streamlit/secrets.toml
    whenever they are out of sync. Runs on every app start.
    On Streamlit Cloud where .env does not exist, returns immediately.
    """
    env_path = Path('.env')
    secrets_dir = Path('.streamlit')
    secrets_path = secrets_dir / 'secrets.toml'

    # Only run locally — on Streamlit Cloud .env does not exist
    if not env_path.exists():
        return

    # Read current .env contents
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value

    # Read current secrets.toml contents if it exists
    secrets_vars = {}
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    secrets_vars[key] = value

    # If already in sync do nothing
    if env_vars == secrets_vars:
        return

    # Out of sync — rewrite secrets.toml
    secrets_dir.mkdir(exist_ok=True)
    lines = [f'{k} = "{v}"' for k, v in env_vars.items()]
    with open(secrets_path, 'w') as f:
        f.write('\n'.join(lines))